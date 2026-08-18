from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, NamedTuple

from psycopg2 import sql as psql

from langchain_gaussdb.errors import GaussDBFilterError, GaussDBSQLBuildError
from langchain_gaussdb.metadata_json import dumps_metadata_json
from langchain_gaussdb.metadata_sql import (
    DATE_FORMAT,
    metadata_literal_index_selector,
    metadata_literal_json_selector,
    metadata_literal_json_text_selector,
    metadata_literal_text_selector,
    normalize_metadata_index_cast,
)
from langchain_gaussdb.sql import identifier

_COMPARISON_OPERATORS = {
    "$eq": "=",
    "$ne": "<>",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
_RANGE_OPERATORS = {"$gt", "$gte", "$lt", "$lte"}
_MEMBERSHIP_OPERATORS = {"$in", "$nin"}
_TEXT_OPERATORS = {"$like": "LIKE", "$ilike": "ILIKE"}
_LOGICAL_OPERATORS = {"$and", "$or", "$not"}
_SUPPORTED_FIELD_OPERATORS = (
    set(_COMPARISON_OPERATORS)
    | _MEMBERSHIP_OPERATORS
    | set(_TEXT_OPERATORS)
    | {"$between", "$contains", "$exists"}
)
_RANGE_KINDS = {"text", "int", "float", "date", "timestamp", "time"}


@dataclass(frozen=True)
class CompiledFilter:
    sql: psql.Composable
    params: tuple[Any, ...]
    fields: tuple[str, ...] = ()


class _ScalarType(NamedTuple):
    kind: str
    cast: str | None


class _MetadataIndex(NamedTuple):
    selector: psql.Composable
    cast: str | None


def compile_metadata_filter(
    filters: dict[str, Any] | None,
    *,
    metadata_column: str,
    metadata_indexes: Mapping[str, str | None] | None = None,
) -> CompiledFilter | None:
    if filters is None or filters == {}:
        return None
    if not isinstance(filters, dict):
        raise GaussDBFilterError("metadata filter must be a dictionary")
    try:
        metadata_identifier = identifier(metadata_column, label="metadata_column")
    except GaussDBSQLBuildError as exc:
        raise GaussDBFilterError("metadata filter metadata_column is invalid") from exc
    return _compile_node(filters, metadata_identifier, metadata_indexes or {})


def _compile_node(
    filters: dict[str, Any],
    metadata_column: psql.Composable,
    metadata_indexes: Mapping[str, str | None],
    *,
    preserve_json_null_unknown: bool = False,
) -> CompiledFilter:
    if not isinstance(filters, dict):
        raise GaussDBFilterError("metadata filter condition must be a dictionary")
    if not filters:
        raise GaussDBFilterError("metadata filter condition must not be empty")

    if len(filters) == 1:
        key, value = next(iter(filters.items()))
        if not isinstance(key, str):
            raise GaussDBFilterError("metadata filter field must be a non-empty string")
        if key.startswith("$"):
            return _compile_logical_operator(
                key,
                value,
                metadata_column,
                metadata_indexes,
                preserve_json_null_unknown=preserve_json_null_unknown,
            )
        return _compile_field_filter(
            key,
            value,
            metadata_column,
            metadata_indexes,
            preserve_json_null_unknown=preserve_json_null_unknown,
        )

    children: list[CompiledFilter] = []
    for key, value in filters.items():
        if not isinstance(key, str):
            raise GaussDBFilterError("metadata filter field must be a non-empty string")
        if key.startswith("$"):
            raise GaussDBFilterError(
                f"metadata filter operator {key} cannot be mixed with fields"
            )
        children.append(
            _compile_field_filter(
                key,
                value,
                metadata_column,
                metadata_indexes,
                preserve_json_null_unknown=preserve_json_null_unknown,
            )
        )
    return _join_children("AND", children)


def _compile_logical_operator(
    operator: str,
    value: Any,
    metadata_column: psql.Composable,
    metadata_indexes: Mapping[str, str | None],
    *,
    preserve_json_null_unknown: bool,
) -> CompiledFilter:
    if operator not in _LOGICAL_OPERATORS:
        raise GaussDBFilterError(
            f"unsupported metadata filter field/operator: {operator}"
        )

    if operator in {"$and", "$or"}:
        if not isinstance(value, list):
            raise GaussDBFilterError(f"{operator} metadata filter value must be a list")
        if not value:
            raise GaussDBFilterError(
                f"{operator} metadata filter list must not be empty"
            )
        children = [
            _compile_node(
                item,
                metadata_column,
                metadata_indexes,
                preserve_json_null_unknown=preserve_json_null_unknown,
            )
            for item in value
        ]
        return _join_children("AND" if operator == "$and" else "OR", children)

    if not isinstance(value, dict):
        raise GaussDBFilterError("$not metadata filter value must be a dictionary")
    child = _compile_node(
        value,
        metadata_column,
        metadata_indexes,
        preserve_json_null_unknown=True,
    )
    # GaussDB can mis-evaluate NOT when it is applied directly to a JSONB
    # containment predicate.  A simple CASE performs the same three-valued
    # negation without duplicating the child SQL or its parameters.
    return CompiledFilter(
        psql.SQL(
            "(CASE ({}) WHEN TRUE THEN FALSE WHEN FALSE THEN TRUE "
            "ELSE NULL::boolean END)"
        ).format(child.sql),
        child.params,
        child.fields,
    )


def _compile_field_filter(
    field: str,
    condition: Any,
    metadata_column: psql.Composable,
    metadata_indexes: Mapping[str, str | None],
    *,
    preserve_json_null_unknown: bool,
) -> CompiledFilter:
    _validate_field_name(field)

    if isinstance(condition, dict):
        if len(condition) != 1:
            raise GaussDBFilterError(
                "metadata filter field condition must contain a single operator"
            )
        operator, value = next(iter(condition.items()))
        if operator not in _SUPPORTED_FIELD_OPERATORS:
            raise GaussDBFilterError(
                f"unsupported metadata filter operator: {operator}"
            )
    else:
        operator = "$eq"
        value = condition

    if operator == "$contains":
        return _compile_contains(field, value, metadata_column)
    if operator == "$exists":
        return _compile_exists(field, value, metadata_column)
    if operator in _TEXT_OPERATORS:
        return _compile_text_operator(
            field,
            operator,
            value,
            metadata_column,
            _metadata_index_selector(field, metadata_column, metadata_indexes),
        )
    if operator == "$between":
        return _compile_between(
            field,
            value,
            metadata_column,
            _metadata_index_selector(field, metadata_column, metadata_indexes),
        )
    if operator in _MEMBERSHIP_OPERATORS:
        return _compile_membership(
            field,
            operator,
            value,
            metadata_column,
            _metadata_index_selector(field, metadata_column, metadata_indexes),
        )
    return _compile_comparison(
        field,
        operator,
        value,
        metadata_column,
        _metadata_index_selector(field, metadata_column, metadata_indexes),
        preserve_json_null_unknown=preserve_json_null_unknown,
    )


def _validate_field_name(field: str) -> None:
    if not isinstance(field, str) or not field:
        raise GaussDBFilterError("metadata filter field must be a non-empty string")
    if field.startswith("$"):
        raise GaussDBFilterError("metadata filter field must not start with $")
    if "\0" in field:
        raise GaussDBFilterError("metadata filter field must not contain NUL")


def _compile_comparison(
    field: str,
    operator: str,
    value: Any,
    metadata_column: psql.Composable,
    metadata_index: _MetadataIndex | None,
    *,
    preserve_json_null_unknown: bool,
) -> CompiledFilter:
    if metadata_index is not None and operator in {"$eq", "$ne"}:
        return _compile_indexed_comparison(
            field,
            operator,
            value,
            metadata_column,
            metadata_index,
            preserve_json_null_unknown=preserve_json_null_unknown,
        )

    if (
        metadata_index is None
        and operator in {"$eq", "$ne"}
        and not (isinstance(value, dt.date) and not isinstance(value, dt.datetime))
    ):
        return _compile_jsonb_comparison(
            field,
            operator,
            value,
            metadata_column,
            preserve_json_null_unknown=preserve_json_null_unknown,
        )

    scalar_type = (
        _infer_indexed_scalar_type(value, metadata_index.cast)
        if metadata_index is not None and metadata_index.cast is not None
        else _infer_scalar_type(value)
    )
    if operator in _RANGE_OPERATORS and scalar_type.kind not in _RANGE_KINDS:
        raise GaussDBFilterError(
            f"metadata filter operator {operator} does not support {scalar_type.kind}"
        )

    use_json_text_parameter = scalar_type.kind == "text" and metadata_index is None
    selector = (
        metadata_literal_text_selector(metadata_column, field)
        if use_json_text_parameter
        else (
            metadata_index.selector
            if metadata_index is not None and metadata_index.cast is not None
            else _selector(metadata_column, field, scalar_type)
        )
    )
    placeholder = (
        _json_text_placeholder()
        if use_json_text_parameter
        else _placeholder(scalar_type)
    )
    parameter = (
        _json_text_parameter(value)
        if use_json_text_parameter
        else (
            _indexed_parameter(metadata_index, value, scalar_type)
            if metadata_index is not None and metadata_index.cast is not None
            else _parameter(value, scalar_type)
        )
    )
    native_operator = _COMPARISON_OPERATORS[operator]
    statement = psql.SQL("({} {} {})").format(
        selector,
        psql.SQL(native_operator),
        placeholder,
    )
    if metadata_index is not None and metadata_index.cast == "text":
        statement = psql.SQL("(jsonb_typeof({}) = 'string' AND {})").format(
            metadata_literal_json_selector(metadata_column, field),
            statement,
        )
    params = (parameter,)
    return CompiledFilter(statement, params, (field,))


def _compile_indexed_comparison(
    field: str,
    operator: str,
    value: Any,
    metadata_column: psql.Composable,
    metadata_index: _MetadataIndex,
    *,
    preserve_json_null_unknown: bool,
) -> CompiledFilter:
    if metadata_index.cast in {None, "text"}:
        if metadata_index.cast == "text":
            _infer_indexed_scalar_type(value, "text")
        json_selector = metadata_literal_json_selector(metadata_column, field)
        selector = (
            psql.SQL("(NULLIF({}, 'null'::jsonb))::text").format(json_selector)
            if preserve_json_null_unknown and operator == "$eq"
            else metadata_index.selector
        )
        parameter = _json_parameter(value)
        if operator == "$eq":
            statement = psql.SQL("({} = (%s::jsonb)::text)").format(selector)
            params: tuple[Any, ...] = (parameter,)
        else:
            type_guard = (
                psql.SQL("jsonb_typeof({}) = 'string'").format(json_selector)
                if metadata_index.cast == "text"
                else psql.SQL("{} <> 'null'::jsonb").format(json_selector)
            )
            statement = psql.SQL("({} ? %s AND {} AND {} <> (%s::jsonb)::text)").format(
                metadata_column,
                type_guard,
                selector,
            )
            params = (field, parameter)
        return CompiledFilter(statement, params, (field,))

    cast = metadata_index.cast
    assert cast is not None
    scalar_type = _infer_indexed_scalar_type(value, cast)
    statement = psql.SQL("({} {} {})").format(
        metadata_index.selector,
        psql.SQL(_COMPARISON_OPERATORS[operator]),
        _indexed_placeholder(metadata_index, scalar_type),
    )
    return CompiledFilter(
        statement,
        (_parameter(value, scalar_type),),
        (field,),
    )


def _compile_membership(
    field: str,
    operator: str,
    value: Any,
    metadata_column: psql.Composable,
    metadata_index: _MetadataIndex | None,
) -> CompiledFilter:
    if not isinstance(value, (list, tuple)):
        raise GaussDBFilterError(f"metadata filter {operator} value must be a list")
    values = list(value)
    if not values:
        return CompiledFilter(
            psql.SQL("FALSE" if operator == "$in" else "TRUE"),
            (),
            (field,),
        )

    if metadata_index is not None and metadata_index.cast in {None, "text"}:
        if metadata_index.cast == "text":
            _infer_homogeneous_indexed_scalar_type(values, "text")
        return _compile_indexed_json_membership(
            field,
            operator,
            values,
            metadata_column,
            metadata_index,
        )

    scalar_type = (
        _infer_homogeneous_indexed_scalar_type(values, metadata_index.cast)
        if metadata_index is not None and metadata_index.cast is not None
        else _infer_homogeneous_scalar_type(values)
    )

    if metadata_index is None and scalar_type.kind not in {
        "date",
        "timestamp",
        "time",
    }:
        return _compile_jsonb_membership(
            field,
            operator,
            values,
            metadata_column,
        )

    selector = (
        metadata_index.selector
        if metadata_index is not None
        else _selector(metadata_column, field, scalar_type)
    )
    parameters = _parameters(values, scalar_type)
    if operator == "$in":
        statement = psql.SQL("({} = ANY({}))").format(
            selector,
            (
                _indexed_membership_placeholder(metadata_index, scalar_type)
                if metadata_index is not None
                else _membership_placeholder(scalar_type)
            ),
        )
        params = (parameters,)
    elif metadata_index is not None:
        statement = psql.SQL("({} <> ALL({}))").format(
            selector,
            _indexed_membership_placeholder(metadata_index, scalar_type),
        )
        params = (parameters,)
    else:
        statement = psql.SQL("(NOT ({} = ANY({})))").format(
            selector,
            _membership_placeholder(scalar_type),
        )
        params = (parameters,)
    return CompiledFilter(statement, params, (field,))


def _compile_indexed_json_membership(
    field: str,
    operator: str,
    values: list[Any],
    metadata_column: psql.Composable,
    metadata_index: _MetadataIndex,
) -> CompiledFilter:
    parameters = [_json_parameter(value) for value in values]
    normalized_parameters = psql.SQL(
        "ARRAY(SELECT (json_value::jsonb)::text FROM unnest(%s::text[]) AS json_value)"
    )
    if operator == "$in":
        statement = psql.SQL("({} = ANY({}))").format(
            metadata_index.selector,
            normalized_parameters,
        )
        params: tuple[Any, ...] = (parameters,)
    else:
        json_selector = metadata_literal_json_selector(metadata_column, field)
        type_guard = (
            psql.SQL("jsonb_typeof({}) = 'string'").format(json_selector)
            if metadata_index.cast == "text"
            else psql.SQL("{} <> 'null'::jsonb").format(json_selector)
        )
        statement = psql.SQL("({} ? %s AND {} AND NOT ({} = ANY({})))").format(
            metadata_column,
            type_guard,
            metadata_index.selector,
            normalized_parameters,
        )
        params = (field, parameters)
    return CompiledFilter(statement, params, (field,))


def _compile_between(
    field: str,
    value: Any,
    metadata_column: psql.Composable,
    metadata_index: _MetadataIndex | None,
) -> CompiledFilter:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GaussDBFilterError(
            "metadata filter $between value must contain two values"
        )
    low, high = value
    try:
        scalar_type = (
            _infer_homogeneous_indexed_scalar_type([low, high], metadata_index.cast)
            if metadata_index is not None and metadata_index.cast is not None
            else _infer_homogeneous_scalar_type([low, high])
        )
    except GaussDBFilterError as exc:
        raise GaussDBFilterError(
            "metadata filter $between values must have the same supported type"
        ) from exc
    if scalar_type.kind not in _RANGE_KINDS:
        raise GaussDBFilterError(
            f"metadata filter $between does not support {scalar_type.kind}"
        )
    declared_index = (
        metadata_index
        if metadata_index is not None and metadata_index.cast is not None
        else None
    )
    use_json_text_parameter = scalar_type.kind == "text" and declared_index is None
    placeholder = (
        _json_text_placeholder()
        if use_json_text_parameter
        else (
            _indexed_placeholder(declared_index, scalar_type)
            if declared_index is not None
            else _placeholder(scalar_type)
        )
    )
    statement = psql.SQL("({} BETWEEN {} AND {})").format(
        (
            metadata_literal_text_selector(metadata_column, field)
            if use_json_text_parameter
            else (
                declared_index.selector
                if declared_index is not None
                else _selector(metadata_column, field, scalar_type)
            )
        ),
        placeholder,
        placeholder,
    )
    if declared_index is not None and declared_index.cast == "text":
        statement = psql.SQL("(jsonb_typeof({}) = 'string' AND {})").format(
            metadata_literal_json_selector(metadata_column, field),
            statement,
        )
    params = (
        (_json_text_parameter(low), _json_text_parameter(high))
        if use_json_text_parameter
        else (
            _indexed_parameter(declared_index, low, scalar_type),
            _indexed_parameter(declared_index, high, scalar_type),
        )
        if declared_index is not None
        else (_parameter(low, scalar_type), _parameter(high, scalar_type))
    )
    return CompiledFilter(
        statement,
        params,
        (field,),
    )


def _compile_text_operator(
    field: str,
    operator: str,
    value: Any,
    metadata_column: psql.Composable,
    metadata_index: _MetadataIndex | None,
) -> CompiledFilter:
    if not isinstance(value, str):
        raise GaussDBFilterError(f"metadata filter {operator} value must be a string")
    if metadata_index is not None and metadata_index.cast not in {None, "text"}:
        raise GaussDBFilterError(
            f"metadata filter {operator} requires a text metadata index"
        )
    declared_index = (
        metadata_index
        if metadata_index is not None and metadata_index.cast == "text"
        else None
    )
    selector = (
        declared_index.selector
        if declared_index is not None
        else metadata_literal_text_selector(metadata_column, field)
    )
    statement = psql.SQL("({} {} {})").format(
        selector,
        psql.SQL(_TEXT_OPERATORS[operator]),
        psql.SQL("%s") if declared_index is not None else _json_text_placeholder(),
    )
    params = (
        (_json_parameter(value),)
        if declared_index is not None
        else (_json_text_parameter(value),)
    )
    return CompiledFilter(statement, params, (field,))


def _compile_contains(
    field: str,
    value: Any,
    metadata_column: psql.Composable,
) -> CompiledFilter:
    json_value = _json_contains_value(field, value)
    statement = psql.SQL("({} @> %s::jsonb)").format(metadata_column)
    return CompiledFilter(statement, (json_value,), (field,))


def _compile_exists(
    field: str,
    value: Any,
    metadata_column: psql.Composable,
) -> CompiledFilter:
    if not isinstance(value, bool):
        raise GaussDBFilterError("$exists metadata filter value must be a boolean")
    statement = psql.SQL("({} ? %s)" if value else "(NOT ({} ? %s))").format(
        metadata_column
    )
    return CompiledFilter(statement, (field,), (field,))


def _metadata_index_selector(
    field: str,
    metadata_column: psql.Composable,
    metadata_indexes: Mapping[str, str | None],
) -> _MetadataIndex | None:
    if field not in metadata_indexes:
        return None
    try:
        cast = normalize_metadata_index_cast(metadata_indexes[field])
    except ValueError as exc:
        raise GaussDBFilterError(str(exc)) from None
    return _MetadataIndex(
        metadata_literal_index_selector(metadata_column, field, cast),
        cast,
    )


def _compile_jsonb_comparison(
    field: str,
    operator: str,
    value: Any,
    metadata_column: psql.Composable,
    *,
    preserve_json_null_unknown: bool,
) -> CompiledFilter:
    params: tuple[Any, ...]
    # JSONB numeric equality deliberately stays native: GaussDB treats 5 and
    # 5.0 as the same JSON number, while a text-normalized comparison does not.
    # Callers that need an indexable numeric predicate declare bigint/float in
    # metadata_indexes, which compiles both DDL and filters to the same cast.
    if _is_json_number(value):
        json_selector = metadata_literal_json_selector(metadata_column, field)
        selector = (
            psql.SQL("NULLIF({}, 'null'::jsonb)").format(json_selector)
            if preserve_json_null_unknown and operator == "$eq"
            else json_selector
        )
        parameter = _json_parameter(value)
        if operator == "$eq":
            statement = psql.SQL("({} = %s::jsonb)").format(selector)
            params = (parameter,)
        else:
            statement = psql.SQL(
                "({} ? %s AND {} <> 'null'::jsonb AND {} <> %s::jsonb)"
            ).format(metadata_column, json_selector, selector)
            params = (field, parameter)
        return CompiledFilter(statement, params, (field,))

    if operator == "$eq":
        json_selector = metadata_literal_json_selector(metadata_column, field)
        selector = (
            psql.SQL("(NULLIF({}, 'null'::jsonb))::text").format(json_selector)
            if preserve_json_null_unknown
            else metadata_literal_json_text_selector(metadata_column, field)
        )
        parameter = _json_parameter(value)
        statement = psql.SQL("({} = (%s::jsonb)::text)").format(selector)
        params = (parameter,)
    elif isinstance(value, str):
        json_selector = metadata_literal_json_selector(metadata_column, field)
        selector = metadata_literal_json_text_selector(metadata_column, field)
        statement = psql.SQL(
            "({} ? %s AND jsonb_typeof({}) = 'string' AND {} <> (%s::jsonb)::text)"
        ).format(metadata_column, json_selector, selector)
        params = (field, _json_parameter(value))
    else:
        json_selector = metadata_literal_json_selector(metadata_column, field)
        selector = metadata_literal_json_text_selector(metadata_column, field)
        parameter = _json_parameter(value)
        statement = psql.SQL(
            "({} ? %s AND {} <> 'null'::jsonb AND {} <> (%s::jsonb)::text)"
        ).format(metadata_column, json_selector, selector)
        params = (field, parameter)
    return CompiledFilter(statement, params, (field,))


def _compile_jsonb_membership(
    field: str,
    operator: str,
    values: list[Any],
    metadata_column: psql.Composable,
) -> CompiledFilter:
    parameters = [_json_parameter(value) for value in values]
    numeric_values = all(_is_json_number(value) for value in values)
    text_values = all(isinstance(value, str) for value in values)
    selector = (
        metadata_literal_json_selector(metadata_column, field)
        if numeric_values
        else metadata_literal_json_text_selector(metadata_column, field)
    )
    normalized_parameters = psql.SQL(
        "ARRAY(SELECT json_value::jsonb FROM unnest(%s::text[]) AS json_value)"
        if numeric_values
        else "ARRAY(SELECT (json_value::jsonb)::text "
        "FROM unnest(%s::text[]) AS json_value)"
    )

    if operator == "$in":
        statement = psql.SQL("({} = ANY({}))").format(
            selector,
            normalized_parameters,
        )
        params: tuple[Any, ...] = (parameters,)
    elif text_values:
        json_selector = metadata_literal_json_selector(metadata_column, field)
        statement = psql.SQL(
            "({} ? %s AND jsonb_typeof({}) = 'string' AND NOT ({} = ANY({})))"
        ).format(
            metadata_column,
            json_selector,
            selector,
            normalized_parameters,
        )
        params = (field, parameters)
    else:
        json_selector = metadata_literal_json_selector(metadata_column, field)
        statement = psql.SQL(
            "({} ? %s AND {} <> 'null'::jsonb AND NOT ({} = ANY({})))"
        ).format(
            metadata_column,
            json_selector,
            selector,
            normalized_parameters,
        )
        params = (field, parameters)
    return CompiledFilter(statement, params, (field,))


def _json_contains_value(field: str, value: Any) -> str:
    try:
        return dumps_metadata_json({field: value})
    except (TypeError, ValueError) as exc:
        raise GaussDBFilterError(
            "metadata filter $contains value must be JSON serializable"
        ) from exc


def _json_parameter(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        raise GaussDBFilterError("metadata filter float value must be finite")
    try:
        return dumps_metadata_json(value)
    except (TypeError, ValueError) as exc:
        raise GaussDBFilterError(
            "metadata filter value must be JSON serializable"
        ) from exc


def _json_text_placeholder() -> psql.Composable:
    return psql.SQL("(%s::jsonb->>0)")


def _json_text_parameter(value: str) -> str:
    return dumps_metadata_json([value])


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _selector(
    metadata_column: psql.Composable,
    field: str,
    scalar_type: _ScalarType,
) -> psql.Composable:
    base = metadata_literal_text_selector(metadata_column, field)
    if scalar_type.kind == "date":
        return psql.SQL("to_date({}, '{}')").format(
            base,
            psql.SQL(DATE_FORMAT),
        )
    if scalar_type.cast is None:
        return base
    return psql.SQL(f"({{}})::{scalar_type.cast}").format(base)


def _placeholder(scalar_type: _ScalarType) -> psql.Composable:
    if scalar_type.kind == "date":
        return psql.SQL(f"to_date(%s, '{DATE_FORMAT}')")
    return psql.SQL("%s")


def _indexed_placeholder(
    metadata_index: _MetadataIndex,
    scalar_type: _ScalarType,
) -> psql.Composable:
    if metadata_index.cast == "date":
        return psql.SQL("%s")
    return _placeholder(scalar_type)


def _indexed_parameter(
    metadata_index: _MetadataIndex,
    value: Any,
    scalar_type: _ScalarType,
) -> Any:
    if metadata_index.cast == "text":
        return _json_parameter(value)
    return _parameter(value, scalar_type)


def _membership_placeholder(scalar_type: _ScalarType) -> psql.Composable:
    if scalar_type.kind == "date":
        return psql.SQL(
            "ARRAY(SELECT to_date(date_value, "
            f"'{DATE_FORMAT}') FROM unnest(%s::text[]) AS date_value)"
        )
    if scalar_type.kind == "timestamp":
        return psql.SQL("%s::timestamp[]")
    if scalar_type.kind == "time":
        return psql.SQL("%s::time[]")
    return psql.SQL("%s")


def _indexed_membership_placeholder(
    metadata_index: _MetadataIndex,
    scalar_type: _ScalarType,
) -> psql.Composable:
    if metadata_index.cast == "date":
        return psql.SQL("%s")
    return _membership_placeholder(scalar_type)


def _parameter(value: Any, scalar_type: _ScalarType) -> Any:
    if scalar_type.kind == "date":
        if isinstance(value, str):
            return value
        return value.isoformat()
    if scalar_type.kind == "timestamp":
        if isinstance(value, str):
            return dt.datetime.fromisoformat(value)
        return value
    if scalar_type.kind == "time":
        if isinstance(value, str):
            return dt.time.fromisoformat(value)
        return value
    return value


def _parameters(values: Iterable[Any], scalar_type: _ScalarType) -> list[Any]:
    if scalar_type.kind == "date":
        return [
            value if isinstance(value, str) else value.isoformat() for value in values
        ]
    return [_parameter(value, scalar_type) for value in values]


def _infer_homogeneous_scalar_type(values: Iterable[Any]) -> _ScalarType:
    values_list = list(values)
    if not values_list:
        raise GaussDBFilterError("metadata filter values must not be empty")
    scalar_types = [_infer_scalar_type(value) for value in values_list]
    kinds = {scalar_type.kind for scalar_type in scalar_types}
    if len(kinds) == 1:
        return scalar_types[0]

    # A Python temporal value supplies enough context to interpret ISO strings in
    # the same membership list.  An all-string list remains text by design.
    for temporal_kind, parser in (
        ("timestamp", dt.datetime.fromisoformat),
        ("date", dt.date.fromisoformat),
        ("time", dt.time.fromisoformat),
    ):
        if kinds == {"text", temporal_kind}:
            try:
                for value in values_list:
                    if isinstance(value, str):
                        parser(value)
            except ValueError as exc:
                raise GaussDBFilterError(
                    "metadata filter list values must have the same type"
                ) from exc
            return _ScalarType(temporal_kind, temporal_kind)

    raise GaussDBFilterError("metadata filter list values must have the same type")


def _infer_homogeneous_indexed_scalar_type(
    values: Iterable[Any],
    cast: str,
) -> _ScalarType:
    values_list = list(values)
    if not values_list:
        raise GaussDBFilterError("metadata filter values must not be empty")
    scalar_type = _infer_indexed_scalar_type(values_list[0], cast)
    for value in values_list[1:]:
        other_type = _infer_indexed_scalar_type(value, cast)
        if other_type.kind != scalar_type.kind:
            raise GaussDBFilterError(
                "metadata filter list values must have the same type"
            )
    return scalar_type


def _infer_indexed_scalar_type(value: Any, cast: str) -> _ScalarType:
    if cast == "text":
        if not isinstance(value, str):
            raise GaussDBFilterError(
                "text metadata index filters require string values"
            )
        return _ScalarType("text", None)
    if cast == "bigint":
        if not isinstance(value, int) or isinstance(value, bool):
            raise GaussDBFilterError("bigint metadata index filters require int values")
        return _ScalarType("int", "bigint")
    if cast == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise GaussDBFilterError(
                "float metadata index filters require numeric values"
            )
        if not math.isfinite(float(value)):
            raise GaussDBFilterError("metadata filter float value must be finite")
        return _ScalarType("float", "float")
    if cast == "boolean":
        if not isinstance(value, bool):
            raise GaussDBFilterError(
                "boolean metadata index filters require bool values"
            )
        return _ScalarType("bool", "boolean")
    if cast == "date":
        if isinstance(value, str):
            try:
                dt.date.fromisoformat(value)
            except ValueError as exc:
                raise GaussDBFilterError(
                    "date metadata index filters require ISO date values"
                ) from exc
            return _ScalarType("date", "date")
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return _ScalarType("date", "date")
        raise GaussDBFilterError("date metadata index filters require date values")
    raise GaussDBFilterError(f"unsupported metadata index cast: {cast}")


def _infer_scalar_type(value: Any) -> _ScalarType:
    if isinstance(value, bool):
        return _ScalarType("bool", "boolean")
    if isinstance(value, int):
        return _ScalarType("int", "bigint")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GaussDBFilterError("metadata filter float value must be finite")
        return _ScalarType("float", "float")
    if isinstance(value, dt.datetime):
        return _ScalarType("timestamp", "timestamp")
    if isinstance(value, dt.date):
        return _ScalarType("date", "date")
    if isinstance(value, dt.time):
        return _ScalarType("time", "time")
    if isinstance(value, str):
        return _ScalarType("text", None)
    raise GaussDBFilterError(
        f"metadata filter value type is not supported: {type(value).__name__}"
    )


def _join_children(operator: str, children: list[CompiledFilter]) -> CompiledFilter:
    if not children:
        raise GaussDBFilterError("metadata filter logical condition must not be empty")
    parts = [psql.SQL("({})").format(child.sql) for child in children]
    statement = psql.SQL(f" {operator} ").join(parts)
    params: list[Any] = []
    fields: list[str] = []
    for child in children:
        params.extend(child.params)
        fields.extend(child.fields)
    return CompiledFilter(statement, tuple(params), _unique(fields))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
