from __future__ import annotations

import copy
import datetime as dt
import json
import math
import re

import pytest
from psycopg2 import sql as psql

from langchain_gaussdb import GaussDBFilterError
from langchain_gaussdb.filters import compile_metadata_filter


def _sql_repr(compiled) -> str:
    return repr(compiled.sql)


@pytest.mark.parametrize("filter_value", [None, {}])
def test_compile_metadata_filter_treats_empty_filter_as_noop(filter_value):
    assert compile_metadata_filter(filter_value, metadata_column="metadata") is None


@pytest.mark.parametrize("filter_value", [[], "source = 'kb'", 1])
def test_compile_metadata_filter_rejects_non_dict_filter(filter_value):
    with pytest.raises(GaussDBFilterError, match="filter"):
        compile_metadata_filter(filter_value, metadata_column="metadata")


@pytest.mark.parametrize(
    "filter_value,expected_params,expected_token,uses_jsonb_scalar",
    [
        ({"source": "kb"}, ('"kb"',), "::text", True),
        ({"source": {"$eq": "kb"}}, ('"kb"',), "::text", True),
        ({"page": 7}, ("7",), "::jsonb", True),
        ({"score": {"$gte": 0.8}}, (0.8,), "::float", False),
        ({"flag": True}, ("true",), "::text", True),
        (
            {"created_at": dt.datetime(2026, 7, 9, 12, 30)},
            ('"2026-07-09T12:30:00"',),
            "::text",
            True,
        ),
        (
            {"start_time": dt.time(9, 30)},
            ('"09:30:00"',),
            "::text",
            True,
        ),
    ],
)
def test_compile_metadata_filter_scalar_selectors(
    filter_value, expected_params, expected_token, uses_jsonb_scalar
):
    compiled = compile_metadata_filter(filter_value, metadata_column="metadata")

    statement = _sql_repr(compiled)
    assert ("->>" not in statement) is uses_jsonb_scalar
    assert expected_token in statement
    assert compiled.params == expected_params


@pytest.mark.parametrize(
    "filter_value,operator",
    [
        ({"publish_date": dt.date(2026, 7, 9)}, "="),
        ({"publish_date": {"$ne": dt.date(2026, 7, 9)}}, "<>"),
    ],
)
def test_compile_metadata_filter_python_date_equality_uses_date_conversion(
    filter_value,
    operator,
):
    compiled = compile_metadata_filter(filter_value, metadata_column="metadata")

    statement = _sql_repr(compiled)
    assert statement.count("to_date") == 2
    assert "YYYY-MM-DD" in statement
    assert "->>" in statement
    assert operator in statement
    assert " ? " not in statement
    assert compiled.params == ("2026-07-09",)


def test_compile_metadata_filter_datetime_is_not_treated_as_date():
    compiled = compile_metadata_filter(
        {"created_at": {"$gte": dt.datetime(2026, 7, 9, 12, 30)}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "::timestamp" in statement
    assert "::date" not in statement


def test_compile_metadata_filter_python_date_range_uses_date_conversion():
    compiled = compile_metadata_filter(
        {"publish_date": {"$gte": dt.date(2026, 7, 9)}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert statement.count("to_date") == 2
    assert "YYYY-MM-DD" in statement
    assert "->>" in statement
    assert "::date" not in statement
    assert compiled.params == ("2026-07-09",)


def test_compile_metadata_filter_string_date_range_remains_text():
    compiled = compile_metadata_filter(
        {"publish_date": {"$gte": "2026-07-09"}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "to_date" not in statement
    assert "->>" in statement
    assert compiled.params == ('["2026-07-09"]',)


def test_compile_metadata_filter_bool_equality_uses_jsonb_boolean():
    compiled = compile_metadata_filter(
        {"flag": {"$eq": False}}, metadata_column="metadata"
    )

    statement = _sql_repr(compiled)
    assert "::text" in statement
    assert "::boolean" not in statement
    assert "::bigint" not in statement
    assert compiled.params == ("false",)


def test_compile_metadata_filter_rejects_non_finite_float():
    with pytest.raises(GaussDBFilterError, match="finite"):
        compile_metadata_filter({"score": math.nan}, metadata_column="metadata")


def test_compile_metadata_filter_does_not_put_values_in_sql():
    malicious_value = "kb'); DROP TABLE documents; --"

    compiled = compile_metadata_filter(
        {"source": malicious_value},
        metadata_column="metadata",
    )

    assert json.dumps(malicious_value) in compiled.params
    assert malicious_value not in _sql_repr(compiled)


def test_compile_metadata_filter_numeric_equality_preserves_json_number_semantics():
    compiled = compile_metadata_filter(
        {"scientific": {"$eq": 1e-7}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "%s::jsonb" in statement
    assert "::text" not in statement
    assert compiled.params == ("1e-07",)


def test_compile_metadata_filter_not_numeric_equality_preserves_json_null_unknown():
    compiled = compile_metadata_filter(
        {"$not": {"shape": {"$eq": 5}}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "CASE" in statement
    assert "WHEN TRUE THEN FALSE" in statement
    assert "WHEN FALSE THEN TRUE" in statement
    assert "ELSE NULL::boolean" in statement
    assert "NULLIF" in statement
    assert "'null'::jsonb" in statement
    assert compiled.params == ("5",)


def test_compile_metadata_filter_scalar_equality_uses_metadata_index_expression():
    compiled = compile_metadata_filter(
        {"source": {"$eq": "kb"}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "NULLIF" not in statement
    assert "->" in statement
    assert "::text" in statement
    assert "::jsonb" in statement
    assert compiled.params == ('"kb"',)


def test_compile_metadata_filter_not_equality_preserves_json_null_as_unknown():
    compiled = compile_metadata_filter(
        {"$not": {"source": {"$eq": "kb"}}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "CASE" in statement
    assert "WHEN TRUE THEN FALSE" in statement
    assert "WHEN FALSE THEN TRUE" in statement
    assert "ELSE NULL::boolean" in statement
    assert "NULLIF" in statement
    assert "'null'::jsonb" in statement
    assert compiled.params == ('"kb"',)


def test_compile_metadata_filter_normalizes_json_membership_in_database():
    compiled = compile_metadata_filter(
        {"scientific": {"$in": [1e-7, 1e20, -0.0]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "unnest" in statement
    assert "%s::text[]" in statement
    assert "::jsonb" in statement
    assert "::text FROM" not in statement
    assert compiled.params == (["1e-07", "1e+20", "-0.0"],)


def test_compile_metadata_filter_empty_string_equality_uses_jsonb_scalar():
    compiled = compile_metadata_filter({"empty": ""}, metadata_column="metadata")

    statement = _sql_repr(compiled)
    assert "@>" not in statement
    assert "->>" not in statement
    assert "::text" in statement
    assert compiled.params == ('""',)


def test_compile_metadata_filter_empty_string_ne_uses_json_text_null_semantics():
    compiled = compile_metadata_filter(
        {"empty": {"$ne": ""}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "->" in statement
    assert "::text" in statement
    assert "<>" in statement
    assert " ? " in statement
    assert "jsonb_typeof" in statement
    assert "@>" not in statement
    assert compiled.params == ("empty", '""')


def test_compile_metadata_filter_including_empty_string_membership_uses_canonical_json():
    compiled = compile_metadata_filter(
        {"empty": {"$in": ["", "x"]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "@>" not in statement
    assert "ANY" in statement
    assert "::text" in statement
    assert compiled.params == (['""', '"x"'],)


def test_compile_metadata_filter_empty_string_nin_uses_json_text_null_semantics():
    compiled = compile_metadata_filter(
        {"empty": {"$nin": [""]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "->" in statement
    assert "::text" in statement
    assert "NOT" in statement
    assert "ANY" in statement
    assert " ? " in statement
    assert "jsonb_typeof" in statement
    assert "@>" not in statement
    assert compiled.params == ("empty", ['""'])


@pytest.mark.parametrize(
    "field",
    ["$source", "", 1, None],
)
def test_compile_metadata_filter_rejects_invalid_field_names(field):
    with pytest.raises(GaussDBFilterError, match="field"):
        compile_metadata_filter({field: "kb"}, metadata_column="metadata")


@pytest.mark.parametrize("field", ["source.name", "source-name", "1source"])
def test_compile_metadata_filter_allows_parameterized_jsonb_keys(field):
    compiled = compile_metadata_filter({field: "kb"}, metadata_column="metadata")

    statement = _sql_repr(compiled)
    assert "->>" not in statement
    assert "::text" in statement
    assert repr(psql.Literal(field)) in statement
    assert compiled.params == ('"kb"',)


def test_compile_metadata_filter_rejects_nul_in_jsonb_key():
    with pytest.raises(GaussDBFilterError, match="NUL"):
        compile_metadata_filter({"bad\0key": "kb"}, metadata_column="metadata")


def test_compile_metadata_filter_rejects_non_string_top_level_operator():
    with pytest.raises(GaussDBFilterError, match="field"):
        compile_metadata_filter({1: "kb", "source": "docs"}, metadata_column="metadata")


@pytest.mark.parametrize(
    "exists_value, expected_sql_token",
    [(True, " ? %s"), (False, "NOT")],
)
def test_compile_metadata_filter_supports_exists_on_jsonb(
    exists_value, expected_sql_token
):
    compiled = compile_metadata_filter(
        {"source": {"$exists": exists_value}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "->>" not in statement
    assert " ? %s" in statement
    assert expected_sql_token in statement
    assert compiled.params == ("source",)


def test_compile_metadata_filter_rejects_non_bool_exists_value():
    with pytest.raises(GaussDBFilterError, match=r"\$exists"):
        compile_metadata_filter(
            {"source": {"$exists": "yes"}}, metadata_column="metadata"
        )


def test_compile_metadata_filter_rejects_multi_operator_dict():
    with pytest.raises(GaussDBFilterError, match="single"):
        compile_metadata_filter(
            {"score": {"$gte": 0.5, "$lte": 0.9}},
            metadata_column="metadata",
        )


def test_compile_metadata_filter_text_ne_uses_json_text_null_semantics():
    compiled = compile_metadata_filter(
        {"source": {"$ne": "kb"}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "->" in statement
    assert "::text" in statement
    assert " ? " in statement
    assert "jsonb_typeof" in statement
    assert "@>" not in statement
    assert "<>" in statement
    assert compiled.params == ("source", '"kb"')


@pytest.mark.parametrize(
    "operator,expected_token",
    [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ],
)
def test_compile_metadata_filter_range_operators(operator, expected_token):
    compiled = compile_metadata_filter(
        {"score": {operator: 0.8}},
        metadata_column="metadata",
    )

    assert expected_token in _sql_repr(compiled)
    assert compiled.params == (0.8,)


def test_compile_metadata_filter_between():
    compiled = compile_metadata_filter(
        {"score": {"$between": [0.6, 0.9]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "BETWEEN" in statement
    assert compiled.params == (0.6, 0.9)


def test_compile_metadata_filter_python_date_between_converts_selector_and_bounds():
    compiled = compile_metadata_filter(
        {"publish_date": {"$between": [dt.date(2026, 7, 1), dt.date(2026, 7, 31)]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert statement.count("to_date") == 3
    assert "YYYY-MM-DD" in statement
    assert "->>" in statement
    assert "BETWEEN" in statement
    assert compiled.params == ("2026-07-01", "2026-07-31")


def test_compile_metadata_filter_string_date_between_remains_text():
    compiled = compile_metadata_filter(
        {"publish_date": {"$between": ["2026-07-01", "2026-07-31"]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "to_date" not in statement
    assert "->>" in statement
    assert "BETWEEN" in statement
    assert compiled.params == ('["2026-07-01"]', '["2026-07-31"]')


@pytest.mark.parametrize(
    "value",
    [
        [1],
        [1, 2, 3],
        [1, "2"],
        [True, False],
    ],
)
def test_compile_metadata_filter_rejects_invalid_between(value):
    with pytest.raises(GaussDBFilterError, match="between"):
        compile_metadata_filter(
            {"score": {"$between": value}}, metadata_column="metadata"
        )


def test_compile_metadata_filter_in_list():
    compiled = compile_metadata_filter(
        {"topic": {"$in": ["a", "b"]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "ANY" in statement
    assert "::text" in statement
    assert compiled.params == (['"a"', '"b"'],)


def test_compile_metadata_filter_text_nin_uses_json_text_null_semantics():
    compiled = compile_metadata_filter(
        {"topic": {"$nin": ["a", "b"]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "->" in statement
    assert "::text" in statement
    assert " ? " in statement
    assert "jsonb_typeof" in statement
    assert "@>" not in statement
    assert "NOT" in statement
    assert "ANY" in statement
    assert compiled.params == ("topic", ['"a"', '"b"'])


@pytest.mark.parametrize(
    "operator,expected_token",
    [
        ("$in", "FALSE"),
        ("$nin", "TRUE"),
    ],
)
def test_compile_metadata_filter_empty_membership_lists(operator, expected_token):
    compiled = compile_metadata_filter(
        {"topic": {operator: []}},
        metadata_column="metadata",
    )

    assert expected_token in _sql_repr(compiled)
    assert compiled.params == ()


@pytest.mark.parametrize("value", [["a", 1], [True, 1], [dt.date(2026, 7, 9), "x"]])
def test_compile_metadata_filter_rejects_mixed_membership_types(value):
    with pytest.raises(GaussDBFilterError, match="same type"):
        compile_metadata_filter({"topic": {"$in": value}}, metadata_column="metadata")


def test_compile_metadata_filter_bool_membership_uses_jsonb_booleans():
    compiled = compile_metadata_filter(
        {"flag": {"$in": [True, False]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "::text" in statement
    assert "::boolean" not in statement
    assert "::bigint" not in statement
    assert compiled.params == (["true", "false"],)


@pytest.mark.parametrize(
    "operator,value,expected_params",
    [
        ("$ne", True, ("flag", "true")),
        ("$nin", [True], ("flag", ["true"])),
    ],
)
def test_compile_metadata_filter_bool_negative_preserves_json_type(
    operator,
    value,
    expected_params,
):
    compiled = compile_metadata_filter(
        {"flag": {operator: value}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "->>" not in statement
    assert " ? " in statement
    assert "'null'" in statement
    assert compiled.params == expected_params


@pytest.mark.parametrize("operator", ["$in", "$nin"])
def test_compile_metadata_filter_python_date_membership_uses_date_conversion(
    operator,
):
    compiled = compile_metadata_filter(
        {"publish_date": {operator: [dt.date(2026, 7, 1), dt.date(2026, 7, 31)]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert statement.count("to_date") == 2
    assert "YYYY-MM-DD" in statement
    assert "to_char" not in statement
    assert "->>" in statement
    assert "ANY" in statement
    assert "unnest" in statement
    assert "::text[]" in statement
    assert compiled.params == (["2026-07-01", "2026-07-31"],)


@pytest.mark.parametrize("operator", ["$like", "$ilike"])
def test_compile_metadata_filter_text_matching(operator):
    compiled = compile_metadata_filter(
        {"title": {operator: "%GaussDB%"}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert operator[1:].upper() in statement
    assert "->>0" in statement
    assert compiled.params == ('["%GaussDB%"]',)


@pytest.mark.parametrize("operator", ["$like", "$ilike"])
def test_compile_metadata_filter_empty_text_matching_preserves_json_empty_string(
    operator,
):
    compiled = compile_metadata_filter(
        {"title": {operator: ""}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "->>0" in statement
    assert compiled.params == ('[""]',)


@pytest.mark.parametrize(
    ("condition", "expected_params"),
    [
        ({"$gt": ""}, ('[""]',)),
        ({"$between": ["", "z"]}, ('[""]', '["z"]')),
    ],
)
def test_compile_metadata_filter_text_ranges_preserve_json_empty_string(
    condition,
    expected_params,
):
    compiled = compile_metadata_filter(
        {"title": condition},
        metadata_column="metadata",
    )

    assert "->>0" in _sql_repr(compiled)
    assert compiled.params == expected_params


def test_compile_metadata_filter_text_range_matches_declared_expression_index():
    compiled = compile_metadata_filter(
        {"status": {"$gte": ""}},
        metadata_column="metadata",
        metadata_indexes={"status": "text"},
    )

    statement = _sql_repr(compiled)
    assert "Identifier('status')" not in statement
    assert "Literal('status')" in statement
    assert "SQL('->')" in statement
    assert "SQL(')::text')" in statement
    assert "jsonb_typeof" in statement
    assert compiled.params == ('""',)


def test_compile_metadata_filter_text_between_guards_json_null_for_declared_index():
    compiled = compile_metadata_filter(
        {"status": {"$between": ["", "z"]}},
        metadata_column="metadata",
        metadata_indexes={"status": "text"},
    )

    statement = _sql_repr(compiled)
    assert "BETWEEN" in statement
    assert "jsonb_typeof" in statement
    assert "Literal('status')" in statement
    assert compiled.params == ('""', '"z"')


@pytest.mark.parametrize("operator", ["$like", "$ilike"])
def test_compile_metadata_filter_text_matching_requires_string(operator):
    with pytest.raises(GaussDBFilterError, match="string"):
        compile_metadata_filter({"title": {operator: 1}}, metadata_column="metadata")


@pytest.mark.parametrize(
    "contains_value,expected_json",
    [
        (["diskann"], '{"tags":["diskann"]}'),
        ({"primary": "manual"}, '{"tags":{"primary":"manual"}}'),
        (None, '{"tags":null}'),
    ],
)
def test_compile_metadata_filter_contains(contains_value, expected_json):
    compiled = compile_metadata_filter(
        {"tags": {"$contains": contains_value}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "@>" in statement
    assert "::jsonb" in statement
    assert compiled.params == (expected_json,)


def test_compile_metadata_filter_contains_normalizes_gaussdb_boolean_for_not():
    compiled = compile_metadata_filter(
        {"$not": {"tags": {"$contains": ["red"]}}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert statement.count("CASE") == 1
    assert statement.count(" @> ") == 1
    assert "WHEN TRUE THEN FALSE" in statement
    assert "WHEN FALSE THEN TRUE" in statement
    assert "ELSE NULL::boolean" in statement
    assert compiled.params == ('{"tags":["red"]}',)


def test_compile_metadata_filter_rejects_unserializable_contains_value():
    with pytest.raises(GaussDBFilterError, match="JSON"):
        compile_metadata_filter(
            {"tags": {"$contains": object()}},
            metadata_column="metadata",
        )


def test_compile_metadata_filter_multi_field_dict_uses_and():
    compiled = compile_metadata_filter(
        {"tenant": "t1", "source": "kb"},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert " AND " in statement
    assert compiled.params == ('"t1"', '"kb"')


def test_compile_metadata_filter_and_or_not():
    compiled = compile_metadata_filter(
        {
            "$and": [
                {"tenant": "t1"},
                {"$or": [{"source": "kb"}, {"$not": {"source": "draft"}}]},
            ]
        },
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert " AND " in statement
    assert " OR " in statement
    assert "CASE" in statement
    assert compiled.params == (
        '"t1"',
        '"kb"',
        '"draft"',
    )


def test_compile_metadata_filter_nested_not_preserves_three_valued_logic():
    compiled = compile_metadata_filter(
        {"$not": {"$not": {"source": {"$eq": "kb"}}}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert statement.count("CASE") == 2
    assert statement.count("WHEN TRUE THEN FALSE") == 2
    assert statement.count("WHEN FALSE THEN TRUE") == 2
    assert statement.count("ELSE NULL::boolean") == 2
    assert "NULLIF" in statement
    assert compiled.params == ('"kb"',)


@pytest.mark.parametrize("operator", ["$and", "$or"])
def test_compile_metadata_filter_rejects_empty_logical_lists(operator):
    with pytest.raises(GaussDBFilterError, match=re.escape(operator)):
        compile_metadata_filter({operator: []}, metadata_column="metadata")


@pytest.mark.parametrize("operator", ["$and", "$or"])
def test_compile_metadata_filter_rejects_logical_non_lists(operator):
    with pytest.raises(GaussDBFilterError, match=re.escape(operator)):
        compile_metadata_filter(
            {operator: {"tenant": "t1"}}, metadata_column="metadata"
        )


def test_compile_metadata_filter_rejects_not_non_dict():
    with pytest.raises(GaussDBFilterError, match=r"\$not"):
        compile_metadata_filter(
            {"$not": [{"tenant": "t1"}]}, metadata_column="metadata"
        )


def test_compile_metadata_filter_rejects_unknown_top_level_operator():
    with pytest.raises(GaussDBFilterError, match="operator"):
        compile_metadata_filter(
            {"$exists": {"tenant": True}}, metadata_column="metadata"
        )


def test_compile_metadata_filter_does_not_mutate_input():
    original = {
        "$and": [
            {"tenant": "t1"},
            {"score": {"$between": [0.6, 0.9]}},
        ]
    }
    snapshot = copy.deepcopy(original)

    compile_metadata_filter(original, metadata_column="metadata")

    assert original == snapshot


def test_compile_metadata_filter_routes_declared_metadata_index_to_json_expression():
    compiled = compile_metadata_filter(
        {"tenant_id": "t1"},
        metadata_column="metadata",
        metadata_indexes={"tenant_id": "text"},
    )

    statement = _sql_repr(compiled)
    assert "Identifier('metadata')" in statement
    assert "Literal('tenant_id')" in statement
    assert "SQL('->')" in statement
    assert "SQL(')::text')" in statement
    assert compiled.params == ('"t1"',)


def test_compile_metadata_filter_keeps_exists_on_jsonb_for_indexed_key():
    compiled = compile_metadata_filter(
        {"tenant_id": {"$exists": True}},
        metadata_column="metadata",
        metadata_indexes={"tenant_id": "text"},
    )

    statement = _sql_repr(compiled)
    assert "Identifier('metadata')" in statement
    assert "Identifier('tenant_id')" not in statement
    assert " ? %s" in statement
    assert compiled.params == ("tenant_id",)


def test_compile_metadata_filter_empty_string_uses_text_index_expression():
    compiled = compile_metadata_filter(
        {"tenant_id": ""},
        metadata_column="metadata",
        metadata_indexes={"tenant_id": "text"},
    )

    statement = _sql_repr(compiled)
    assert "@>" not in statement
    assert "SQL('->')" in statement
    assert "SQL(')::text')" in statement
    assert "Identifier('tenant_id')" not in statement
    assert compiled.params == ('""',)


def test_compile_metadata_filter_text_index_ne_uses_same_expression():
    compiled = compile_metadata_filter(
        {"tenant_id": {"$ne": "kb"}},
        metadata_column="metadata",
        metadata_indexes={"tenant_id": "text"},
    )

    statement = _sql_repr(compiled)
    assert "SQL('->')" in statement
    assert "SQL(')::text')" in statement
    assert " ? " in statement
    assert "@>" not in statement
    assert "Identifier('metadata')" in statement
    assert "Identifier('tenant_id')" not in statement
    assert "<>" in statement
    assert compiled.params == ("tenant_id", '"kb"')


def test_compile_metadata_filter_text_index_nin_uses_same_expression():
    compiled = compile_metadata_filter(
        {"tenant_id": {"$nin": ["kb", "docs"]}},
        metadata_column="metadata",
        metadata_indexes={"tenant_id": "text"},
    )

    statement = _sql_repr(compiled)
    assert "SQL('->')" in statement
    assert "SQL(')::text')" in statement
    assert " ? " in statement
    assert "jsonb_typeof" in statement
    assert "@>" not in statement
    assert "Identifier('metadata')" in statement
    assert "Identifier('tenant_id')" not in statement
    assert "NOT" in statement
    assert "ANY" in statement
    assert compiled.params == ("tenant_id", ['"kb"', '"docs"'])


def test_compile_metadata_filter_bigint_index_ne_uses_json_expression():
    compiled = compile_metadata_filter(
        {"rank": {"$ne": 20}},
        metadata_column="metadata",
        metadata_indexes={"rank": "bigint"},
    )

    statement = _sql_repr(compiled)
    assert "Literal('rank')" in statement
    assert "->>" in statement
    assert "SQL('bigint')" in statement
    assert "<>" in statement
    assert "ALL" not in statement
    assert " ? " not in statement
    assert compiled.params == (20,)


def test_compile_metadata_filter_bigint_index_nin_uses_json_expression():
    compiled = compile_metadata_filter(
        {"rank": {"$nin": [20]}},
        metadata_column="metadata",
        metadata_indexes={"rank": "bigint"},
    )

    statement = _sql_repr(compiled)
    assert "Literal('rank')" in statement
    assert "->>" in statement
    assert "SQL('bigint')" in statement
    assert "<>" in statement
    assert "ALL" in statement
    assert " ? " not in statement
    assert compiled.params == ([20],)


def test_compile_metadata_filter_routes_declared_date_index_with_iso_string():
    compiled = compile_metadata_filter(
        {"event_date": "2026-07-01"},
        metadata_column="metadata",
        metadata_indexes={"event_date": "date"},
    )

    statement = _sql_repr(compiled)
    assert "Literal('event_date')" in statement
    assert "to_date" not in statement
    assert "->>" in statement
    assert compiled.params == ("2026-07-01",)


@pytest.mark.parametrize("operator", ["$in", "$nin"])
def test_compile_metadata_filter_routes_declared_date_index_membership(operator):
    compiled = compile_metadata_filter(
        {"event_date": {operator: ["2026-07-01", dt.date(2026, 7, 2)]}},
        metadata_column="metadata",
        metadata_indexes={"event_date": "date"},
    )

    statement = _sql_repr(compiled)
    assert "Literal('event_date')" in statement
    assert "->>" in statement
    assert ("ANY" if operator == "$in" else "ALL") in statement
    assert "::text[]" not in statement
    assert "unnest" not in statement
    assert "to_date" not in statement
    assert "to_char" not in statement
    expected_values = ["2026-07-01", "2026-07-02"]
    assert compiled.params == (expected_values,)


def test_compile_metadata_filter_supports_dynamic_timestamp_membership():
    compiled = compile_metadata_filter(
        {
            "created_at": {
                "$in": [
                    "2026-07-01T12:30:45",
                    dt.datetime(2026, 7, 2, 8, 15, 0),
                ]
            }
        },
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "Literal('created_at')" in statement
    assert "->>" in statement
    assert "ANY" in statement
    assert "::timestamp[]" in statement
    assert compiled.params == (
        [
            dt.datetime(2026, 7, 1, 12, 30, 45),
            dt.datetime(2026, 7, 2, 8, 15, 0),
        ],
    )


def test_compile_metadata_filter_supports_dynamic_time_membership():
    compiled = compile_metadata_filter(
        {"start_time": {"$nin": ["09:30:00", dt.time(10, 45, 0)]}},
        metadata_column="metadata",
    )

    statement = _sql_repr(compiled)
    assert "Literal('start_time')" in statement
    assert "->>" in statement
    assert "NOT" in statement
    assert "ANY" in statement
    assert "::time[]" in statement
    assert compiled.params == ([dt.time(9, 30, 0), dt.time(10, 45, 0)],)


def test_compile_metadata_filter_rejects_like_on_non_text_metadata_index():
    with pytest.raises(GaussDBFilterError, match="text"):
        compile_metadata_filter(
            {"score": {"$like": "%1%"}},
            metadata_column="metadata",
            metadata_indexes={"score": "float"},
        )
