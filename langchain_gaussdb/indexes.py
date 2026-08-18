from __future__ import annotations

import hashlib
import re
from typing import Any

from psycopg2 import sql

from langchain_gaussdb.errors import GaussDBSQLBuildError
from langchain_gaussdb.metadata_sql import (
    METADATA_INDEX_CASTS,
    metadata_literal_index_selector,
)
from langchain_gaussdb.sql import CompiledSQL, identifier, qualified_name

# GsDiskANN supports up to 4096 dimensions on centralized GaussDB and up to
# 1024 on distributed GaussDB. SQL builders enforce the global maximum because
# they do not own a connection; GaussDBVectorStore applies the deployment-
# specific limit before initialization DDL.
MAX_EMBEDDING_DIMENSION = 4096
MAX_DISTRIBUTED_EMBEDDING_DIMENSION = 1024

_VECTOR_METRICS = {"cosine": "COSINE", "l2": "L2"}
_DISKANN_OPTIONS = {"pq_nseg", "pq_nclus", "queue_size", "enable_pq"}
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_create_vector_index(
    schema: str | None,
    table: str,
    embedding_column: str,
    embedding_dimension: int,
    *,
    index_name: str | None = None,
    distance_strategy: str = "cosine",
    **params: Any,
) -> tuple[str, CompiledSQL]:
    _validate_identifier(table, "table")
    _validate_identifier(embedding_column, "embedding_column")
    if schema is not None:
        _validate_identifier(schema, "schema")
    if (
        not isinstance(embedding_dimension, int)
        or isinstance(embedding_dimension, bool)
        or embedding_dimension <= 0
    ):
        raise GaussDBSQLBuildError("embedding_dimension must be a positive integer")
    if embedding_dimension > MAX_EMBEDDING_DIMENSION:
        raise GaussDBSQLBuildError(
            f"embedding_dimension must be less than or equal to "
            f"{MAX_EMBEDDING_DIMENSION}"
        )
    if (
        not isinstance(distance_strategy, str)
        or distance_strategy not in _VECTOR_METRICS
    ):
        raise GaussDBSQLBuildError("distance_strategy must be one of: cosine, l2")

    metric = _VECTOR_METRICS[distance_strategy]
    if index_name is None:
        readable_name = f"{table}_{embedding_column}_gsdiskann_{distance_strategy}_idx"
        index_name = _stable_automatic_index_name(
            readable_name,
            f"{table}_{embedding_column}_gsdiskann_{distance_strategy}",
        )
    _validate_identifier(index_name, "index_name")

    if (
        "pq_nseg" in params
        and isinstance(params["pq_nseg"], int)
        and not isinstance(params["pq_nseg"], bool)
        and params["pq_nseg"] > embedding_dimension
    ):
        raise GaussDBSQLBuildError(
            "pq_nseg must be less than or equal to embedding_dimension"
        )
    options = _build_relation_options(params, _DISKANN_OPTIONS)
    statement = sql.SQL(
        "CREATE INDEX IF NOT EXISTS {} ON {} USING gsdiskann ({} {}){}"
    ).format(
        identifier(index_name, label="index_name"),
        qualified_name(schema, table),
        identifier(embedding_column, label="embedding_column"),
        sql.SQL(metric),
        options,
    )
    return index_name, CompiledSQL(statement)


def build_create_metadata_index(
    schema: str | None,
    table: str,
    metadata_column: str,
    key: str,
    index_name: str | None = None,
    cast: str | None = None,
) -> tuple[str, CompiledSQL]:
    _validate_identifier(table, "table")
    _validate_identifier(metadata_column, "metadata_column")
    _validate_metadata_key(key)
    if schema is not None:
        _validate_identifier(schema, "schema")

    cast_suffix = ""
    if cast is not None:
        if cast not in METADATA_INDEX_CASTS:
            raise GaussDBSQLBuildError(
                f"cast must be one of {sorted(METADATA_INDEX_CASTS)}"
            )
        cast_suffix = f"_{cast}"
    if index_name is None:
        naming_suffix = "_json_text" if cast is None else cast_suffix
        index_name = _automatic_metadata_index_name(
            table,
            metadata_column,
            key,
            naming_suffix,
        )
    _validate_identifier(index_name, "index_name")

    metadata_identifier = identifier(metadata_column, label="metadata_column")
    metadata_expr = metadata_literal_index_selector(metadata_identifier, key, cast)

    statement = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (({}))").format(
        identifier(index_name, label="index_name"),
        qualified_name(schema, table),
        metadata_expr,
    )
    return index_name, CompiledSQL(statement)


def build_create_bm25_index(
    schema: str | None,
    table: str,
    field: str,
    index_name: str | None = None,
) -> tuple[str, CompiledSQL]:
    _validate_identifier(table, "table")
    _validate_identifier(field, "field")
    if schema is not None:
        _validate_identifier(schema, "schema")
    if index_name is None:
        index_name = _stable_automatic_index_name(
            f"{table}_{field}_bm25_idx",
            f"{table}_{field}_bm25",
        )
    _validate_identifier(index_name, "index_name")

    statement = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING bm25({})").format(
        identifier(index_name, label="index_name"),
        qualified_name(schema, table),
        identifier(field, label="field"),
    )
    return index_name, CompiledSQL(statement)


def _build_relation_options(
    params: dict[str, Any],
    allowed: set[str],
    ddl_names: dict[str, str] | None = None,
) -> sql.Composable:
    ddl_names = ddl_names or {}
    options: list[sql.Composable] = []
    for key, value in params.items():
        if key not in allowed:
            allowed_names = ", ".join(sorted(allowed))
            raise GaussDBSQLBuildError(
                f"relation option {key} must be one of: {allowed_names}"
            )
        ddl_key = ddl_names.get(key, key)
        if key == "enable_pq":
            if not isinstance(value, bool):
                raise GaussDBSQLBuildError("enable_pq must be a boolean")
            rendered_value = "true" if value else "false"
        elif isinstance(value, int) and not isinstance(value, bool):
            if value <= 0:
                raise GaussDBSQLBuildError(f"{key} must be a positive integer")
            rendered_value = str(value)
        else:
            raise GaussDBSQLBuildError(f"{key} must be a positive integer")
        options.append(
            sql.SQL("{}={}").format(sql.SQL(ddl_key), sql.SQL(rendered_value))
        )

    if not options:
        return sql.SQL("")
    return sql.SQL(" WITH ({})").format(sql.SQL(", ").join(options))


def _validate_identifier(value: str, label: str) -> None:
    identifier(value, label=label)


def _validate_metadata_key(key: str) -> None:
    if not isinstance(key, str) or not key:
        raise GaussDBSQLBuildError("metadata key must be a non-empty string")
    if key.startswith("$"):
        raise GaussDBSQLBuildError("metadata key must not start with '$'")
    if "\0" in key:
        raise GaussDBSQLBuildError("metadata key must not contain NUL")


def _automatic_metadata_index_name(
    table: str,
    metadata_column: str,
    key: str,
    cast_suffix: str,
) -> str:
    readable_name = f"{table}_{metadata_column}_{key}{cast_suffix}_idx"
    return _stable_automatic_index_name(
        readable_name,
        f"{table}_{metadata_column}{cast_suffix}",
        allow_readable=_SAFE_TOKEN_RE.fullmatch(key) is not None,
    )


def _stable_automatic_index_name(
    readable_name: str,
    readable_prefix: str,
    *,
    allow_readable: bool = True,
) -> str:
    if (
        allow_readable
        and _SAFE_TOKEN_RE.fullmatch(readable_name)
        and len(readable_name.encode("utf-8")) <= 63
    ):
        return readable_name
    digest = hashlib.sha256(readable_name.encode("utf-8")).hexdigest()[:12]
    suffix = f"_{digest}_idx"
    prefix = _truncate_utf8(readable_prefix, 63 - len(suffix.encode("utf-8")))
    return f"{prefix}{suffix}"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
