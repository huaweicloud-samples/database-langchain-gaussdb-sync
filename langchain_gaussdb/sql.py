from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from psycopg2 import sql

from langchain_gaussdb.errors import GaussDBSQLBuildError


@dataclass(frozen=True)
class CompiledSQL:
    statement: sql.Composable
    params: tuple[Any, ...]

    def __init__(self, statement: sql.Composable, params: Iterable[Any] = ()) -> None:
        object.__setattr__(self, "statement", statement)
        object.__setattr__(self, "params", tuple(params))


def identifier(name: str, label: str = "identifier") -> sql.Identifier:
    if not isinstance(name, str) or not name.strip():
        raise GaussDBSQLBuildError(f"{label} must be a non-empty string")
    if "." in name:
        raise GaussDBSQLBuildError(f"{label} must be a single identifier")
    if "\x00" in name:
        raise GaussDBSQLBuildError(f"{label} must not contain NUL characters")
    if len(name.encode("utf-8")) > 63:
        raise GaussDBSQLBuildError(f"{label} must not exceed 63 UTF-8 bytes")
    return sql.Identifier(name)


def qualified_name(schema: str | None, name: str) -> sql.Composable:
    if not schema:
        return identifier(name)
    return sql.SQL(".").join(
        [identifier(schema, label="schema"), identifier(name, label="identifier")]
    )


def build_odku_insert(
    *,
    schema: str | None,
    table: str,
    insert_columns: Iterable[str],
    update_columns: Iterable[str],
    rows: Iterable[Iterable[Any]],
    value_expressions: Mapping[str, sql.Composable] | None = None,
) -> CompiledSQL:
    insert_column_names = tuple(insert_columns)
    update_column_names = tuple(update_columns)
    value_expressions = value_expressions or {}
    row_values = [tuple(row) for row in rows]

    table_name = qualified_name(schema, table)
    insert_identifiers = [
        identifier(column, label="insert_columns") for column in insert_column_names
    ]
    update_assignments = [
        sql.SQL("{} = VALUES({})").format(
            identifier(column, label="update_columns"),
            identifier(column, label="update_columns"),
        )
        for column in update_column_names
    ]
    value_template = sql.SQL("({})").format(
        sql.SQL(", ").join(
            value_expressions.get(column, sql.Placeholder())
            for column in insert_column_names
        )
    )
    values_clause = sql.SQL(", ").join(value_template for _ in row_values)

    statement = sql.SQL(
        "INSERT INTO {} ({}) VALUES {} ON DUPLICATE KEY UPDATE {}"
    ).format(
        table_name,
        sql.SQL(", ").join(insert_identifiers),
        values_clause,
        sql.SQL(", ").join(update_assignments),
    )
    params = tuple(value for row in row_values for value in row)
    return CompiledSQL(statement, params)
