from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from psycopg2 import sql

from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.errors import (
    GaussDBCapabilityError,
    GaussDBConnectionError,
    GaussDBSQLError,
)
from langchain_gaussdb.sql import CompiledSQL, identifier, qualified_name

_DUPLICATE_OBJECT_SQLSTATES = frozenset({"42P07", "42710", "23505"})
_WRITE_BATCH_SIZE = 1000


class GaussDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        *,
        session_id: str,
        table_name: str = "langchain_chat_message",
        schema_name: str | None = None,
        engine: GaussDBEngine | None = None,
        dsn: str | None = None,
        connection_kwargs: dict[str, Any] | None = None,
        create_table: bool = False,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        identifier(table_name, label="table_name")
        if schema_name is not None:
            identifier(schema_name, label="schema_name")

        connection_sources = [
            dsn is not None,
            connection_kwargs is not None,
        ]
        if engine is not None and any(connection_sources):
            raise GaussDBConnectionError(
                "engine is mutually exclusive with other connection sources"
            )
        if engine is None and sum(connection_sources) != 1:
            raise GaussDBConnectionError(
                "GaussDBChatMessageHistory requires exactly one connection source"
            )

        self._session_id = session_id
        self._table_name = table_name
        self._schema_name = schema_name
        self._owns_engine = engine is None
        self._engine = engine or GaussDBEngine(
            dsn=dsn,
            connection_kwargs=connection_kwargs,
        )

        if create_table:
            try:
                self.create_table_if_not_exists()
            except BaseException:
                if self._owns_engine:
                    try:
                        self._engine.close()
                    except BaseException:
                        pass
                raise

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def schema_name(self) -> str | None:
        return self._schema_name

    def close(self) -> None:
        if self._owns_engine:
            self._engine.close()

    def create_table_if_not_exists(self) -> None:
        try:
            self._engine.execute(
                self._build_create_table_sql(),
                operation="create chat message table",
            )
        except GaussDBSQLError as exc:
            if exc.sqlstate not in _DUPLICATE_OBJECT_SQLSTATES:
                raise
        self._validate_generated_table()
        try:
            self._engine.execute(
                self._build_create_index_sql(),
                operation="create chat message index",
            )
        except GaussDBSQLError as exc:
            if exc.sqlstate not in _DUPLICATE_OBJECT_SQLSTATES:
                raise
        self._validate_generated_index()

    @property
    def messages(self) -> list[BaseMessage]:
        rows = self._engine.fetch_all(
            self._build_get_messages_sql(),
            operation="get chat messages",
        )
        return self._messages_from_rows(rows)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        message_list = self._validate_messages(messages)
        if not message_list:
            return
        serialized_messages = self._serialize_messages(message_list)
        for start in range(0, len(serialized_messages), _WRITE_BATCH_SIZE):
            compiled = self._build_serialized_messages_sql(
                serialized_messages[start : start + _WRITE_BATCH_SIZE]
            )
            self._engine.execute(
                compiled,
                operation="add chat messages",
            )

    def clear(self) -> None:
        self._engine.execute(
            self._build_clear_sql(),
            operation="clear chat messages",
        )

    def _build_get_messages_sql(self) -> CompiledSQL:
        statement = sql.SQL(
            "SELECT message FROM {} WHERE session_id=%s ORDER BY id ASC"
        ).format(qualified_name(self._schema_name, self._table_name))
        return CompiledSQL(statement, [self._session_id])

    def _messages_from_rows(
        self,
        rows: Sequence[Sequence[Any]],
    ) -> list[BaseMessage]:
        row_dicts = [
            self._message_dict_from_row(row, index) for index, row in enumerate(rows)
        ]
        messages: list[BaseMessage] = []
        for index, row_dict in enumerate(row_dicts):
            message_type = row_dict.get("type", "unknown")
            try:
                messages.extend(messages_from_dict([row_dict]))
            except Exception:
                raise ValueError(
                    f"Invalid chat message row {index} of type {message_type!r}"
                ) from None
        return messages

    def _build_add_messages_sql(
        self,
        messages: Sequence[BaseMessage],
    ) -> CompiledSQL | None:
        message_list = self._validate_messages(messages)
        if not message_list:
            return None
        return self._build_serialized_messages_sql(
            self._serialize_messages(message_list)
        )

    def _serialize_messages(
        self,
        messages: Sequence[BaseMessage],
    ) -> list[str]:
        serialized_messages: list[str] = []
        for index, message in enumerate(messages):
            serialized = message_to_dict(message)
            message_type = serialized.get("type", type(message).__name__)
            try:
                serialized_messages.append(json.dumps(serialized, allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Chat message {index} of type {message_type!r} is not JSON serializable"
                ) from exc
        return serialized_messages

    def _build_serialized_messages_sql(
        self,
        serialized_messages: Sequence[str],
    ) -> CompiledSQL:
        params: list[Any] = []
        for message_json in serialized_messages:
            params.extend([self._session_id, message_json])
        values_clause = sql.SQL(", ").join(
            sql.SQL("(%s, %s::jsonb)") for _ in serialized_messages
        )
        statement = sql.SQL("INSERT INTO {} (session_id, message) VALUES {}").format(
            qualified_name(self._schema_name, self._table_name),
            values_clause,
        )
        return CompiledSQL(statement, params)

    def _build_clear_sql(self) -> CompiledSQL:
        statement = sql.SQL("DELETE FROM {} WHERE session_id = %s").format(
            qualified_name(self._schema_name, self._table_name)
        )
        return CompiledSQL(statement, [self._session_id])

    def _validate_messages(self, messages: Any) -> list[BaseMessage]:
        if messages is None or isinstance(messages, (str, bytes)):
            raise ValueError("messages must be a sequence of BaseMessage instances")
        try:
            message_list = list(messages)
        except TypeError as exc:
            raise ValueError(
                "messages must be a sequence of BaseMessage instances"
            ) from exc

        for message in message_list:
            if not isinstance(message, BaseMessage):
                raise ValueError("messages must contain only BaseMessage instances")
        return message_list

    def _message_dict_from_row(self, row: Any, index: int) -> dict[str, Any]:
        message_value = row[0] if isinstance(row, (tuple, list)) else row
        if isinstance(message_value, dict):
            return self._validate_message_dict_row(message_value, index)
        if isinstance(message_value, str):
            try:
                decoded = json.loads(message_value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid chat message JSON row {index}: {exc.msg}"
                ) from exc
            if isinstance(decoded, dict):
                return self._validate_message_dict_row(decoded, index)
            raise ValueError(f"Invalid chat message JSON row {index}: expected object")
        raise ValueError(
            f"Invalid chat message row {index}: expected dict or JSON string"
        )

    def _validate_message_dict_row(
        self, message_dict: dict[str, Any], index: int
    ) -> dict[str, Any]:
        missing = [key for key in ("type", "data") if key not in message_dict]
        if missing:
            raise ValueError(
                f"Invalid chat message row {index}: missing {', '.join(missing)}"
            )
        return message_dict

    def _build_create_table_sql(self) -> CompiledSQL:
        statement = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "id BIGSERIAL PRIMARY KEY, "
            "session_id TEXT NOT NULL, "
            "message JSONB NOT NULL, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ") WITH (storage_type=ustore)"
        ).format(qualified_name(self._schema_name, self._table_name))
        return CompiledSQL(statement)

    def _build_create_index_sql(self) -> CompiledSQL:
        index_name = _generated_index_name(
            self._table_name,
            suffix="session_id_id_idx",
        )
        statement = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON {} (session_id, id)"
        ).format(
            identifier(index_name, label="index_name"),
            qualified_name(self._schema_name, self._table_name),
        )
        return CompiledSQL(statement)

    def _validate_generated_table(self) -> None:
        rows = self._engine.fetch_all(
            self._build_validate_table_sql(),
            operation="validate chat message table",
        )
        if self._table_definition_matches(rows):
            return
        expected_schema = self._schema_name or "visible search_path schema"
        raise GaussDBCapabilityError(
            f"GaussDB table {self._table_name} already exists with a different "
            f"definition; expected schema={expected_schema}, exactly four ordered "
            "columns: id bigint not-null primary key with nextval default, "
            "session_id text not-null, message jsonb not-null, created_at "
            f"timestamp without time zone with a default; actual={rows!r}"
        )

    def _build_validate_table_sql(self) -> CompiledSQL:
        statement = sql.SQL(
            """
            WITH expected_target AS (
                SELECT table_rel.oid AS table_oid, table_ns.nspname
                FROM pg_class AS table_rel
                JOIN pg_namespace AS table_ns
                  ON table_ns.oid = table_rel.relnamespace
                WHERE table_rel.relname = %s
                  AND table_rel.relkind IN ('r', 'p')
                  AND ((%s IS NULL AND pg_table_is_visible(table_rel.oid))
                       OR table_ns.nspname = %s)
            ),
            primary_key AS (
                SELECT target.table_oid, idx.indkey[0] AS key_attnum,
                       idx.indnkeyatts, idx.indnatts,
                       idx.indisvalid, idx.indisready, idx.indisusable,
                       idx.indpred, idx.indexprs
                FROM expected_target AS target
                JOIN pg_index AS idx ON idx.indrelid = target.table_oid
                WHERE idx.indisprimary IS TRUE
            )
            SELECT target.nspname, table_rel.relname,
                   attr.attnum, attr.attname,
                   format_type(attr.atttypid, attr.atttypmod),
                   attr.attnotnull,
                   default_def.adbin IS NOT NULL,
                   pg_get_expr(default_def.adbin, default_def.adrelid),
                   primary_key.key_attnum,
                   primary_key.indnkeyatts, primary_key.indnatts,
                   primary_key.indisvalid, primary_key.indisready,
                   primary_key.indisusable,
                   primary_key.indpred, primary_key.indexprs
            FROM expected_target AS target
            JOIN pg_class AS table_rel ON table_rel.oid = target.table_oid
            JOIN pg_attribute AS attr ON attr.attrelid = target.table_oid
            LEFT JOIN pg_attrdef AS default_def
              ON default_def.adrelid = attr.attrelid
             AND default_def.adnum = attr.attnum
            LEFT JOIN primary_key ON primary_key.table_oid = target.table_oid
            WHERE attr.attnum > 0 AND attr.attisdropped IS FALSE
            ORDER BY target.nspname, table_rel.relname, attr.attnum
            """
        )
        return CompiledSQL(
            statement,
            (self._table_name, self._schema_name, self._schema_name),
        )

    def _table_definition_matches(
        self,
        rows: Sequence[Sequence[Any]],
    ) -> bool:
        if len(rows) != 4:
            return False
        checked_rows: list[Sequence[Any]] = []
        for row in rows:
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != 16
            ):
                return False
            checked_rows.append(row)

        expected_columns = (
            (1, "id", "bigint", True),
            (2, "session_id", "text", True),
            (3, "message", "jsonb", True),
            (4, "created_at", "timestamp without time zone", False),
        )
        for row, expected_column in zip(checked_rows, expected_columns):
            schema_matches = (
                row[0] == self._schema_name
                if self._schema_name is not None
                else isinstance(row[0], str) and bool(row[0])
            )
            if not schema_matches or row[1] != self._table_name:
                return False
            if (
                type(row[2]) is not int
                or row[2] != expected_column[0]
                or row[3] != expected_column[1]
                or not isinstance(row[4], str)
                or row[4].lower() != expected_column[2]
                or row[5] is not expected_column[3]
            ):
                return False
            if (
                type(row[8]) is not int
                or row[8] != 1
                or type(row[9]) is not int
                or row[9] != 1
                or type(row[10]) is not int
                or row[10] != 1
                or row[11] is not True
                or row[12] is not True
                or row[13] is not True
                or row[14] is not None
                or row[15] is not None
            ):
                return False

        id_default = checked_rows[0][7]
        created_at_default = checked_rows[3][7]
        return (
            checked_rows[0][6] is True
            and isinstance(id_default, str)
            and id_default.strip().lower().startswith("nextval(")
            and checked_rows[1][6] is False
            and checked_rows[1][7] is None
            and checked_rows[2][6] is False
            and checked_rows[2][7] is None
            and checked_rows[3][6] is True
            and isinstance(created_at_default, str)
            and bool(created_at_default.strip())
        )

    def _validate_generated_index(self) -> None:
        index_name = _generated_index_name(
            self._table_name,
            suffix="session_id_id_idx",
        )
        rows = self._engine.fetch_all(
            self._build_validate_index_sql(index_name),
            operation="validate chat message index",
        )
        if len(rows) == 1 and self._index_definition_matches(rows[0]):
            return
        actual: Any = rows[0] if len(rows) == 1 else rows
        expected_schema = self._schema_name or "visible search_path schema"
        raise GaussDBCapabilityError(
            f"GaussDB index {index_name} already exists with a different "
            f"definition; expected schema={expected_schema}, table="
            f"{self._table_name}, btree-compatible method, columns="
            "(session_id, id), exactly two keys, valid, ready, usable, "
            f"non-partial and non-expression; actual={actual!r}"
        )

    def _build_validate_index_sql(self, index_name: str) -> CompiledSQL:
        statement = sql.SQL(
            """
            WITH expected_target AS (
                SELECT table_ns.oid AS schema_oid, table_ns.nspname,
                       %s::text AS index_name
                FROM pg_class AS table_rel
                JOIN pg_namespace AS table_ns
                  ON table_ns.oid = table_rel.relnamespace
                WHERE table_rel.relname = %s
                  AND table_rel.relkind IN ('r', 'p')
                  AND ((%s IS NULL AND pg_table_is_visible(table_rel.oid))
                       OR table_ns.nspname = %s)
            )
            SELECT target.nspname, actual_table.relname, am.amname,
                   idx.indnkeyatts, idx.indnatts,
                   first_key.attname, second_key.attname,
                   idx.indisvalid, idx.indisready, idx.indisusable,
                   idx.indpred, idx.indexprs
            FROM expected_target AS target
            JOIN pg_class AS index_rel
              ON index_rel.relnamespace = target.schema_oid
             AND index_rel.relname = target.index_name
            JOIN pg_index AS idx ON idx.indexrelid = index_rel.oid
            JOIN pg_class AS actual_table ON actual_table.oid = idx.indrelid
            JOIN pg_am AS am ON am.oid = index_rel.relam
            JOIN pg_attribute AS first_key
              ON first_key.attrelid = actual_table.oid
             AND first_key.attnum = idx.indkey[0]
             AND first_key.attisdropped IS FALSE
            JOIN pg_attribute AS second_key
              ON second_key.attrelid = actual_table.oid
             AND second_key.attnum = idx.indkey[1]
             AND second_key.attisdropped IS FALSE
            """
        )
        return CompiledSQL(
            statement,
            (
                index_name,
                self._table_name,
                self._schema_name,
                self._schema_name,
            ),
        )

    def _index_definition_matches(self, row: object) -> bool:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return False
        if len(row) != 12:
            return False
        schema_matches = (
            row[0] == self._schema_name
            if self._schema_name is not None
            else isinstance(row[0], str) and bool(row[0])
        )
        return (
            schema_matches
            and row[1] == self._table_name
            and str(row[2]).lower() in {"btree", "ubtree"}
            and type(row[3]) is int
            and row[3] == 2
            and type(row[4]) is int
            and row[4] == 2
            and tuple(row[5:7]) == ("session_id", "id")
            and row[7] is True
            and row[8] is True
            and row[9] is True
            and row[10] is None
            and row[11] is None
        )


def _generated_index_name(table_name: str, *, suffix: str) -> str:
    raw_name = f"{table_name}_{suffix}"
    if len(raw_name.encode("utf-8")) <= 63:
        return raw_name

    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:8]
    suffix_part = f"_{digest}_{suffix}"
    prefix_budget = 63 - len(suffix_part.encode("utf-8"))
    prefix = _truncate_utf8(table_name, prefix_budget)
    return f"{prefix}{suffix_part}"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore")
