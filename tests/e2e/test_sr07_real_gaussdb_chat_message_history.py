import asyncio
import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from psycopg2 import sql

from langchain_gaussdb.chat_message_history import GaussDBChatMessageHistory
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL, qualified_name

pytestmark = pytest.mark.gaussdb_e2e


def _dsn():
    value = os.getenv("GAUSSDB_TEST_DSN")
    if not value:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return value


def _table_name():
    return "sr07_" + uuid.uuid4().hex[:12]


def _engine():
    return GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)


def _enable_session_writes(engine):
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable sr07 e2e writes",
    )


def _drop_table(engine, table, *, strict):
    if not table.startswith("sr07_"):
        raise AssertionError("SR-07 e2e cleanup only drops sr07_ generated tables")
    try:
        _enable_session_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
            ),
            operation="drop sr07 e2e table",
        )
    except Exception:
        if strict:
            raise


def _close_engine(engine, *, strict):
    try:
        engine.close()
    except Exception:
        if strict:
            raise


def _history(engine, table, session_id="session-1"):
    return GaussDBChatMessageHistory(
        session_id=session_id,
        table_name=table,
        engine=engine,
    )


def _rich_messages():
    return [
        HumanMessage(
            content="hello",
            id="human-id",
            name="human-name",
            additional_kwargs={"client": {"trace": "h-1"}},
        ),
        AIMessage(
            content="answer",
            id="ai-id",
            name="assistant-name",
            response_metadata={"finish_reason": "tool_calls"},
            additional_kwargs={"vendor": {"trace": "abc"}},
            tool_calls=[
                {
                    "name": "lookup",
                    "args": {"query": "GaussDB"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        SystemMessage(
            content="system policy",
            id="system-id",
            name="system-name",
            response_metadata={"source": "test"},
        ),
        ToolMessage(
            content="tool result",
            id="tool-id",
            name="tool-name",
            tool_call_id="call-1",
        ),
    ]


def test_real_gaussdb_create_table_and_index():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        history = _history(engine, table)

        history.create_table_if_not_exists()

        columns = engine.fetch_all(
            CompiledSQL(
                sql.SQL(
                    """
                    SELECT column_name, data_type, udt_name
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = current_schema()
                    ORDER BY ordinal_position
                    """
                ),
                [table],
            ),
            operation="inspect sr07 chat message columns",
        )
        column_map = {row[0]: row for row in columns}

        assert list(column_map) == ["id", "session_id", "message", "created_at"]
        assert column_map["id"][0] == "id"
        assert column_map["session_id"][1] == "text"
        assert column_map["message"][1:] == ("jsonb", "jsonb")
        assert column_map["created_at"][0] == "created_at"

        indexes = engine.fetch_all(
            CompiledSQL(
                sql.SQL(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = current_schema() AND tablename = %s
                    """
                ),
                [table],
            ),
            operation="inspect sr07 chat message indexes",
        )
        expected_index = f"{table}_session_id_id_idx"
        matching = [row[1].lower() for row in indexes if row[0] == expected_index]

        assert matching
        assert any("(session_id, id)" in indexdef for indexdef in matching)
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_message_roundtrip_preserves_types_fields_and_order():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        history = _history(engine, table)
        history.create_table_if_not_exists()

        history.add_messages(_rich_messages())
        restored = history.messages

        assert [type(message) for message in restored] == [
            HumanMessage,
            AIMessage,
            SystemMessage,
            ToolMessage,
        ]
        assert [message.content for message in restored] == [
            "hello",
            "answer",
            "system policy",
            "tool result",
        ]

        restored_human, restored_ai, restored_system, restored_tool = restored
        assert restored_human.id == "human-id"
        assert restored_human.name == "human-name"
        assert restored_human.additional_kwargs == {"client": {"trace": "h-1"}}
        assert restored_ai.id == "ai-id"
        assert restored_ai.name == "assistant-name"
        assert restored_ai.response_metadata == {"finish_reason": "tool_calls"}
        assert restored_ai.additional_kwargs == {"vendor": {"trace": "abc"}}
        assert restored_ai.tool_calls == [
            {
                "name": "lookup",
                "args": {"query": "GaussDB"},
                "id": "call-1",
                "type": "tool_call",
            }
        ]
        assert restored_system.id == "system-id"
        assert restored_system.name == "system-name"
        assert restored_system.response_metadata == {"source": "test"}
        assert restored_tool.id == "tool-id"
        assert restored_tool.name == "tool-name"
        assert restored_tool.tool_call_id == "call-1"
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_session_isolation_and_clear():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        first = _history(engine, table, session_id="session-1")
        second = _history(engine, table, session_id="session-2")
        first.create_table_if_not_exists()

        first.add_messages([HumanMessage(content="first")])
        second.add_messages([HumanMessage(content="second")])

        assert [message.content for message in first.messages] == ["first"]
        assert [message.content for message in second.messages] == ["second"]

        first.clear()

        assert first.messages == []
        assert [message.content for message in second.messages] == ["second"]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_duplicate_messages_are_append_only():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        history = _history(engine, table)
        history.create_table_if_not_exists()
        message = HumanMessage(content="repeat", id="duplicate-id")

        history.add_messages([message])
        history.add_messages([message])

        restored = history.messages
        assert [message.content for message in restored] == ["repeat", "repeat"]
        assert [message.id for message in restored] == [
            "duplicate-id",
            "duplicate-id",
        ]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_async_methods_roundtrip():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        history = _history(engine, table)
        history.create_table_if_not_exists()

        asyncio.run(history.aadd_messages([HumanMessage(content="async hello")]))
        restored = asyncio.run(history.aget_messages())

        assert [message.content for message in restored] == ["async hello"]

        asyncio.run(history.aclear())

        assert asyncio.run(history.aget_messages()) == []
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)
