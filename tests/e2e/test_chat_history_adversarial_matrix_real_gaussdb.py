from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from psycopg2 import sql

from langchain_gaussdb import (
    GaussDBCapabilityError,
    GaussDBChatMessageHistory,
    GaussDBConnectionError,
    GaussDBEngine,
    GaussDBVectorStore,
)
from langchain_gaussdb.sql import CompiledSQL

MESSAGE_CASES = (
    {"kind": "human", "position": 0},
    {"kind": "ai", "position": 1},
    {"kind": "system", "position": 2},
    {"kind": "tool", "position": 3},
    {"kind": "duplicate", "position": 4},
)

MALFORMED_ROW_CASES = (
    {
        "state": "missing-data",
        "payload": {"type": "human", "private": "missing-data-secret"},
        "error": "missing data",
    },
    {
        "state": "unknown-type",
        "payload": {
            "type": "private_unknown",
            "data": {"content": "unknown-type-secret"},
        },
        "error": "private_unknown",
    },
    {
        "state": "non-object",
        "payload": ["non-object-secret", {"private": "payload"}],
        "error": "expected dict or JSON string",
    },
)

SESSION_CLEAR_CASES = (
    {"path": "sync-sync", "add_async": False, "clear_async": False},
    {"path": "sync-async", "add_async": False, "clear_async": True},
    {"path": "async-sync", "add_async": True, "clear_async": False},
    {"path": "async-async", "add_async": True, "clear_async": True},
)

CLOSE_OWNERSHIP_CASES = (
    {"ownership": "owned"},
    {"ownership": "external"},
)

CHAT_INDEX_MISMATCH_CASES = (
    {"state": "wrong-order"},
    {"state": "wrong-column"},
    {"state": "wrong-table"},
)


def _safe_role(value: str) -> str:
    return value.replace("-", "_")


def _history(
    engine: GaussDBEngine,
    table: Any,
    session_id: str,
) -> GaussDBChatMessageHistory:
    return GaussDBChatMessageHistory(
        engine=engine,
        schema_name=table.schema,
        table_name=table.name,
        session_id=session_id,
    )


def _rich_messages() -> list[BaseMessage]:
    human = HumanMessage(
        content="human-content",
        id="human-id",
        name="human-name",
        additional_kwargs={"client": {"trace": "human-trace"}},
        response_metadata={"locale": "zh-CN"},
    )
    ai = AIMessage(
        content="ai-content",
        id="ai-id",
        name="assistant-name",
        additional_kwargs={"vendor": {"trace": "ai-trace"}},
        response_metadata={"finish_reason": "tool_calls"},
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "GaussDB"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    system = SystemMessage(
        content="system-content",
        id="system-id",
        name="system-name",
        additional_kwargs={"policy": {"revision": 2}},
        response_metadata={"source": "e2e"},
    )
    tool = ToolMessage(
        content="tool-content",
        id="tool-id",
        name="tool-name",
        tool_call_id="call-1",
        artifact={"rows": [1, 2]},
        status="success",
        additional_kwargs={"server": "gaussdb"},
        response_metadata={"latency_ms": 4},
    )
    duplicate = HumanMessage(
        content="human-content",
        id="human-id",
        name="human-name",
        additional_kwargs={"client": {"trace": "human-trace"}},
        response_metadata={"locale": "zh-CN"},
    )
    return [human, ai, system, tool, duplicate]


def _assert_exact_messages(
    restored: list[BaseMessage],
    expected: list[BaseMessage],
) -> None:
    assert [type(message) for message in restored] == [
        HumanMessage,
        AIMessage,
        SystemMessage,
        ToolMessage,
        HumanMessage,
    ]
    assert [message_to_dict(message) for message in restored] == [
        message_to_dict(message) for message in expected
    ]
    assert message_to_dict(restored[0]) == message_to_dict(restored[-1])


def _payload_content(value: Any) -> str:
    payload = json.loads(value) if isinstance(value, str) else value
    assert isinstance(payload, dict)
    data = payload.get("data")
    assert isinstance(data, dict)
    content = data.get("content")
    assert isinstance(content, str)
    return content


@dataclass
class _HeldTableLock:
    connection_context: Any
    connection: Any
    cursor: Any


def _hold_table_lock(
    engine: GaussDBEngine,
    table: Any,
) -> tuple[_HeldTableLock, int]:
    connection_context = engine.connection()
    connection = connection_context.__enter__()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            sql.SQL("LOCK TABLE {}.{} IN ACCESS EXCLUSIVE MODE").format(
                sql.Identifier(table.schema),
                sql.Identifier(table.name),
            )
        )
        cursor.execute("SELECT pg_backend_pid()")
        pid = int(cursor.fetchone()[0])
        return _HeldTableLock(connection_context, connection, cursor), pid
    except BaseException:
        try:
            connection.rollback()
        finally:
            if cursor is not None:
                cursor.close()
            connection_context.__exit__(None, None, None)
        raise


def _release_table_lock(
    held_lock: _HeldTableLock,
    *,
    rollback: bool = False,
) -> None:
    primary_error: BaseException | None = None
    try:
        try:
            if rollback:
                held_lock.connection.rollback()
            else:
                held_lock.connection.commit()
        except BaseException as exc:
            primary_error = exc
            try:
                held_lock.connection.rollback()
            except BaseException:
                pass
            raise
    finally:
        try:
            held_lock.cursor.close()
        finally:
            try:
                held_lock.connection_context.__exit__(None, None, None)
            except BaseException:
                if primary_error is None:
                    raise


async def _wait_for_lock_waiters(
    control_engine: GaussDBEngine,
    table: Any,
    *,
    expected: int,
    timeout: float = 8.0,
) -> list[int]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        rows = await asyncio.to_thread(
            control_engine.fetch_all,
            CompiledSQL(
                sql.SQL(
                    "SELECT DISTINCT activity.pid "
                    "FROM pg_locks AS locks "
                    "JOIN pg_stat_activity AS activity "
                    "ON activity.pid = locks.pid "
                    "JOIN pg_class AS relation ON relation.oid = locks.relation "
                    "JOIN pg_namespace AS ns ON ns.oid = relation.relnamespace "
                    "WHERE activity.datname = current_database() "
                    "AND activity.pid <> pg_backend_pid() "
                    "AND locks.granted IS FALSE "
                    "AND ns.nspname = %s "
                    "AND relation.relname = %s "
                    "ORDER BY activity.pid"
                ),
                (table.schema, table.name),
            ),
            operation="observe blocked chat inserts",
        )
        pids = [int(row[0]) for row in rows]
        if len(pids) >= expected:
            return pids
        if loop.time() >= deadline:
            raise TimeoutError(
                f"observed {len(pids)} chat lock waiter(s); expected {expected}"
            )
        await asyncio.sleep(0.02)


@pytest.mark.parametrize("case", MESSAGE_CASES, ids=lambda case: case["kind"])
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_fast
@pytest.mark.gaussdb_e2e_full
def test_chat_sync_message_type_field_order_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    temporary_chat_table: Any,
    e2e_namespace: Any,
) -> None:
    history = _history(
        writable_engine,
        temporary_chat_table,
        e2e_namespace.session_id(f"sync_{case['kind']}"),
    )
    expected = _rich_messages()
    history.add_messages(expected)
    restored = history.messages

    _assert_exact_messages(restored, expected)
    position = int(case["position"])
    assert message_to_dict(restored[position]) == message_to_dict(expected[position])


@pytest.mark.parametrize("case", MESSAGE_CASES, ids=lambda case: case["kind"])
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_fast
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_chat_async_message_type_field_order_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    temporary_chat_table: Any,
    e2e_namespace: Any,
) -> None:
    history = _history(
        writable_engine,
        temporary_chat_table,
        e2e_namespace.session_id(f"async_{case['kind']}"),
    )
    expected = _rich_messages()
    await history.aadd_messages(expected)
    restored = await history.aget_messages()

    _assert_exact_messages(restored, expected)
    position = int(case["position"])
    assert message_to_dict(restored[position]) == message_to_dict(expected[position])
    await history.aclear()
    assert await history.aget_messages() == []


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_exhaustive
@pytest.mark.asyncio
async def test_chat_sync_async_interleaving_preserves_global_id_order(
    writable_engine: GaussDBEngine,
    control_engine: GaussDBEngine,
    temporary_chat_table: Any,
    e2e_namespace: Any,
    resource_registry: Any,
) -> None:
    session_id = e2e_namespace.session_id("interleave")
    history = _history(writable_engine, temporary_chat_table, session_id)
    blocker, blocker_pid = _hold_table_lock(
        writable_engine,
        temporary_chat_table,
    )
    blocker_label = e2e_namespace.name("chat_interleave_blocker")
    resource_registry.register_session(blocker_label, blocker_pid)
    blocker_owned = True
    sync_outcome: queue.Queue[Any] = queue.Queue()
    sync_started = threading.Event()

    def sync_add() -> None:
        sync_started.set()
        try:
            history.add_messages(
                [
                    HumanMessage(content="sync-1", id="sync-1"),
                    HumanMessage(content="sync-2", id="sync-2"),
                ]
            )
            sync_outcome.put(("result", None))
        except BaseException as exc:
            sync_outcome.put(("error", exc))

    worker = threading.Thread(
        target=sync_add,
        name="gaussdb-e2e-chat-sync-add",
        daemon=True,
    )
    async_task: asyncio.Task[None] | None = None
    try:
        worker.start()
        assert sync_started.wait(timeout=2.0)
        async_task = asyncio.create_task(
            history.aadd_messages(
                [
                    AIMessage(content="async-1", id="async-1"),
                    AIMessage(content="async-2", id="async-2"),
                ]
            )
        )
        waiting_pids = await _wait_for_lock_waiters(
            control_engine,
            temporary_chat_table,
            expected=2,
        )
        assert blocker_pid not in waiting_pids
        assert worker.is_alive()
        assert not async_task.done()

        _release_table_lock(blocker)
        resource_registry.release_session(blocker_label, blocker_pid)
        blocker_owned = False

        await asyncio.wait_for(async_task, timeout=10.0)
        worker.join(timeout=10.0)
        assert not worker.is_alive()
        state, payload = sync_outcome.get_nowait()
        assert state == "result", repr(payload)

        rows = writable_engine.fetch_all(
            CompiledSQL(
                sql.SQL(
                    "SELECT id, message FROM {}.{} "
                    "WHERE session_id = %s ORDER BY id ASC"
                ).format(
                    sql.Identifier(temporary_chat_table.schema),
                    sql.Identifier(temporary_chat_table.name),
                ),
                (session_id,),
            ),
            operation="verify global chat id order",
        )
        raw_contents = [_payload_content(row[1]) for row in rows]
        restored_contents = [message.content for message in history.messages]
        assert [int(row[0]) for row in rows] == sorted(int(row[0]) for row in rows)
        assert restored_contents == raw_contents
        assert set(raw_contents) == {"sync-1", "sync-2", "async-1", "async-2"}
        assert len(raw_contents) == len(set(raw_contents)) == 4
    finally:
        if async_task is not None and not async_task.done():
            async_task.cancel()
            try:
                await asyncio.wait_for(async_task, timeout=5.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
        if blocker_owned:
            try:
                _release_table_lock(blocker, rollback=True)
            except BaseException:
                pass
            else:
                resource_registry.release_session(blocker_label, blocker_pid)
        if worker.is_alive():
            worker.join(timeout=10.0)


@pytest.mark.parametrize(
    "case",
    SESSION_CLEAR_CASES,
    ids=lambda case: case["path"],
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_chat_session_isolation_and_clear_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    temporary_chat_table: Any,
    e2e_namespace: Any,
) -> None:
    target = _history(
        writable_engine,
        temporary_chat_table,
        e2e_namespace.session_id(f"target_{_safe_role(case['path'])}"),
    )
    survivor = _history(
        writable_engine,
        temporary_chat_table,
        e2e_namespace.session_id(f"survivor_{_safe_role(case['path'])}"),
    )
    target_messages = [HumanMessage(content="target", id="target")]
    survivor_messages = [AIMessage(content="survivor", id="survivor")]
    if case["add_async"]:
        await target.aadd_messages(target_messages)
        await survivor.aadd_messages(survivor_messages)
    else:
        target.add_messages(target_messages)
        survivor.add_messages(survivor_messages)

    assert [message.content for message in target.messages] == ["target"]
    assert [message.content for message in await survivor.aget_messages()] == [
        "survivor"
    ]
    if case["clear_async"]:
        await target.aclear()
    else:
        target.clear()

    assert target.messages == []
    assert await target.aget_messages() == []
    assert [message.content for message in survivor.messages] == ["survivor"]
    assert [message.content for message in await survivor.aget_messages()] == [
        "survivor"
    ]


@pytest.mark.parametrize(
    "case",
    MALFORMED_ROW_CASES,
    ids=lambda case: case["state"],
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_malformed_structural_jsonb_row_fails_without_payload_leak(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    temporary_chat_table: Any,
    e2e_namespace: Any,
) -> None:
    session_id = e2e_namespace.session_id(f"malformed_{_safe_role(case['state'])}")
    history = _history(writable_engine, temporary_chat_table, session_id)
    serialized = json.dumps(case["payload"], separators=(",", ":"))
    writable_engine.execute(
        CompiledSQL(
            sql.SQL(
                "INSERT INTO {}.{} (session_id, message) VALUES (%s, %s::jsonb)"
            ).format(
                sql.Identifier(temporary_chat_table.schema),
                sql.Identifier(temporary_chat_table.name),
            ),
            (session_id, serialized),
        ),
        operation="insert malformed structural chat JSONB",
    )

    with pytest.raises(ValueError) as caught:
        _ = history.messages
    error = str(caught.value)
    assert "row 0" in error
    assert str(case["error"]) in error
    assert serialized not in error
    assert "secret" not in error
    assert "password" not in error.lower()
    assert "GAUSSDB_TEST_DSN" not in error

    history.clear()
    assert history.messages == []
    assert writable_engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 1")),
        operation="verify chat recovery after malformed JSONB",
    ) == [(1,)]


@pytest.mark.parametrize(
    "case",
    CLOSE_OWNERSHIP_CASES,
    ids=lambda case: case["ownership"],
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_chat_owned_and_external_engine_close_matrix(
    case: dict[str, str],
    writable_engine: GaussDBEngine,
    temporary_chat_table: Any,
    e2e_namespace: Any,
) -> None:
    session_id = e2e_namespace.session_id(f"close_{case['ownership']}")
    seed = _history(writable_engine, temporary_chat_table, session_id)
    seed.add_messages([HumanMessage(content="ownership", id="ownership")])

    if case["ownership"] == "owned":
        dsn = os.environ.get("GAUSSDB_TEST_DSN")
        assert dsn
        history = GaussDBChatMessageHistory(
            dsn=dsn,
            schema_name=temporary_chat_table.schema,
            table_name=temporary_chat_table.name,
            session_id=session_id,
        )
        owned_engine = history._engine
        try:
            assert [message.content for message in history.messages] == ["ownership"]
        finally:
            history.close()
        assert owned_engine._closed is True
        with pytest.raises(GaussDBConnectionError, match="closed"):
            owned_engine.fetch_all(CompiledSQL(sql.SQL("SELECT 1")))
    else:
        assert case["ownership"] == "external"
        history = _history(writable_engine, temporary_chat_table, session_id)
        assert [message.content for message in history.messages] == ["ownership"]
        history.close()
        assert writable_engine.fetch_all(
            CompiledSQL(sql.SQL("SELECT 1")),
            operation="verify external chat Engine survives history close",
        ) == [(1,)]


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_exhaustive
@pytest.mark.asyncio
async def test_cancelled_async_add_finishes_executor_write_and_history_remains_usable(
    writable_engine_factory: Any,
    control_engine: GaussDBEngine,
    temporary_chat_table: Any,
    e2e_namespace: Any,
    resource_registry: Any,
) -> None:
    engine = writable_engine_factory(minconn=2, maxconn=3)
    session_id = e2e_namespace.session_id("cancel_add")
    history = _history(engine, temporary_chat_table, session_id)
    blocker, blocker_pid = _hold_table_lock(engine, temporary_chat_table)
    blocker_label = e2e_namespace.name("chat_cancel_blocker")
    resource_registry.register_session(blocker_label, blocker_pid)
    blocker_owned = True
    task: asyncio.Task[None] | None = None
    try:
        task = asyncio.create_task(
            history.aadd_messages(
                [
                    HumanMessage(
                        content="continues-after-cancel",
                        id="cancelled-waiter",
                    )
                ]
            )
        )
        waiting_pids = await _wait_for_lock_waiters(
            control_engine,
            temporary_chat_table,
            expected=1,
        )
        assert blocker_pid not in waiting_pids
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=8.0)
        assert task.cancelled()

        _release_table_lock(blocker)
        resource_registry.release_session(blocker_label, blocker_pid)
        blocker_owned = False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 8.0
        while True:
            committed = await history.aget_messages()
            if [message.content for message in committed] == ["continues-after-cancel"]:
                break
            if loop.time() >= deadline:
                raise TimeoutError("cancelled async add executor work did not settle")
            await asyncio.sleep(0.02)
        await history.aadd_messages([HumanMessage(content="recovered", id="recovered")])
        assert [message.content for message in await history.aget_messages()] == [
            "continues-after-cancel",
            "recovered",
        ]
        assert engine.fetch_all(
            CompiledSQL(sql.SQL("SELECT 1")),
            operation="verify connection after cancelled chat add",
        ) == [(1,)]
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
        if blocker_owned:
            try:
                _release_table_lock(blocker, rollback=True)
            except BaseException:
                pass
            else:
                resource_registry.release_session(blocker_label, blocker_pid)


def _chat_index_name(table_name: str) -> str:
    name = f"{table_name}_session_id_id_idx"
    assert len(name.encode("utf-8")) <= 63
    return name


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_fast
@pytest.mark.gaussdb_e2e_full
def test_existing_wrong_chat_table_fails_before_index_creation(
    writable_engine: GaussDBEngine,
    temporary_schema: str,
    e2e_namespace: Any,
) -> None:
    table = e2e_namespace.table(temporary_schema, "chat_bad_definition")
    writable_engine.execute(
        CompiledSQL(
            sql.SQL(
                "CREATE TABLE {}.{} ("
                "id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL, "
                "message TEXT NOT NULL) WITH (storage_type=ustore)"
            ).format(
                sql.Identifier(table.schema),
                sql.Identifier(table.name),
            )
        ),
        operation="create mismatched SR-15 chat table",
    )

    with pytest.raises(GaussDBCapabilityError, match="table|definition"):
        GaussDBChatMessageHistory(
            engine=writable_engine,
            schema_name=table.schema,
            table_name=table.name,
            session_id=e2e_namespace.session_id("chat_bad_definition"),
            create_table=True,
        )

    index_name = _chat_index_name(table.name)
    rows = writable_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT 1 FROM pg_class AS idx "
                "JOIN pg_namespace AS ns ON ns.oid = idx.relnamespace "
                "WHERE ns.nspname = %s AND idx.relname = %s "
                "AND idx.relkind = 'i'"
            ),
            (table.schema, index_name),
        ),
        operation="verify mismatched chat table created no index",
    )
    assert rows == []
    assert writable_engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 1")),
        operation="verify Engine after chat table mismatch",
    ) == [(1,)]


@pytest.mark.parametrize(
    "case",
    CHAT_INDEX_MISMATCH_CASES,
    ids=lambda case: case["state"],
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_chat_table_index_definition_mismatch_fails_closed(
    case: dict[str, str],
    writable_engine: GaussDBEngine,
    temporary_chat_table: Any,
    temporary_schema: str,
    e2e_namespace: Any,
    resource_registry: Any,
) -> None:
    index_name = _chat_index_name(temporary_chat_table.name)
    resource_registry.register_index(temporary_chat_table.schema, index_name)
    indexed_table = temporary_chat_table
    if case["state"] == "wrong-table":
        indexed_table = e2e_namespace.table(temporary_schema, "chat_wrong_target")
        writable_engine.execute(
            CompiledSQL(
                sql.SQL(
                    "CREATE TABLE {}.{} ("
                    "id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL, "
                    "message JSONB NOT NULL) WITH (storage_type=ustore)"
                ).format(
                    sql.Identifier(indexed_table.schema),
                    sql.Identifier(indexed_table.name),
                )
            ),
            operation="create wrong target chat table",
        )

    columns = {
        "wrong-order": ("id", "session_id"),
        "wrong-column": ("session_id",),
        "wrong-table": ("session_id", "id"),
    }[case["state"]]
    writable_engine.execute(
        CompiledSQL(
            sql.SQL("CREATE INDEX {} ON {}.{} ({})").format(
                sql.Identifier(index_name),
                sql.Identifier(indexed_table.schema),
                sql.Identifier(indexed_table.name),
                sql.SQL(", ").join(sql.Identifier(column) for column in columns),
            )
        ),
        operation="create same-name wrong-definition chat index",
    )
    history = _history(
        writable_engine,
        temporary_chat_table,
        e2e_namespace.session_id(f"index_{_safe_role(case['state'])}"),
    )

    with pytest.raises(
        GaussDBCapabilityError,
        match="index|definition|session_id|table",
    ):
        history.create_table_if_not_exists()

    rows = writable_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT tab.relname, pg_get_indexdef(ind.indexrelid) "
                "FROM pg_index AS ind "
                "JOIN pg_class AS idx ON idx.oid = ind.indexrelid "
                "JOIN pg_class AS tab ON tab.oid = ind.indrelid "
                "JOIN pg_namespace AS ns ON ns.oid = idx.relnamespace "
                "WHERE ns.nspname = %s AND idx.relname = %s"
            ),
            (temporary_chat_table.schema, index_name),
        ),
        operation="verify wrong chat index was not accepted or replaced",
    )
    assert len(rows) == 1
    assert str(rows[0][0]) == indexed_table.name
    assert writable_engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 1")),
        operation="verify Engine after chat index mismatch",
    ) == [(1,)]


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_chat_and_vectorstore_share_engine_without_state_leak(
    writable_engine: GaussDBEngine,
    temporary_chat_table: Any,
    temporary_vector_table: Any,
    e2e_namespace: Any,
    deterministic_embeddings: Any,
    requires_gsdiskann: bool,
) -> None:
    session_id = e2e_namespace.session_id("shared_engine")
    history = _history(writable_engine, temporary_chat_table, session_id)
    store = GaussDBVectorStore(
        engine=writable_engine,
        embedding=deterministic_embeddings,
        schema_name=temporary_vector_table.schema,
        table_name=temporary_vector_table.name,
        embedding_dimension=3,
    )

    await history.aadd_messages([HumanMessage(content="chat-only", id="chat-only")])
    assert await store.aadd_texts(
        ["vector-only"],
        metadatas=[{"channel": "vector"}],
        ids=["vector-only"],
    ) == ["vector-only"]
    documents = await store.asimilarity_search("vector-only", k=1)
    messages = await history.aget_messages()

    assert [str(document.id) for document in documents] == ["vector-only"]
    assert [document.page_content for document in documents] == ["vector-only"]
    assert [message.content for message in messages] == ["chat-only"]
    chat_count = writable_engine.fetch_all(
        CompiledSQL(
            sql.SQL("SELECT count(*) FROM {}.{} WHERE session_id = %s").format(
                sql.Identifier(temporary_chat_table.schema),
                sql.Identifier(temporary_chat_table.name),
            ),
            (session_id,),
        ),
        operation="verify shared Engine chat isolation",
    )
    vector_count = writable_engine.fetch_all(
        CompiledSQL(
            sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(temporary_vector_table.schema),
                sql.Identifier(temporary_vector_table.name),
            )
        ),
        operation="verify shared Engine vector isolation",
    )
    assert chat_count == [(1,)]
    assert vector_count == [(1,)]

    assert writable_engine.check_connection() is True
    assert writable_engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 1")),
        operation="verify shared Chat/Vector Engine remains reusable",
    ) == [(1,)]
