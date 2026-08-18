from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables.config import run_in_executor
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine
from langchain_gaussdb.errors import (
    GaussDBConnectionError,
    GaussDBError,
    GaussDBSQLError,
)
from langchain_gaussdb.sql import CompiledSQL

pytestmark = [pytest.mark.gaussdb_e2e, pytest.mark.gaussdb_e2e_full]


def _dedicated_engine(
    writable_engine_factory: Callable[..., GaussDBEngine],
    resource_registry: Any,
    e2e_namespace: Any,
    role: str,
    *,
    maxconn: int = 1,
) -> GaussDBEngine:
    engine = writable_engine_factory(minconn=1, maxconn=maxconn)
    return resource_registry.register_engine(
        e2e_namespace.name(role),
        engine,
    )


def _create_table(
    engine: GaussDBEngine,
    schema_name: str,
    resource_registry: Any,
    e2e_namespace: Any,
    role: str,
) -> str:
    table_name = e2e_namespace.name(role)
    resource_registry.register_table(schema_name, table_name)
    engine.execute(
        CompiledSQL(
            sql.SQL(
                "CREATE TABLE {}.{} (id TEXT PRIMARY KEY, value TEXT NOT NULL)"
                " WITH (storage_type=ustore)"
            ).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
            )
        ),
        operation="create Engine lifecycle witness table",
    )
    return table_name


def _count_rows(
    engine: GaussDBEngine,
    schema_name: str,
    table_name: str,
) -> int:
    rows = engine.fetch_all(
        CompiledSQL(
            sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
            )
        ),
        operation="count Engine lifecycle witness rows",
    )
    return int(rows[0][0])


def _wait_for_active_query(
    control_engine: GaussDBEngine,
    backend_pid: int,
    query_fragment: str,
) -> None:
    delay = threading.Event()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rows = control_engine.fetch_all(
            CompiledSQL(
                sql.SQL(
                    "SELECT state, position(%s in query) > 0 "
                    "FROM pg_stat_activity WHERE pid = %s"
                ),
                (query_fragment, backend_pid),
            ),
            operation="observe active E2E backend query",
        )
        if rows == [("active", True)]:
            return
        delay.wait(0.01)
    pytest.fail("backend did not expose the expected active query state")


async def _wait_for_thread_event(event: threading.Event) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    pytest.fail("executor worker did not reach the expected state")


@pytest.mark.gaussdb_e2e_fast
def test_sync_transaction_success_commits_and_reuses_connection(
    writable_engine_factory: Callable[..., GaussDBEngine],
    resource_registry: Any,
    e2e_namespace: Any,
    temporary_schema: str,
) -> None:
    engine = _dedicated_engine(
        writable_engine_factory,
        resource_registry,
        e2e_namespace,
        "commit_engine",
    )
    table_name = _create_table(
        engine,
        temporary_schema,
        resource_registry,
        e2e_namespace,
        "commit_rows",
    )

    def insert(cursor: Any) -> int:
        cursor.execute("SELECT pg_backend_pid()")
        pid = int(cursor.fetchone()[0])
        cursor.execute(
            sql.SQL("INSERT INTO {}.{} (id, value) VALUES (%s, %s)").format(
                sql.Identifier(temporary_schema),
                sql.Identifier(table_name),
            ),
            ("committed", "visible"),
        )
        return pid

    transaction_pid = engine.transaction(
        insert,
        operation="synchronous commit witness",
    )
    reused_pid = int(
        engine.fetch_all(
            CompiledSQL(sql.SQL("SELECT pg_backend_pid()")),
            operation="verify synchronous connection reuse",
        )[0][0]
    )

    assert _count_rows(engine, temporary_schema, table_name) == 1
    assert reused_pid == transaction_pid


def test_sync_transaction_business_error_rolls_back_and_reuses_connection(
    writable_engine_factory: Callable[..., GaussDBEngine],
    resource_registry: Any,
    e2e_namespace: Any,
    temporary_schema: str,
) -> None:
    engine = _dedicated_engine(
        writable_engine_factory,
        resource_registry,
        e2e_namespace,
        "rollback_engine",
    )
    table_name = _create_table(
        engine,
        temporary_schema,
        resource_registry,
        e2e_namespace,
        "rollback_rows",
    )
    observed_pid: list[int] = []

    class BusinessError(RuntimeError):
        pass

    failure = BusinessError("business failure")

    def fail(cursor: Any) -> None:
        cursor.execute("SELECT pg_backend_pid()")
        observed_pid.append(int(cursor.fetchone()[0]))
        cursor.execute(
            sql.SQL("INSERT INTO {}.{} (id, value) VALUES (%s, %s)").format(
                sql.Identifier(temporary_schema),
                sql.Identifier(table_name),
            ),
            ("rolled-back", "invisible"),
        )
        raise failure

    with pytest.raises(BusinessError, match="business failure") as captured:
        engine.transaction(
            fail,
            operation="synchronous rollback witness",
        )

    assert captured.value is failure
    reused_pid = int(
        engine.fetch_all(
            CompiledSQL(sql.SQL("SELECT pg_backend_pid()")),
            operation="verify rollback connection reuse",
        )[0][0]
    )
    assert _count_rows(engine, temporary_schema, table_name) == 0
    assert reused_pid == observed_pid[0]


@pytest.mark.asyncio
async def test_executor_cancellation_does_not_cancel_sync_transaction(
    writable_engine_factory: Callable[..., GaussDBEngine],
    resource_registry: Any,
    e2e_namespace: Any,
    temporary_schema: str,
) -> None:
    engine = _dedicated_engine(
        writable_engine_factory,
        resource_registry,
        e2e_namespace,
        "cancel_engine",
    )
    table_name = _create_table(
        engine,
        temporary_schema,
        resource_registry,
        e2e_namespace,
        "cancel_rows",
    )
    transaction_entered = threading.Event()
    release_transaction = threading.Event()
    worker_done = threading.Event()

    def insert_then_wait(cursor: Any) -> None:
        cursor.execute(
            sql.SQL("INSERT INTO {}.{} (id, value) VALUES (%s, %s)").format(
                sql.Identifier(temporary_schema),
                sql.Identifier(table_name),
            ),
            ("cancelled-waiter", "executor-keeps-running"),
        )
        transaction_entered.set()
        assert release_transaction.wait(timeout=5)

    def run_transaction() -> None:
        try:
            engine.transaction(
                insert_then_wait,
                operation="executor cancellation witness",
            )
        finally:
            worker_done.set()

    task = asyncio.create_task(run_in_executor(None, run_transaction))
    try:
        await _wait_for_thread_event(transaction_entered)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_transaction.set()

    await _wait_for_thread_event(worker_done)
    assert _count_rows(engine, temporary_schema, table_name) == 1


def test_terminated_backend_discards_connection_and_replenishes_pool(
    writable_engine_factory: Callable[..., GaussDBEngine],
    resource_registry: Any,
    e2e_namespace: Any,
    control_engine: GaussDBEngine,
    requires_backend_termination_privilege: bool,
) -> None:
    assert requires_backend_termination_privilege
    engine = _dedicated_engine(
        writable_engine_factory,
        resource_registry,
        e2e_namespace,
        "discard_engine",
    )
    backend_pid_ready = threading.Event()
    observed_pid: list[int] = []
    observed_connections: list[Any] = []
    failures: queue.Queue[BaseException] = queue.Queue()

    def wait(cursor: Any) -> None:
        observed_connections.append(cursor.connection)
        cursor.execute("SELECT pg_backend_pid()")
        observed_pid.append(int(cursor.fetchone()[0]))
        backend_pid_ready.set()
        cursor.execute("SELECT pg_sleep(30)")

    def run_transaction() -> None:
        try:
            engine.transaction(
                wait,
                operation="terminated backend recovery witness",
            )
        except BaseException as exc:
            failures.put(exc)

    worker = threading.Thread(target=run_transaction)
    worker.start()
    assert backend_pid_ready.wait(timeout=5)
    _wait_for_active_query(control_engine, observed_pid[0], "pg_sleep")
    terminated = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL("SELECT pg_terminate_backend(%s)"),
            (observed_pid[0],),
        ),
        operation="terminate E2E backend",
    )
    assert terminated == [(True,)]
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert not failures.empty()
    assert isinstance(failures.get_nowait(), GaussDBError)
    old_connection = observed_connections[0]
    assert old_connection.closed

    with engine.connection() as replacement_connection:
        assert replacement_connection is not old_connection
        cursor = replacement_connection.cursor()
        try:
            cursor.execute("SELECT 1")
            assert cursor.fetchall() == [(1,)]
        finally:
            cursor.close()

    assert engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 2")),
        operation="verify replenished pool remains reusable",
    ) == [(2,)]


def test_sync_connection_context_cursor_roundtrip(
    writable_engine: GaussDBEngine,
) -> None:
    with writable_engine.connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT value FROM (VALUES (%s), (%s)) AS witness(value) "
                "ORDER BY value",
                (7, 8),
            )
            assert cursor.rowcount == 2
            assert cursor.description is not None
            assert cursor.fetchall() == [(7,), (8,)]
        finally:
            cursor.close()
        assert cursor.closed

    assert not connection.closed


def test_connection_context_rolls_back_uncommitted_work_before_reuse(
    writable_engine: GaussDBEngine,
    resource_registry: Any,
    e2e_namespace: Any,
    temporary_schema: str,
) -> None:
    table_name = _create_table(
        writable_engine,
        temporary_schema,
        resource_registry,
        e2e_namespace,
        "context_rollback_rows",
    )

    with writable_engine.connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                sql.SQL("INSERT INTO {}.{} (id, value) VALUES (%s, %s)").format(
                    sql.Identifier(temporary_schema),
                    sql.Identifier(table_name),
                ),
                ("uncommitted", "must-roll-back"),
            )
        finally:
            cursor.close()

    assert not connection.closed
    assert _count_rows(writable_engine, temporary_schema, table_name) == 0
    assert writable_engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 2")),
        operation="verify Engine reuse after raw connection rollback",
    ) == [(2,)]


@pytest.mark.gaussdb_e2e_fast
def test_read_only_session_reports_sqlstate_25006_without_policy_change(
    gaussdb_engine_factory: Callable[..., GaussDBEngine],
    writable_engine: GaussDBEngine,
    resource_registry: Any,
    e2e_namespace: Any,
    temporary_schema: str,
) -> None:
    table_name = _create_table(
        writable_engine,
        temporary_schema,
        resource_registry,
        e2e_namespace,
        "readonly_rows",
    )
    engine = gaussdb_engine_factory(minconn=1, maxconn=1, read_only=True)

    def insert(cursor: Any) -> None:
        cursor.execute(
            sql.SQL("INSERT INTO {}.{} (id, value) VALUES (%s, %s)").format(
                sql.Identifier(temporary_schema),
                sql.Identifier(table_name),
            ),
            ("blocked", "read-only"),
        )

    with pytest.raises(GaussDBSQLError) as captured:
        engine.transaction(insert, operation="read-only policy witness")

    assert captured.value.sqlstate == "25006"
    assert engine.fetch_all(
        CompiledSQL(sql.SQL("SHOW default_transaction_read_only")),
        operation="verify unchanged read-only policy",
    ) == [("on",)]
    assert _count_rows(writable_engine, temporary_schema, table_name) == 0


def test_close_waits_for_active_operation_and_rejects_new_work(
    writable_engine_factory: Callable[..., GaussDBEngine],
    resource_registry: Any,
    e2e_namespace: Any,
) -> None:
    engine = _dedicated_engine(
        writable_engine_factory,
        resource_registry,
        e2e_namespace,
        "close_wait_engine",
    )
    operation_entered = threading.Event()
    release_operation = threading.Event()
    operation_done = threading.Event()
    operation_outcome: queue.Queue[BaseException] = queue.Queue()
    close_outcome: queue.Queue[BaseException] = queue.Queue()

    def hold(cursor: Any) -> None:
        cursor.execute("SELECT 1")
        operation_entered.set()
        assert release_operation.wait(timeout=5)

    def run_operation() -> None:
        try:
            engine.transaction(hold, operation="active close witness")
        except BaseException as exc:
            operation_outcome.put(exc)
        finally:
            operation_done.set()

    def close_engine() -> None:
        try:
            engine.close()
        except BaseException as exc:
            close_outcome.put(exc)

    operation_thread = threading.Thread(target=run_operation)
    close_thread = threading.Thread(target=close_engine)
    operation_thread.start()
    assert operation_entered.wait(timeout=5)
    close_thread.start()

    with engine._lifecycle:
        assert engine._closing
        assert not engine._closed

    with pytest.raises(GaussDBConnectionError, match="closing"):
        engine.fetch_all(CompiledSQL(sql.SQL("SELECT 1")))
    release_operation.set()
    operation_thread.join(timeout=5)
    close_thread.join(timeout=5)

    assert operation_done.is_set()
    assert not operation_thread.is_alive()
    assert not close_thread.is_alive()
    if not operation_outcome.empty():
        raise operation_outcome.get_nowait()
    if not close_outcome.empty():
        raise close_outcome.get_nowait()
    assert engine._closed


def test_concurrent_close_callers_share_success_result(
    writable_engine_factory: Callable[..., GaussDBEngine],
    resource_registry: Any,
    e2e_namespace: Any,
) -> None:
    engine = _dedicated_engine(
        writable_engine_factory,
        resource_registry,
        e2e_namespace,
        "concurrent_close_engine",
        maxconn=2,
    )
    barrier = threading.Barrier(3)
    outcomes: queue.Queue[str] = queue.Queue()
    failures: queue.Queue[BaseException] = queue.Queue()

    def close_engine() -> None:
        try:
            barrier.wait(timeout=5)
            engine.close()
            outcomes.put("closed")
        except BaseException as exc:
            failures.put(exc)

    callers = [threading.Thread(target=close_engine) for _ in range(2)]
    for caller in callers:
        caller.start()
    barrier.wait(timeout=5)
    for caller in callers:
        caller.join(timeout=5)

    if not failures.empty():
        raise failures.get_nowait()
    assert [outcomes.get_nowait(), outcomes.get_nowait()] == ["closed", "closed"]
    assert all(not caller.is_alive() for caller in callers)
    assert engine._closed
    engine.close()
