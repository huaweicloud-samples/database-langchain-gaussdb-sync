import threading
from collections import deque

import pytest
from psycopg2 import sql

import langchain_gaussdb.engine as engine_module
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.errors import (
    GaussDBConnectionError,
    GaussDBSQLError,
    GaussDBTransactionError,
)
from langchain_gaussdb.sql import CompiledSQL


class FakePsycopgError(Exception):
    pgcode = "XX001"


class PrivatePayloadBaseException(BaseException):
    def __init__(self, payload):
        super().__init__("opaque")
        self.private_payload = payload

    def __str__(self):
        return f"private payload: {self.private_payload}"


class ConstructibleKeyboardInterrupt(KeyboardInterrupt):
    def __init__(self, payload):
        super().__init__(payload)
        self.private_payload = payload


class ConstructibleSystemExit(SystemExit):
    def __init__(self, payload):
        super().__init__(payload)
        self.private_payload = payload


class RecordingCursor:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False
        self.rowcount = -1
        self.description = None
        self._rows = []

    def execute(self, statement, params=None):
        text = statement.string if isinstance(statement, sql.SQL) else str(statement)
        values = tuple(params or ())
        self.connection.executed.append((text, values))
        failure = self.connection.failures.get(text)
        if failure:
            raise failure.popleft()
        self._rows = list(self.connection.rows_by_sql.get(text, self.connection.rows))
        self.rowcount = len(self._rows)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        if self.connection.cursor_close_failure is not None:
            raise self.connection.cursor_close_failure
        self.closed = True


class RecordingConnection:
    def __init__(self, rows=None):
        self.rows = [("ok",)] if rows is None else list(rows)
        self.rows_by_sql = {}
        self.executed = []
        self.failures = {}
        self.cursors = []
        self.commits = 0
        self.rollbacks = 0
        self.commit_failure = None
        self.rollback_failure = None
        self.cursor_failure = None
        self.cursor_close_failure = None
        self.closed = False

    def cursor(self):
        if self.cursor_failure is not None:
            raise self.cursor_failure
        cursor = RecordingCursor(self)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commits += 1
        if self.commit_failure is not None:
            raise self.commit_failure

    def rollback(self):
        self.rollbacks += 1
        if self.rollback_failure is not None:
            raise self.rollback_failure

    def close(self):
        self.closed = True

    def fail_next(self, statement, error):
        self.failures.setdefault(statement, deque()).append(error)


class RecordingPool:
    def __init__(self, connections):
        self.connections = deque(connections)
        self.borrowed = []
        self.returned = []
        self.discarded = []
        self.closed = False
        self.getconn_failure = None
        self.putconn_failure = None

    def getconn(self):
        if self.getconn_failure is not None:
            raise self.getconn_failure
        connection = self.connections.popleft()
        self.borrowed.append(connection)
        return connection

    def putconn(self, connection, key=None, close=False):
        if self.putconn_failure is not None:
            raise self.putconn_failure
        if close:
            connection.close()
            self.discarded.append(connection)
            return
        self.returned.append(connection)
        self.connections.append(connection)

    def closeall(self):
        self.closed = True


class FailOnReentrantAcquireSemaphore:
    """Make a same-thread second acquire observable without blocking the test."""

    def __init__(self):
        self.acquired = False
        self.reentrant_acquires = 0

    def acquire(self):
        if self.acquired:
            self.reentrant_acquires += 1
            raise AssertionError("engine attempted a second semaphore acquire")
        self.acquired = True
        return True

    def release(self):
        if not self.acquired:
            raise AssertionError("semaphore released without an acquire")
        self.acquired = False


def assert_exception_graph_is_redacted(
    error: BaseException,
    secret: str,
) -> None:
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        assert secret not in repr(getattr(current, "__notes__", ()))
        assert secret not in repr(vars(current))
        if isinstance(current, SystemExit):
            assert secret not in repr(current.code)
        for related in (current.__cause__, current.__context__):
            if related is not None:
                pending.append(related)


@pytest.fixture
def sync_engine(monkeypatch):
    connection = RecordingConnection(rows=[("one",), ("two",)])
    pool = RecordingPool([connection])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")
    yield engine, pool, connection
    engine.close()


def test_execute_uses_sync_cursor_and_connection_commit(sync_engine) -> None:
    engine, _, connection = sync_engine

    engine.execute(CompiledSQL(sql.SQL("SELECT %s"), [1]))

    assert connection.executed == [("SELECT %s", (1,))]
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.cursors[0].closed is True


def test_fetch_all_returns_rows_and_commits(sync_engine) -> None:
    engine, _, connection = sync_engine

    rows = engine.fetch_all(CompiledSQL(sql.SQL("SELECT %s"), [1]))

    assert rows == [("one",), ("two",)]
    assert connection.commits == 1


def test_transaction_callback_runs_on_caller_thread(sync_engine) -> None:
    engine, _, connection = sync_engine
    caller_thread = threading.get_ident()
    callback_threads = []

    def callback(cursor):
        callback_threads.append(threading.get_ident())
        cursor.execute(sql.SQL("SELECT %s"), [1])
        return cursor.fetchall()

    rows = engine.transaction(callback)

    assert rows == [("one",), ("two",)]
    assert callback_threads == [caller_thread]
    assert connection.commits == 1


@pytest.mark.parametrize(
    "nested_operation",
    ["execute", "fetch_all", "transaction", "connection"],
)
def test_transaction_rejects_same_engine_reentry_and_remains_reusable(
    sync_engine,
    nested_operation,
) -> None:
    engine, _, connection = sync_engine
    semaphore = FailOnReentrantAcquireSemaphore()
    engine._pool_semaphore = semaphore

    def reenter_engine(cursor):
        if nested_operation == "execute":
            engine.execute(CompiledSQL(sql.SQL("SELECT nested")))
        elif nested_operation == "fetch_all":
            engine.fetch_all(CompiledSQL(sql.SQL("SELECT nested")))
        elif nested_operation == "transaction":
            engine.transaction(lambda nested_cursor: None)
        else:
            with engine.connection():
                pass

    with pytest.raises(GaussDBConnectionError, match="same-thread reentrant"):
        engine.transaction(reenter_engine, operation="outer transaction")

    assert semaphore.reentrant_acquires == 0
    assert connection.rollbacks == 1
    assert engine.fetch_all(CompiledSQL(sql.SQL("SELECT reusable"))) == [
        ("one",),
        ("two",),
    ]


def test_transaction_rolls_back_and_preserves_application_error(sync_engine) -> None:
    engine, _, connection = sync_engine
    application_error = RuntimeError("application failed")

    def fail(cursor):
        raise application_error

    with pytest.raises(RuntimeError) as exc_info:
        engine.transaction(fail, operation="application callback")

    assert exc_info.value is application_error
    assert connection.rollbacks == 1
    assert connection.commits == 0


def test_cursor_close_failure_does_not_mask_application_error(sync_engine) -> None:
    engine, _, connection = sync_engine
    application_error = RuntimeError("application failed")
    connection.cursor_close_failure = RuntimeError("cursor close failed")

    def fail(cursor):
        raise application_error

    with pytest.raises(RuntimeError) as exc_info:
        engine.transaction(fail)

    assert exc_info.value is application_error
    assert connection.rollbacks == 1


def test_cursor_close_failure_after_success_is_wrapped(sync_engine) -> None:
    engine, _, connection = sync_engine
    connection.cursor_close_failure = FakePsycopgError("cursor close failed")

    with pytest.raises(GaussDBTransactionError, match="cursor close"):
        engine.transaction(lambda cursor: "ok", operation="write docs")

    assert connection.commits == 1


def test_sql_error_is_wrapped_with_sqlstate(sync_engine) -> None:
    engine, _, connection = sync_engine
    connection.fail_next("SELECT %s", FakePsycopgError("bad SQL"))

    with pytest.raises(GaussDBSQLError) as exc_info:
        engine.execute(CompiledSQL(sql.SQL("SELECT %s"), [1]))

    assert exc_info.value.sqlstate == "XX001"
    assert connection.rollbacks == 1


def test_commit_error_attempts_rollback_and_reports_commit(sync_engine) -> None:
    engine, _, connection = sync_engine
    connection.commit_failure = FakePsycopgError("commit failed")

    with pytest.raises(GaussDBTransactionError, match="commit"):
        engine.execute(CompiledSQL(sql.SQL("SELECT %s"), [1]))

    assert connection.rollbacks == 1


def test_rollback_error_discards_connection_and_next_operation_recovers(
    monkeypatch,
) -> None:
    broken = RecordingConnection()
    healthy = RecordingConnection(rows=[("healthy",)])
    broken.fail_next("SELECT broken", FakePsycopgError("query failed"))
    broken.rollback_failure = FakePsycopgError("rollback failed")
    pool = RecordingPool([broken, healthy])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake", minconn=1, maxconn=1)
    try:
        with pytest.raises(GaussDBTransactionError, match="rollback"):
            engine.execute(CompiledSQL(sql.SQL("SELECT broken")))
        rows = engine.fetch_all(CompiledSQL(sql.SQL("SELECT healthy")))
    finally:
        engine.close()

    assert broken.closed is True
    assert pool.discarded == [broken]
    assert rows == [("healthy",)]


def test_cursor_creation_error_rolls_back_and_discards_connection(
    monkeypatch,
) -> None:
    broken = RecordingConnection()
    healthy = RecordingConnection()
    broken.cursor_failure = FakePsycopgError("cursor failed")
    pool = RecordingPool([broken, healthy])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")
    try:
        with pytest.raises(GaussDBTransactionError, match="cursor"):
            engine.execute(CompiledSQL(sql.SQL("SELECT 1")))
    finally:
        engine.close()

    assert broken.rollbacks == 1
    assert broken.closed is True
    assert pool.discarded == [broken]


def test_borrow_failure_is_connection_error(monkeypatch) -> None:
    pool = RecordingPool([RecordingConnection()])
    pool.getconn_failure = RuntimeError("pool exhausted")
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")
    try:
        with pytest.raises(GaussDBConnectionError, match="borrow connection"):
            engine.check_connection()
    finally:
        engine.close()


def test_return_failure_reports_primary_application_error(monkeypatch) -> None:
    connection = RecordingConnection()
    pool = RecordingPool([connection])
    pool.putconn_failure = RuntimeError("return failed")
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")
    try:
        with pytest.raises(GaussDBConnectionError) as exc_info:
            engine.transaction(
                lambda cursor: (_ for _ in ()).throw(RuntimeError("application failed"))
            )
    finally:
        pool.putconn_failure = None
        engine.close()

    message = str(exc_info.value)
    assert "return connection" in message
    assert "application failed" in message
    assert connection.closed is True


def test_connection_context_yields_raw_sync_connection(sync_engine) -> None:
    engine, _, raw_connection = sync_engine

    with engine.connection() as connection:
        assert connection is raw_connection
        cursor = connection.cursor()
        cursor.execute(sql.SQL("SELECT %s"), [1])
        assert cursor.fetchall() == [("one",), ("two",)]


@pytest.mark.parametrize("connection_state", ["closed", "unknown"])
def test_unusable_connection_is_discarded_without_poisoning_engine(
    monkeypatch,
    connection_state,
) -> None:
    broken = RecordingConnection()
    healthy = RecordingConnection(rows=[("healthy",)])
    pool = RecordingPool([broken, healthy])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake", minconn=1, maxconn=1)
    try:
        with engine.connection() as connection:
            if connection_state == "closed":
                connection.closed = True
            elif connection_state == "unknown":
                connection.get_transaction_status = lambda: (
                    engine_module.psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN
                )
        rows = engine.fetch_all(CompiledSQL(sql.SQL("SELECT healthy")))
    finally:
        engine.close()

    assert pool.discarded == [broken]
    assert broken.rollbacks == 0
    assert rows == [("healthy",)]


@pytest.mark.parametrize(
    "recovery_error",
    [
        RuntimeError("recovery failed"),
        KeyboardInterrupt("recovery interrupted"),
        SystemExit("recovery exited"),
    ],
)
def test_recovery_failure_discards_then_reraises_and_engine_recovers(
    monkeypatch,
    recovery_error,
) -> None:
    broken = RecordingConnection()
    healthy = RecordingConnection(rows=[("healthy",)])
    broken.get_transaction_status = lambda: (
        engine_module.psycopg2.extensions.TRANSACTION_STATUS_INTRANS
    )
    broken.rollback_failure = recovery_error
    pool = RecordingPool([broken, healthy])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake", minconn=1, maxconn=1)
    try:
        with pytest.raises(type(recovery_error)) as exc_info:
            with engine.connection():
                pass

        assert exc_info.value is not recovery_error
        assert type(exc_info.value) is type(recovery_error)
        rows = engine.fetch_all(CompiledSQL(sql.SQL("SELECT healthy")))
    finally:
        engine.close()

    assert pool.discarded == [broken]
    assert rows == [("healthy",)]


@pytest.mark.parametrize(
    ("recovery_error", "secret"),
    [
        (
            PrivatePayloadBaseException("private-recovery-secret"),
            "private-recovery-secret",
        ),
        (
            SystemExit("system-exit-recovery-secret"),
            "system-exit-recovery-secret",
        ),
    ],
)
def test_recovery_failure_uses_safe_copy_without_private_sensitive_state(
    monkeypatch,
    recovery_error,
    secret,
) -> None:
    broken = RecordingConnection()
    healthy = RecordingConnection(rows=[("healthy",)])
    broken.get_transaction_status = lambda: (
        engine_module.psycopg2.extensions.TRANSACTION_STATUS_INTRANS
    )
    broken.rollback_failure = recovery_error
    pool = RecordingPool([broken, healthy])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(
        dsn=f"host=fake password={secret}",
        minconn=1,
        maxconn=1,
    )
    try:
        with pytest.raises(type(recovery_error)) as exc_info:
            with engine.connection():
                pass

        assert exc_info.value is not recovery_error
        assert_exception_graph_is_redacted(exc_info.value, secret)
        assert engine.fetch_all(CompiledSQL(sql.SQL("SELECT healthy"))) == [
            ("healthy",)
        ]
    finally:
        engine.close()


def test_recovery_failure_preserves_body_error_with_sanitized_cleanup_note(
    monkeypatch,
) -> None:
    broken = RecordingConnection()
    broken.get_transaction_status = lambda: (
        engine_module.psycopg2.extensions.TRANSACTION_STATUS_INTRANS
    )
    broken.rollback_failure = RuntimeError("cleanup failed for cleanup-secret")
    pool = RecordingPool([broken])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake password=cleanup-secret")
    body_error = ValueError("business failed")
    try:
        with pytest.raises(ValueError) as exc_info:
            with engine.connection():
                raise body_error
    finally:
        engine.close()

    assert exc_info.value is body_error
    notes = getattr(body_error, "__notes__", ())
    assert any("connection recovery" in note for note in notes)
    assert_exception_graph_is_redacted(body_error, "cleanup-secret")
    assert pool.discarded == [broken]


def test_recovery_failure_redacts_free_text_connection_secret_from_exception_graph(
    monkeypatch,
) -> None:
    secret = "free-text-recovery-secret"
    recovery_error = RuntimeError(f"recovery failed for {secret}")
    broken = RecordingConnection()
    healthy = RecordingConnection(rows=[("healthy",)])
    broken.get_transaction_status = lambda: (
        engine_module.psycopg2.extensions.TRANSACTION_STATUS_INTRANS
    )
    broken.rollback_failure = recovery_error
    pool = RecordingPool([broken, healthy])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(
        dsn=f"host=fake user=tester password={secret}",
        minconn=1,
        maxconn=1,
    )
    try:
        with pytest.raises(RuntimeError) as exc_info:
            with engine.connection():
                pass

        assert_exception_graph_is_redacted(exc_info.value, secret)
        assert engine.fetch_all(CompiledSQL(sql.SQL("SELECT healthy"))) == [
            ("healthy",)
        ]
    finally:
        engine.close()


@pytest.mark.parametrize("connection_state", ["return", "discard"])
@pytest.mark.parametrize(
    ("pool_error_factory", "expected_type"),
    [
        (
            lambda secret: RuntimeError(f"pool failed for {secret}"),
            GaussDBConnectionError,
        ),
        (
            lambda secret: KeyboardInterrupt(f"pool interrupted for {secret}"),
            KeyboardInterrupt,
        ),
    ],
)
def test_pool_failure_redacts_exception_graph_for_return_and_discard(
    monkeypatch,
    connection_state,
    pool_error_factory,
    expected_type,
) -> None:
    secret = f"free-text-{connection_state}-pool-secret"
    connection = RecordingConnection()
    if connection_state == "discard":
        connection.get_transaction_status = lambda: (
            engine_module.psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN
        )
    pool = RecordingPool([connection])
    pool.putconn_failure = pool_error_factory(secret)
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn=f"host=fake password={secret}")
    try:
        with pytest.raises(expected_type) as exc_info:
            with engine.connection():
                pass

        assert_exception_graph_is_redacted(exc_info.value, secret)
        pool.putconn_failure = None
        with pytest.raises(GaussDBConnectionError, match="unusable"):
            engine.check_connection()
    finally:
        pool.putconn_failure = None
        engine.close()


@pytest.mark.parametrize(
    ("connection_state", "error_type"),
    [
        ("return", ConstructibleKeyboardInterrupt),
        ("discard", ConstructibleSystemExit),
    ],
)
def test_pool_failure_preserves_constructible_base_exception_subclass(
    monkeypatch,
    connection_state,
    error_type,
) -> None:
    secret = f"{connection_state}-subclass-pool-secret"
    connection = RecordingConnection()
    if connection_state == "discard":
        connection.get_transaction_status = lambda: (
            engine_module.psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN
        )
    pool = RecordingPool([connection])
    original_error = error_type(f"pool failed for {secret}")
    pool.putconn_failure = original_error
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn=f"host=fake password={secret}")
    try:
        try:
            with engine.connection():
                pass
        except BaseException as exc:
            captured = exc
        else:
            raise AssertionError("pool failure was not propagated")

        assert type(captured) is error_type
        assert captured is not original_error
        assert_exception_graph_is_redacted(captured, secret)
        pool.putconn_failure = None
        with pytest.raises(GaussDBConnectionError, match="unusable"):
            engine.check_connection()
    finally:
        pool.putconn_failure = None
        engine.close()


def test_discard_failure_poison_engine(monkeypatch) -> None:
    broken = RecordingConnection()
    broken.get_transaction_status = lambda: (
        engine_module.psycopg2.extensions.TRANSACTION_STATUS_INTRANS
    )
    broken.rollback_failure = RuntimeError("password=recovery-secret")
    pool = RecordingPool([broken])
    pool.putconn_failure = RuntimeError("discard failed")
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")
    try:
        with pytest.raises(GaussDBConnectionError) as exc_info:
            with engine.connection():
                pass

        message = str(exc_info.value)
        assert "discard connection" in message
        assert "primary_failure=password=***" in message
        assert "recovery-secret" not in message
        pool.putconn_failure = None
        with pytest.raises(GaussDBConnectionError, match="unusable"):
            engine.check_connection()
    finally:
        pool.putconn_failure = None
        engine.close()


@pytest.mark.parametrize(
    ("failure_path", "discard_error", "secret"),
    [
        (
            "cursor",
            KeyboardInterrupt("cursor discard failed for cursor-discard-secret"),
            "cursor-discard-secret",
        ),
        (
            "rollback",
            SystemExit("rollback discard failed for rollback-discard-secret"),
            "rollback-discard-secret",
        ),
        (
            "commit",
            RuntimeError("commit discard failed for commit-discard-secret"),
            "commit-discard-secret",
        ),
    ],
)
def test_transaction_discard_failure_preserves_type_and_poisons_engine(
    monkeypatch,
    failure_path,
    discard_error,
    secret,
) -> None:
    broken = RecordingConnection()
    if failure_path == "cursor":
        broken.cursor_failure = FakePsycopgError("cursor failed")
    elif failure_path == "rollback":
        broken.fail_next("SELECT broken", FakePsycopgError("query failed"))
        broken.rollback_failure = FakePsycopgError("rollback failed")
    else:
        broken.commit_failure = FakePsycopgError("commit failed")
        broken.rollback_failure = FakePsycopgError("rollback failed")
    pool = RecordingPool([broken])
    pool.putconn_failure = discard_error
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn=f"host=fake password={secret}")
    try:
        with pytest.raises(type(discard_error)) as exc_info:
            engine.execute(CompiledSQL(sql.SQL("SELECT broken")))

        assert_exception_graph_is_redacted(exc_info.value, secret)
        pool.putconn_failure = None
        with pytest.raises(GaussDBConnectionError, match="unusable"):
            engine.check_connection()
    finally:
        pool.putconn_failure = None
        engine.close()


def test_closed_connection_is_returned_for_pool_discard_without_recovery(
    sync_engine,
) -> None:
    engine, _, connection = sync_engine
    connection.closed = True
    connection.get_transaction_status = lambda: (_ for _ in ()).throw(
        AssertionError("closed connection status must not be inspected")
    )
    connection.rollback_failure = AssertionError(
        "closed connection must not be rolled back"
    )

    engine._prepare_connection_for_return(connection)

    assert connection.rollbacks == 0


def test_check_connection_uses_parameterized_select(sync_engine) -> None:
    engine, _, connection = sync_engine

    assert engine.check_connection() is True

    assert ("SELECT %s", (1,)) in connection.executed
