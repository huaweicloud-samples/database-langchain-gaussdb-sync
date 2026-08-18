import inspect
import threading
import time

import pytest

import langchain_gaussdb.engine as engine_module
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.errors import GaussDBConnectionError
from tests.helpers.fakes import FakeConnection, FakeThreadedPool


class RecordingPoolFactory:
    def __init__(self, pool=None):
        self.pool = pool or FakeThreadedPool()
        self.calls = []

    def __call__(self, minconn, maxconn, *args, **kwargs):
        self.calls.append((minconn, maxconn, args, kwargs))
        return self.pool


@pytest.fixture
def pool_factory(monkeypatch):
    factory = RecordingPoolFactory()
    monkeypatch.setattr(engine_module, "ThreadedConnectionPool", factory, raising=False)
    return factory


def test_engine_accepts_only_dsn_or_connection_kwargs() -> None:
    with pytest.raises(GaussDBConnectionError, match="exactly one"):
        GaussDBEngine()
    with pytest.raises(GaussDBConnectionError, match="exactly one"):
        GaussDBEngine(
            dsn="host=one",
            connection_kwargs={"host": "two"},
        )


def test_engine_public_constructor_has_no_pool_or_connection_injection() -> None:
    parameters = inspect.signature(GaussDBEngine).parameters

    assert tuple(parameters) == ("dsn", "connection_kwargs", "minconn", "maxconn")


@pytest.mark.parametrize(
    ("minconn", "maxconn"),
    [(0, 1), (2, 1), (1, 0)],
)
def test_engine_validates_connection_limits(minconn: int, maxconn: int) -> None:
    with pytest.raises(GaussDBConnectionError, match="minconn"):
        GaussDBEngine(dsn="host=fake", minconn=minconn, maxconn=maxconn)


def test_engine_rejects_user_async_flag() -> None:
    with pytest.raises(GaussDBConnectionError, match="async_"):
        GaussDBEngine(connection_kwargs={"host": "fake", "async_": 1})


def test_engine_builds_threaded_pool_from_dsn(pool_factory) -> None:
    engine = GaussDBEngine(dsn="host=fake", minconn=2, maxconn=4)
    try:
        assert pool_factory.calls == [(2, 4, ("host=fake",), {})]
    finally:
        engine.close()


def test_engine_builds_threaded_pool_from_connection_kwargs(pool_factory) -> None:
    engine = GaussDBEngine(
        connection_kwargs={"host": "fake", "user": "tester"},
        minconn=1,
        maxconn=3,
    )
    try:
        assert pool_factory.calls == [(1, 3, (), {"host": "fake", "user": "tester"})]
    finally:
        engine.close()


def test_engine_rejects_new_operations_after_close(pool_factory) -> None:
    engine = GaussDBEngine(dsn="host=fake")
    engine.close()

    with pytest.raises(GaussDBConnectionError, match="closed"):
        engine.check_connection()


def test_engine_rejects_new_operations_while_close_is_in_progress(
    pool_factory,
) -> None:
    engine = GaussDBEngine(dsn="host=fake")
    engine._enter_operation()
    close_thread = threading.Thread(target=engine.close)
    close_thread.start()
    try:
        for _ in range(100):
            with engine._lifecycle:
                if engine._closing:
                    break
            close_thread.join(0.01)
        else:
            raise AssertionError("engine did not enter closing state")

        with pytest.raises(GaussDBConnectionError, match="closing"):
            engine.check_connection()
    finally:
        engine._exit_operation()
        close_thread.join(2)

    assert not close_thread.is_alive()


def test_close_from_active_operation_fails_fast_without_entering_closing(
    monkeypatch,
) -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [(1,)]
    pool = FakeThreadedPool([connection])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        RecordingPoolFactory(pool),
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")

    def reject_wait(*args, **kwargs):
        raise AssertionError("close waited for its own active operation")

    monkeypatch.setattr(engine._lifecycle, "wait", reject_wait)
    try:
        with pytest.raises(GaussDBConnectionError, match="active operation"):
            engine.transaction(lambda cursor: engine.close())

        assert engine._closing is False
        assert connection.rollbacks == 1
        assert engine.check_connection() is True
    finally:
        # Keep the RED implementation from leaving teardown in a closing state.
        with engine._lifecycle:
            engine._closing = False
            engine._close_attempt = None
        engine.close()


def test_semaphore_applies_backpressure_before_pool_getconn(monkeypatch) -> None:
    first_connection = FakeConnection()
    second_connection = FakeConnection()
    pool = FakeThreadedPool([first_connection, second_connection])
    factory = RecordingPoolFactory(pool)
    monkeypatch.setattr(engine_module, "ThreadedConnectionPool", factory, raising=False)
    engine = GaussDBEngine(dsn="host=fake", minconn=1, maxconn=1)
    first_borrowed = threading.Event()
    release_first = threading.Event()
    second_borrowed = threading.Event()

    def first_operation() -> None:
        with engine.connection():
            first_borrowed.set()
            release_first.wait(2)

    def second_operation() -> None:
        with engine.connection():
            second_borrowed.set()

    first = threading.Thread(target=first_operation)
    second = threading.Thread(target=second_operation)
    first.start()
    assert first_borrowed.wait(1)
    second.start()
    time.sleep(0.05)
    assert second_borrowed.is_set() is False
    assert len(pool.borrowed) == 1

    release_first.set()
    first.join(2)
    second.join(2)
    engine.close()

    assert second_borrowed.is_set() is True
    assert len(pool.borrowed) == 2


def test_concurrent_close_is_idempotent(pool_factory) -> None:
    engine = GaussDBEngine(dsn="host=fake")
    errors = []

    def close_engine() -> None:
        try:
            engine.close()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=close_engine)
    second = threading.Thread(target=close_engine)
    first.start()
    second.start()
    first.join(2)
    second.join(2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert pool_factory.pool.closed is True


def test_close_failure_can_be_retried(monkeypatch) -> None:
    class RetryableClosePool(FakeThreadedPool):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        def closeall(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("close failed")
            super().closeall()

    pool = RetryableClosePool()
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        RecordingPoolFactory(pool),
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")

    with pytest.raises(GaussDBConnectionError, match="close"):
        engine.close()

    engine.close()
    assert pool.closed is True
    assert pool.close_calls == 2


def test_pool_creation_error_is_sanitized_and_preserves_sqlstate(monkeypatch) -> None:
    class ConnectFailure(Exception):
        pgcode = "08001"

    def fail(*args, **kwargs):
        raise ConnectFailure("password=topsecret host=fake")

    monkeypatch.setattr(engine_module, "ThreadedConnectionPool", fail, raising=False)

    with pytest.raises(GaussDBConnectionError) as exc_info:
        GaussDBEngine(dsn="host=fake password=topsecret")

    assert "topsecret" not in str(exc_info.value)
    assert exc_info.value.sqlstate == "08001"
