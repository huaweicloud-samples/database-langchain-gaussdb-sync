import asyncio
import inspect
import threading

import pytest
from psycopg2 import sql

import langchain_gaussdb.engine as engine_module
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL
from tests.helpers.fakes import FakeConnection, FakeCursor, FakeThreadedPool


def test_engine_has_no_native_async_or_reactor_api() -> None:
    for name in (
        "_run_as_async",
        "_run_as_sync",
        "_transaction_async",
        "_transaction_on_reactor",
        "_execute_async",
        "_execute_on_reactor",
        "_fetch_all_async",
        "_fetch_all_on_reactor",
        "aexecute",
        "afetch_all",
        "atransaction",
        "aclose",
        "run_sync_db",
    ):
        assert not hasattr(GaussDBEngine, name)


def test_engine_source_has_no_async_driver_or_asyncio_dependency() -> None:
    source = inspect.getsource(engine_module)
    assert "_async_driver" not in source
    assert "asyncio" not in source
    assert "async def" not in source


@pytest.mark.asyncio
async def test_sync_engine_can_be_called_from_executor_without_blocking_loop(
    monkeypatch,
) -> None:
    query_started = threading.Event()
    release_query = threading.Event()

    class BlockingCursor(FakeCursor):
        def execute(self, statement, params=None):
            query_started.set()
            release_query.wait(2)
            super().execute(statement, params)

    pool = FakeThreadedPool([FakeConnection(BlockingCursor(rows=[("ok",)]))])
    monkeypatch.setattr(
        engine_module,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: pool,
        raising=False,
    )
    engine = GaussDBEngine(dsn="host=fake")
    heartbeat = asyncio.Event()

    async def beat() -> None:
        await asyncio.sleep(0.01)
        heartbeat.set()

    query = asyncio.create_task(
        asyncio.to_thread(
            engine.fetch_all,
            CompiledSQL(sql.SQL("SELECT %s"), [1]),
        )
    )
    beat_task = asyncio.create_task(beat())
    try:
        assert await asyncio.to_thread(query_started.wait, 1)
        await asyncio.wait_for(heartbeat.wait(), 1)
        release_query.set()
        assert await query == [("ok",)]
        await beat_task
    finally:
        release_query.set()
        await asyncio.gather(query, beat_task, return_exceptions=True)
        engine.close()
