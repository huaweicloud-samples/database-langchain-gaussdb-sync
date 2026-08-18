from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest
from langchain_core.runnables.config import run_in_executor
from psycopg2 import sql

from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL

pytestmark = pytest.mark.gaussdb_e2e


def _dsn() -> str:
    value = os.getenv("GAUSSDB_TEST_DSN")
    if not value:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return value


def test_synchronous_engine_connects_and_selects_one() -> None:
    engine = GaussDBEngine(dsn=_dsn())
    try:
        rows = engine.fetch_all(CompiledSQL(sql.SQL("SELECT %s"), [1]))
    finally:
        engine.close()

    assert rows == [(1,)]


@pytest.mark.asyncio
async def test_executor_queries_use_synchronous_pool_concurrently() -> None:
    engine = GaussDBEngine(dsn=_dsn(), minconn=2, maxconn=2)
    try:
        started = time.perf_counter()
        first, second = await asyncio.gather(
            run_in_executor(
                None,
                engine.fetch_all,
                CompiledSQL(sql.SQL("SELECT pg_sleep(1), %s"), [1]),
            ),
            run_in_executor(
                None,
                engine.fetch_all,
                CompiledSQL(sql.SQL("SELECT pg_sleep(1), %s"), [2]),
            ),
        )
        elapsed = time.perf_counter() - started
    finally:
        engine.close()

    assert first[0][1] == 1
    assert second[0][1] == 2
    assert elapsed < 1.8


@pytest.mark.asyncio
async def test_executor_query_does_not_block_caller_event_loop() -> None:
    engine = GaussDBEngine(dsn=_dsn())
    query = asyncio.create_task(
        run_in_executor(
            None,
            engine.fetch_all,
            CompiledSQL(sql.SQL("SELECT pg_sleep(1)")),
        )
    )
    heartbeat = 0
    try:
        while not query.done():
            heartbeat += 1
            await asyncio.sleep(0.05)
        await query
    finally:
        engine.close()

    assert heartbeat >= 10


@pytest.mark.asyncio
async def test_executor_timeout_does_not_cancel_database_work() -> None:
    """Executor cancellation stops waiting, not the synchronous DB operation."""

    engine = GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)
    query = asyncio.create_task(
        run_in_executor(
            None,
            engine.fetch_all,
            CompiledSQL(sql.SQL("SELECT pg_sleep(1)")),
        )
    )
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(query), timeout=0.2)

        # The worker owns the only pool slot until the synchronous query ends.
        assert not query.done()
        await query
        rows = await run_in_executor(
            None,
            engine.fetch_all,
            CompiledSQL(sql.SQL("SELECT %s"), [1]),
        )
    finally:
        engine.close()

    assert rows == [(1,)]


@pytest.mark.asyncio
async def test_sync_and_executor_queries_share_pool_capacity() -> None:
    engine = GaussDBEngine(dsn=_dsn(), minconn=2, maxconn=2)
    sync_rows: list[object] = []
    sync_errors: list[BaseException] = []

    def run_sync_query() -> None:
        try:
            sync_rows.extend(
                engine.fetch_all(CompiledSQL(sql.SQL("SELECT pg_sleep(1), %s"), [1]))
            )
        except BaseException as exc:
            sync_errors.append(exc)

    worker = threading.Thread(target=run_sync_query)
    try:
        started = time.perf_counter()
        worker.start()
        executor_rows = await run_in_executor(
            None,
            engine.fetch_all,
            CompiledSQL(sql.SQL("SELECT pg_sleep(1), %s"), [2]),
        )
        worker.join(3)
        elapsed = time.perf_counter() - started
    finally:
        engine.close()

    assert not worker.is_alive()
    assert sync_errors == []
    assert sync_rows[0][1] == 1
    assert executor_rows[0][1] == 2
    assert elapsed < 1.8
