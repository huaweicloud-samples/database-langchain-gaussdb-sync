from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from psycopg2 import sql

from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL, qualified_name
from langchain_gaussdb.vectorstore import GaussDBVectorStore

pytestmark = [
    pytest.mark.gaussdb_e2e,
    pytest.mark.skipif(
        not os.getenv("GAUSSDB_TEST_DSN"),
        reason="GAUSSDB_TEST_DSN is not set",
    ),
]


def _dsn() -> str:
    value = os.getenv("GAUSSDB_TEST_DSN")
    if not value:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return value


def _table_name() -> str:
    return "sr08_standard_" + uuid.uuid4().hex[:12]


def _enable_session_writes(engine: GaussDBEngine) -> None:
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable sr08 standard e2e writes",
    )


def _drop_table(
    engine: GaussDBEngine,
    table: str,
    *,
    strict: bool,
) -> None:
    try:
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
            ),
            operation="drop sr08 standard e2e table",
        )
    except Exception:
        if strict:
            raise


def _close_engine(engine: GaussDBEngine, *, strict: bool) -> None:
    try:
        engine.close()
    except Exception:
        if strict:
            raise


class TestGaussDBStandardVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Iterator[VectorStore]:
        engine = GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)
        table = _table_name()
        completed = False
        try:
            _enable_session_writes(engine)
            store = GaussDBVectorStore(
                embedding=self.get_embeddings(),
                engine=engine,
                table_name=table,
                embedding_dimension=6,
            )
            store.setup()
            completed = True
            yield store
        finally:
            try:
                _drop_table(engine, table, strict=completed)
            finally:
                _close_engine(engine, strict=completed)
