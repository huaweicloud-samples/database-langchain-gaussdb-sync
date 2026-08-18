from __future__ import annotations

import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.errors import GaussDBSQLError
from langchain_gaussdb.sql import CompiledSQL

pytestmark = pytest.mark.gaussdb_e2e


class _Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _engine() -> GaussDBEngine:
    dsn = os.getenv("GAUSSDB_TEST_DSN")
    if not dsn:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return GaussDBEngine(dsn=dsn, minconn=1, maxconn=1)


def _read_only_setting(engine: GaussDBEngine) -> str:
    rows = engine.fetch_all(
        CompiledSQL(sql.SQL("SHOW default_transaction_read_only")),
        operation="read transaction read-only policy",
    )
    assert len(rows) == 1
    return str(rows[0][0]).lower()


def test_engine_preserves_server_read_only_policy() -> None:
    engine = _engine()
    try:
        before = _read_only_setting(engine)
        engine.fetch_all(CompiledSQL(sql.SQL("SELECT 1")), operation="read only probe")
        assert _read_only_setting(engine) == before
    finally:
        engine.close()


def test_default_read_only_server_rejects_missing_automatic_objects() -> None:
    engine = _engine()
    try:
        if _read_only_setting(engine) not in {"on", "true", "t"}:
            pytest.skip("target server is not read-only by default")
        store = GaussDBVectorStore(
            engine=engine,
            embedding=_Embeddings(),
            embedding_dimension=3,
            table_name=f"sr11ro_{uuid.uuid4().hex[:10]}",
        )

        with pytest.raises(GaussDBSQLError):
            store.similarity_search("missing table", k=1)
    finally:
        engine.close()
