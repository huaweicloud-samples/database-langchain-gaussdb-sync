from __future__ import annotations

import os
import uuid

import pytest
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.sql import CompiledSQL, qualified_name
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings

pytestmark = pytest.mark.gaussdb_e2e


def _dsn() -> str:
    value = os.getenv("GAUSSDB_TEST_DSN")
    if not value:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return value


@pytest.mark.asyncio
async def test_executor_async_vectorstore_crud_filter_dense_and_mmr() -> None:
    table_name = "sr13_async_" + uuid.uuid4().hex[:10]
    engine = GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable executor async e2e writes",
    )
    engine.execute(
        CompiledSQL(sql.SQL("SET maintenance_work_mem = '128MB'")),
        operation="set sr13 vector index memory",
    )
    store = GaussDBVectorStore(
        engine=engine,
        embedding=DeterministicEmbeddings(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            query_vector=[1.0, 0.0, 0.0],
        ),
        embedding_dimension=3,
        table_name=table_name,
    )

    try:
        ids = await store.aadd_texts(
            ["async alpha", "async beta"],
            metadatas=[{"tenant": "a"}, {"tenant": "b"}],
            ids=["async-alpha", "async-beta"],
        )
        by_ids = await store.aget_by_ids(["async-alpha", "async-beta"])
        filtered = await store.asimilarity_search(
            "alpha",
            k=2,
            filter={"tenant": {"$eq": "a"}},
        )
        mmr = await store.amax_marginal_relevance_search(
            "alpha",
            k=1,
            fetch_k=2,
        )
        deleted = await store.adelete(ids=["async-alpha"])
        remaining = await store.aget_by_ids(["async-alpha"])

        assert ids == ["async-alpha", "async-beta"]
        assert {document.id for document in by_ids} == {
            "async-alpha",
            "async-beta",
        }
        assert [document.id for document in filtered] == ["async-alpha"]
        assert [document.id for document in mmr] == ["async-alpha"]
        assert deleted is True
        assert remaining == []
    finally:
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(
                    qualified_name(None, table_name)
                )
            ),
            operation="drop sr13 e2e table",
        )
        engine.close()
