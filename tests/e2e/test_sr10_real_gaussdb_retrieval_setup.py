from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.errors import GaussDBCapabilityError
from langchain_gaussdb.sql import CompiledSQL, qualified_name
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings

pytestmark = pytest.mark.gaussdb_e2e


def _dsn() -> str:
    value = os.getenv("GAUSSDB_TEST_DSN")
    if not value:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return value


def _table_name() -> str:
    return "sr10_setup_" + uuid.uuid4().hex[:10]


def _engine() -> GaussDBEngine:
    return GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)


def _prepare_test_session(engine: GaussDBEngine) -> None:
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable sr10 e2e writes",
    )
    engine.execute(
        CompiledSQL(sql.SQL("SET maintenance_work_mem = '128MB'")),
        operation="set sr10 vector index memory",
    )


def _store(
    engine: GaussDBEngine,
    table_name: str,
    *,
    retrieval_mode: str,
    embedding_dimension: int = 3,
) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        engine=engine,
        embedding=DeterministicEmbeddings(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            query_vector=[1.0, 0.0, 0.0],
        ),
        embedding_dimension=embedding_dimension,
        table_name=table_name,
        retrieval_mode=retrieval_mode,
    )


def _index_access_methods(
    engine: GaussDBEngine,
    table_name: str,
) -> set[str]:
    rows = engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT am.amname "
                "FROM pg_index AS idx "
                "JOIN pg_class AS tbl ON tbl.oid = idx.indrelid "
                "JOIN pg_class AS index_rel ON index_rel.oid = idx.indexrelid "
                "JOIN pg_am AS am ON am.oid = index_rel.relam "
                "WHERE tbl.relname = %s AND pg_table_is_visible(tbl.oid)"
            ),
            (table_name,),
        ),
        operation="inspect sr10 e2e indexes",
    )
    return {str(row[0]).lower() for row in rows}


def _drop_table(engine: GaussDBEngine, table_name: str) -> None:
    engine.execute(
        CompiledSQL(
            sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table_name))
        ),
        operation="drop sr10 e2e table",
    )


def _is_distributed(engine: GaussDBEngine) -> bool:
    return engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pgxc_node "
                "WHERE node_type = 'D')"
            )
        ),
        operation="probe sr10 GaussDB deployment topology",
    ) == [(True,)]


def _table_exists(engine: GaussDBEngine, table_name: str) -> bool:
    return engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_class AS rel "
                "JOIN pg_namespace AS ns ON ns.oid = rel.relnamespace "
                "WHERE ns.nspname = 'public' AND rel.relname = %s"
                ")"
            ),
            (table_name,),
        ),
        operation="check rejected sr10 table was not created",
    ) == [(True,)]


@pytest.mark.gaussdb_e2e_full
def test_real_gaussdb_dense_setup_is_idempotent_and_searchable() -> None:
    engine = _engine()
    table_name = _table_name()
    try:
        _prepare_test_session(engine)
        store = _store(engine, table_name, retrieval_mode="dense")

        store.setup()
        store.setup()
        store.add_texts(
            ["dense alpha", "dense beta"],
            ids=["dense-alpha", "dense-beta"],
        )

        assert store.similarity_search("dense alpha", k=1)[0].id == "dense-alpha"
        methods = _index_access_methods(engine, table_name)
        assert "gsdiskann" in methods
        assert "bm25" not in methods
    finally:
        try:
            _drop_table(engine, table_name)
        finally:
            engine.close()


@pytest.mark.gaussdb_e2e_full
def test_real_gaussdb_bm25_setup_is_ready_before_first_query() -> None:
    engine = _engine()
    table_name = _table_name()
    try:
        _prepare_test_session(engine)
        if _is_distributed(engine):
            pytest.skip("BM25 retrieval is not supported on distributed GaussDB")
        store = _store(engine, table_name, retrieval_mode="bm25")

        store.setup()
        store.setup()
        store.add_texts(
            ["contract alpha", "vector beta"],
            ids=["bm25-alpha", "bm25-beta"],
        )

        assert store.similarity_search("contract", k=1)[0].id == "bm25-alpha"
        methods = _index_access_methods(engine, table_name)
        assert "bm25" in methods
        assert "gsdiskann" not in methods
    finally:
        try:
            _drop_table(engine, table_name)
        finally:
            engine.close()


@pytest.mark.gaussdb_e2e_full
def test_real_gaussdb_hybrid_async_first_write_prepares_both_indexes() -> None:
    engine = _engine()
    table_name = _table_name()
    try:
        _prepare_test_session(engine)
        if _is_distributed(engine):
            pytest.skip("Hybrid retrieval is not supported on distributed GaussDB")
        store = _store(engine, table_name, retrieval_mode="hybrid")

        async def run() -> list:
            await store.aadd_texts(
                ["hybrid contract alpha", "unrelated beta"],
                ids=["hybrid-alpha", "hybrid-beta"],
            )
            return await store.asimilarity_search("contract alpha", k=1)

        assert asyncio.run(run())[0].id == "hybrid-alpha"
        methods = _index_access_methods(engine, table_name)
        assert {"gsdiskann", "bm25"} <= methods
    finally:
        try:
            _drop_table(engine, table_name)
        finally:
            engine.close()


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
@pytest.mark.gaussdb_e2e_full
def test_distributed_gaussdb_rejects_lexical_mode_before_ddl(mode: str) -> None:
    engine = _engine()
    table_name = _table_name()
    try:
        _prepare_test_session(engine)
        if not _is_distributed(engine):
            pytest.skip("distributed GaussDB is required")
        store = _store(engine, table_name, retrieval_mode=mode)

        with pytest.raises(GaussDBCapabilityError, match="distributed.*dense"):
            store.setup()

        assert not _table_exists(engine, table_name)
    finally:
        try:
            _drop_table(engine, table_name)
        finally:
            engine.close()


@pytest.mark.gaussdb_e2e_full
def test_distributed_gaussdb_rejects_dimension_above_1024_before_ddl() -> None:
    engine = _engine()
    table_name = _table_name()
    try:
        _prepare_test_session(engine)
        if not _is_distributed(engine):
            pytest.skip("distributed GaussDB is required")
        store = _store(
            engine,
            table_name,
            retrieval_mode="dense",
            embedding_dimension=1025,
        )

        with pytest.raises(GaussDBCapabilityError, match="up to 1024"):
            store.setup()

        assert not _table_exists(engine, table_name)
    finally:
        try:
            _drop_table(engine, table_name)
        finally:
            engine.close()
