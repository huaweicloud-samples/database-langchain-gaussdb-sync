"""E2E: JSONB NULL and key-presence filters in lexical retrieval modes.

sr06 only exercises a simple equality filter in lexical modes. This verifies
that scalar comparisons exclude missing keys and JSON null, while ``$exists``
uses JSONB key presence and an empty string remains matchable, through the
BM25 and hybrid retrieval paths.
"""

import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBVectorStore
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL, qualified_name

pytestmark = pytest.mark.gaussdb_e2e


class _E(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def _dsn():
    v = os.getenv("GAUSSDB_TEST_DSN")
    if not v:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return v


def _engine():
    return GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)


def _enable_writes(engine):
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable writes",
    )


def _table():
    return "sr10bm_" + uuid.uuid4().hex[:10]


def _store(engine, table, mode):
    return GaussDBVectorStore(
        embedding=_E(),
        table_name=table,
        embedding_dimension=3,
        engine=engine,
        retrieval_mode=mode,
    )


def _seed(store):
    store.setup()
    # All docs contain the query token so BM25 returns all of them and the
    # metadata filter is the only discriminator (comparable to dense mode).
    store.add_texts(
        ["contract alpha", "contract beta", "contract gamma", "contract delta"],
        metadatas=[
            {"status": ""},  # empty string
            {"status": None},  # json null
            {},  # absent
            {"status": "active"},  # real value
        ],
        ids=["empty", "jnull", "absent", "active"],
    )


def _require_bm25(engine):
    distributed = engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pgxc_node "
                "WHERE node_type = 'D')"
            )
        ),
        operation="probe sr10 GaussDB deployment topology",
    ) == [(True,)]
    if distributed:
        pytest.skip(
            "BM25 and hybrid retrieval are not supported on distributed GaussDB"
        )
    if not engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 1 FROM pg_am WHERE lower(amname) = 'bm25'")),
        operation="probe sr10 bm25 access method",
    ):
        pytest.skip("GaussDB has no bm25 access method")


def _drop(engine, table):
    try:
        _enable_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
            ),
            operation="drop table",
        )
    except Exception:
        pass


def _ids(docs):
    return sorted(d.id for d in docs)


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_real_gaussdb_lexical_mode_eq_in_exists_filter(mode):
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        _require_bm25(engine)
        store = _store(engine, table, mode)
        _seed(store)

        q = "contract"
        assert _ids(
            store.similarity_search(q, k=10, filter={"status": {"$eq": ""}})
        ) == ["empty"]
        assert _ids(
            store.similarity_search(q, k=10, filter={"status": {"$in": ["", "active"]}})
        ) == ["active", "empty"]
        assert _ids(
            store.similarity_search(
                q,
                k=10,
                filter={"status": {"$exists": True}},
            )
        ) == ["active", "empty", "jnull"]
        assert _ids(
            store.similarity_search(
                q,
                k=10,
                filter={"status": {"$exists": False}},
            )
        ) == ["absent"]
        completed = True
    finally:
        _drop(engine, table)
        try:
            engine.close()
        except Exception:
            if completed:
                raise


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_real_gaussdb_lexical_mode_ne_nin_filter(mode):
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        _require_bm25(engine)
        store = _store(engine, table, mode)
        _seed(store)

        q = "contract"
        # SQL three-valued logic excludes missing and JSON-null rows.
        assert _ids(
            store.similarity_search(
                q,
                k=10,
                filter={"status": {"$ne": ""}},
            )
        ) == ["active"]
        assert _ids(
            store.similarity_search(
                q,
                k=10,
                filter={"status": {"$nin": [""]}},
            )
        ) == ["active"]
        completed = True
    finally:
        _drop(engine, table)
        try:
            engine.close()
        except Exception:
            if completed:
                raise
