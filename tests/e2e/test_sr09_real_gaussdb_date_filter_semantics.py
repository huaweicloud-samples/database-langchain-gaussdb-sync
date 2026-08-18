"""E2E: PGVectorStore v2-style date filters on real GaussDB."""

import datetime as dt
import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBFilterError, GaussDBVectorStore
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
    return "sr09date_" + uuid.uuid4().hex[:10]


def _store(engine, table):
    return GaussDBVectorStore(
        embedding=_E(),
        table_name=table,
        embedding_dimension=3,
        engine=engine,
    )


def _seed(store, *, include_invalid=False):
    store.setup()
    texts = ["good jul1", "good aug1", "no date"]
    metadatas = [
        {"pd": "2026-07-01"},
        {"pd": "2026-08-01"},
        {"other": "x"},  # field absent
    ]
    ids = ["jul1", "aug1", "absent"]
    if include_invalid:
        texts.append("bad feb31")
        metadatas.append({"pd": "2026-02-31"})
        ids.append("bad")
    store.add_texts(
        texts,
        metadatas=metadatas,
        ids=ids,
    )


def _ids(docs):
    return [d.id for d in docs]


def test_real_gaussdb_date_filter_good_values_match():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # String values keep text semantics; Python dates select real DATE
        # conversion on both sides of the predicate.
        assert _ids(
            store.similarity_search(
                "x",
                k=10,
                filter={"pd": {"$eq": "2026-07-01"}},
            )
        ) == ["jul1"]
        assert _ids(
            store.similarity_search(
                "x",
                k=10,
                filter={"pd": {"$eq": dt.date(2026, 7, 1)}},
            )
        ) == ["jul1"]

        # range: good ISO dates compare correctly (lexicographic == chronological for ISO)
        gte = store.similarity_search("x", k=10, filter={"pd": {"$gte": "2026-08-01"}})
        assert _ids(gte) == ["aug1"]

        between = store.similarity_search(
            "x", k=10, filter={"pd": {"$between": ["2026-07-01", "2026-07-31"]}}
        )
        assert _ids(between) == ["jul1"]

        date_between = store.similarity_search(
            "x",
            k=10,
            filter={
                "pd": {
                    "$between": [
                        dt.date(2026, 7, 1),
                        dt.date(2026, 7, 31),
                    ]
                }
            },
        )
        assert _ids(date_between) == ["jul1"]

        in_docs = store.similarity_search(
            "x",
            k=10,
            filter={"pd": {"$in": [dt.date(2026, 8, 1)]}},
        )
        assert _ids(in_docs) == ["aug1"]

        ne_docs = store.similarity_search(
            "x",
            k=10,
            filter={"pd": {"$ne": dt.date(2026, 7, 1)}},
        )
        assert _ids(ne_docs) == ["aug1"]
        completed = True
    finally:
        try:
            _enable_writes(engine)
            engine.execute(
                CompiledSQL(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        qualified_name(None, table)
                    )
                ),
                operation="drop table",
            )
        finally:
            try:
                engine.close()
            except Exception:
                if completed:
                    raise


def test_real_gaussdb_date_filter_rejects_invalid_stored_date():
    """A stored invalid date (2026-02-31) must surface as a clear filter error
    when used in a date comparison, not be silently matched/excluded."""
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store, include_invalid=True)
        with pytest.raises(GaussDBFilterError) as exc:
            store.similarity_search(
                "x", k=10, filter={"pd": {"$gte": dt.date(2026, 1, 1)}}
            )
        assert "pd" in str(exc.value) or "publish_date" in str(exc.value)
        completed = True
    finally:
        try:
            _enable_writes(engine)
            engine.execute(
                CompiledSQL(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        qualified_name(None, table)
                    )
                ),
                operation="drop table",
            )
        finally:
            try:
                engine.close()
            except Exception:
                if completed:
                    raise
