import asyncio
import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBVectorStore
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL, qualified_name

pytestmark = pytest.mark.gaussdb_e2e


class FixedRetrievalEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "same": [1.0, 0.0, 0.0],
            "orthogonal": [0.0, 1.0, 0.0],
            "opposite": [-1.0, 0.0, 0.0],
        }
        return [vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _dsn():
    value = os.getenv("GAUSSDB_TEST_DSN")
    if not value:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return value


def _table_name():
    return "sr03_" + uuid.uuid4().hex[:12]


def _engine():
    return GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)


def _enable_session_writes(engine):
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable sr03 e2e writes",
    )


def _drop_table(engine, table, *, strict):
    try:
        _enable_session_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
            ),
            operation="drop sr03 e2e table",
        )
    except Exception:
        if strict:
            raise


def _close_engine(engine, *, strict):
    try:
        engine.close()
    except Exception:
        if strict:
            raise


def _store(engine, table, *, distance_strategy="cosine"):
    return GaussDBVectorStore(
        embedding=FixedRetrievalEmbeddings(),
        table_name=table,
        embedding_dimension=3,
        engine=engine,
        distance_strategy=distance_strategy,
    )


def _seed_store(store):
    store.setup()
    store.add_texts(
        ["same", "orthogonal", "opposite"],
        metadatas=[
            {"kind": "same"},
            {"kind": "orthogonal"},
            {"kind": "opposite"},
        ],
        ids=["doc-same", "doc-orthogonal", "doc-opposite"],
    )


def test_real_gaussdb_dense_similarity_score_relevance_and_retriever():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _store(engine, table)
        _seed_store(store)

        documents = store.similarity_search("needle", k=3)
        assert [document.id for document in documents] == [
            "doc-same",
            "doc-orthogonal",
            "doc-opposite",
        ]

        scored = store.similarity_search_with_score("needle", k=3)
        assert [document.id for document, _score in scored] == [
            "doc-same",
            "doc-orthogonal",
            "doc-opposite",
        ]
        assert [round(score, 6) for _document, score in scored] == [0.0, 1.0, 2.0]

        relevant = store.similarity_search_with_relevance_scores("needle", k=3)
        assert [document.id for document, _score in relevant] == [
            "doc-same",
            "doc-orthogonal",
            "doc-opposite",
        ]
        assert [round(score, 6) for _document, score in relevant] == [1.0, 0.5, 0.0]

        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        retrieved = retriever.invoke("needle")
        assert [document.id for document in retrieved] == ["doc-same"]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_l2_distance_strategy():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _store(engine, table, distance_strategy="l2")
        _seed_store(store)

        scored = store.similarity_search_with_score("needle", k=3)

        assert [document.id for document, _score in scored] == [
            "doc-same",
            "doc-orthogonal",
            "doc-opposite",
        ]
        assert [round(score, 6) for _document, score in scored] == [
            0.0,
            round(2**0.5, 6),
            2.0,
        ]

        relevant = store.similarity_search_with_relevance_scores("needle", k=1)
        assert [(document.id, round(score, 6)) for document, score in relevant] == [
            ("doc-same", 1.0)
        ]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_mmr_and_async_similarity():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _store(engine, table)
        _seed_store(store)

        mmr_docs = store.max_marginal_relevance_search(
            "needle",
            k=2,
            fetch_k=3,
            lambda_mult=1.0,
        )
        assert [document.id for document in mmr_docs] == [
            "doc-same",
            "doc-orthogonal",
        ]

        async_docs = asyncio.run(store.asimilarity_search("needle", k=1))
        assert [document.id for document in async_docs] == ["doc-same"]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)
