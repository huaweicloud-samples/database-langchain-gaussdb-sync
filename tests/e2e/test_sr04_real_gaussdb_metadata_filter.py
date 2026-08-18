import asyncio
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


class FilterEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "same-a": [1.0, 0.0, 0.0],
            "same-b": [0.9, 0.1, 0.0],
            "tenant-c": [0.0, 1.0, 0.0],
            "bad-score": [0.8, 0.2, 0.0],
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
    return "sr04_" + uuid.uuid4().hex[:12]


def _engine():
    return GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)


def _enable_session_writes(engine):
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable sr04 e2e writes",
    )
    engine.execute(
        CompiledSQL(sql.SQL("SET maintenance_work_mem = '128MB'")),
        operation="set sr04 index maintenance memory",
    )


def _drop_table(engine, table, *, strict):
    try:
        _enable_session_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
            ),
            operation="drop sr04 e2e table",
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


def _store(engine, table):
    return GaussDBVectorStore(
        embedding=FilterEmbeddings(),
        table_name=table,
        embedding_dimension=3,
        engine=engine,
    )


def _indexed_store(engine, table):
    return GaussDBVectorStore(
        embedding=FilterEmbeddings(),
        table_name=table,
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={
            "event_date": "date",
            "status": "text",
        },
    )


def _seed_store(store):
    store.setup()
    store.add_texts(
        ["same-a", "same-b", "tenant-c"],
        metadatas=[
            {
                "tenant": "t1",
                "source": "manual",
                "source-name": "manual-v1",
                "source.name": "manual.dot",
                "rank": 10,
                "score": 0.95,
                "title": "GaussDB Guide",
                "tags": ["diskann", "vector"],
                "status": "active",
                "optional": "value",
                "empty": "",
                "nullable_text": "",
                "shape": 5,
                "bool_shape": True,
                "special": '第一行\n"引号"\\路径',
                "scientific": 1e-7,
                "event_date": "2026-07-01",
            },
            {
                "tenant": "t1",
                "source": "faq",
                "source-name": "faq-v1",
                "source.name": "faq.dot",
                "rank": 3,
                "score": 0.7,
                "title": "FAQ",
                "tags": ["bm25"],
                "status": "draft",
                "optional": None,
                "nullable_text": None,
                "shape": "5",
                "bool_shape": "true",
                "special": "other",
                "scientific": 1e20,
                "event_date": "2026-07-10",
            },
            {
                "tenant": "t2",
                "source": "manual",
                "source-name": "manual-v2",
                "source.name": "manual.other",
                "rank": 8,
                "score": 0.4,
                "title": "Other",
                "tags": ["diskann"],
                "status": "active",
                "nullable_text": "filled",
                "scientific": -0.0,
                "event_date": "2026-08-01",
            },
        ],
        ids=["doc-a", "doc-b", "doc-c"],
    )


def test_real_gaussdb_jsonb_temporal_metadata_membership_filters():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _indexed_store(engine, table)
        store.setup()
        store.add_texts(
            ["same-a", "same-b"],
            metadatas=[
                {
                    "created_at": "2026-07-01T12:30:45",
                    "start_time": "09:30:00",
                    "event_date": "2026-07-01",
                    "status": "",
                },
                {
                    "created_at": dt.datetime(2026, 7, 2, 8, 15, 0),
                    "start_time": dt.time(10, 45, 0),
                    "event_date": dt.date(2026, 7, 2),
                    "status": "active",
                },
            ],
            ids=["doc-a", "doc-b"],
        )

        timestamp_docs = store.similarity_search(
            "needle",
            k=2,
            filter={
                "created_at": {
                    "$in": [
                        "2026-07-01T12:30:45",
                        "2026-07-03T08:15:00",
                    ]
                }
            },
        )
        assert [document.id for document in timestamp_docs] == ["doc-a"]

        time_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"start_time": {"$nin": ["09:30:00"]}},
        )
        assert [document.id for document in time_docs] == ["doc-b"]

        date_in_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"event_date": {"$in": ["2026-07-01"]}},
        )
        assert [document.id for document in date_in_docs] == ["doc-a"]

        date_nin_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"event_date": {"$nin": [dt.date(2026, 7, 1)]}},
        )
        assert [document.id for document in date_nin_docs] == ["doc-b"]

        empty_text_ne_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$ne": "active"}},
        )
        assert [document.id for document in empty_text_ne_docs] == ["doc-a"]

        empty_text_nin_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$nin": ["active"]}},
        )
        assert [document.id for document in empty_text_nin_docs] == ["doc-a"]

        empty_text_eq_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$eq": ""}},
        )
        assert [document.id for document in empty_text_eq_docs] == ["doc-a"]

        empty_text_in_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$in": [""]}},
        )
        assert [document.id for document in empty_text_in_docs] == ["doc-a"]

        non_empty_text_ne_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$ne": ""}},
        )
        assert [document.id for document in non_empty_text_ne_docs] == ["doc-b"]

        non_empty_text_nin_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$nin": [""]}},
        )
        assert [document.id for document in non_empty_text_nin_docs] == ["doc-b"]

        empty_text_like_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$like": ""}},
        )
        assert [document.id for document in empty_text_like_docs] == ["doc-a"]

        empty_text_range_docs = store.similarity_search(
            "needle",
            k=2,
            filter={"status": {"$gte": ""}},
        )
        assert [document.id for document in empty_text_range_docs] == [
            "doc-a",
            "doc-b",
        ]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_metadata_filter_similarity_score_and_retriever():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _store(engine, table)
        _seed_store(store)

        tenant_docs = store.similarity_search("needle", k=3, filter={"tenant": "t1"})
        assert [document.id for document in tenant_docs] == ["doc-a", "doc-b"]

        high_score = store.similarity_search_with_score(
            "needle",
            k=3,
            filter={
                "$and": [
                    {"tenant": {"$in": ["t1", "t3"]}},
                    {"score": {"$gte": 0.8}},
                ]
            },
        )
        assert [document.id for document, _score in high_score] == ["doc-a"]

        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2, "filter": {"source": "faq"}},
        )
        retrieved = retriever.invoke("needle")
        assert [document.id for document in retrieved] == ["doc-b"]

        hyphen_key_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"source-name": {"$in": ["manual-v1", "missing"]}},
        )
        assert [document.id for document in hyphen_key_docs] == ["doc-a"]

        dotted_key_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"source.name": "faq.dot"},
        )
        assert [document.id for document in dotted_key_docs] == ["doc-b"]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_metadata_filter_logic_contains_mmr_and_async():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _store(engine, table)
        _seed_store(store)

        contains_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"tags": {"$contains": ["diskann"]}},
        )
        assert [document.id for document in contains_docs] == ["doc-a", "doc-c"]

        logic_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"$or": [{"source": "faq"}, {"$not": {"tenant": "t2"}}]},
        )
        assert [document.id for document in logic_docs] == ["doc-a", "doc-b"]

        mmr_docs = store.max_marginal_relevance_search(
            "needle",
            k=2,
            fetch_k=3,
            filter={"tenant": "t1"},
        )
        assert [document.id for document in mmr_docs] == ["doc-a", "doc-b"]

        async_docs = asyncio.run(
            store.asimilarity_search(
                "needle",
                k=2,
                filter={"$and": [{"tenant": "t2"}, {"rank": {"$gte": 8}}]},
            )
        )
        assert [document.id for document in async_docs] == ["doc-c"]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_metadata_filter_missing_null_and_empty_string_semantics():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _store(engine, table)
        _seed_store(store)

        equal_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"optional": {"$eq": "value"}},
        )
        assert [document.id for document in equal_docs] == ["doc-a"]

        in_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"optional": {"$in": ["value"]}},
        )
        assert [document.id for document in in_docs] == ["doc-a"]

        not_equal_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"optional": {"$ne": "value"}},
        )
        assert not_equal_docs == []

        not_in_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"optional": {"$nin": ["value"]}},
        )
        assert not_in_docs == []

        exists_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"optional": {"$exists": True}},
        )
        assert sorted(document.id for document in exists_docs) == ["doc-a", "doc-b"]

        null_or_missing_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"optional": {"$exists": False}},
        )
        assert [document.id for document in null_or_missing_docs] == ["doc-c"]

        empty_string_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"empty": ""},
        )
        assert [document.id for document in empty_string_docs] == ["doc-a"]

        empty_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"empty": {"$ne": "non-empty"}},
        )
        assert [document.id for document in empty_ne_docs] == ["doc-a"]

        empty_nin_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"empty": {"$nin": ["non-empty"]}},
        )
        assert [document.id for document in empty_nin_docs] == ["doc-a"]

        nullable_empty_eq_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"nullable_text": {"$eq": ""}},
        )
        assert [document.id for document in nullable_empty_eq_docs] == ["doc-a"]

        nullable_empty_in_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"nullable_text": {"$in": [""]}},
        )
        assert [document.id for document in nullable_empty_in_docs] == ["doc-a"]

        nullable_empty_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"nullable_text": {"$ne": ""}},
        )
        assert [document.id for document in nullable_empty_ne_docs] == ["doc-c"]

        nullable_empty_nin_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"nullable_text": {"$nin": [""]}},
        )
        assert [document.id for document in nullable_empty_nin_docs] == ["doc-c"]

        empty_like_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"empty": {"$like": ""}},
        )
        assert [document.id for document in empty_like_docs] == ["doc-a"]

        empty_ilike_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"empty": {"$ilike": ""}},
        )
        assert [document.id for document in empty_ilike_docs] == ["doc-a"]

        empty_range_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"empty": {"$between": ["", ""]}},
        )
        assert [document.id for document in empty_range_docs] == ["doc-a"]

        numeric_shape_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$eq": 5}},
        )
        assert [document.id for document in numeric_shape_docs] == ["doc-a"]

        equivalent_float_shape_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$eq": 5.0}},
        )
        assert [document.id for document in equivalent_float_shape_docs] == ["doc-a"]

        equivalent_float_shape_in_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$in": [5.0]}},
        )
        assert [document.id for document in equivalent_float_shape_in_docs] == ["doc-a"]

        text_shape_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$in": ["5"]}},
        )
        assert [document.id for document in text_shape_docs] == ["doc-b"]

        numeric_shape_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$ne": 5}},
        )
        assert [document.id for document in numeric_shape_ne_docs] == ["doc-b"]

        text_shape_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$ne": "5"}},
        )
        assert text_shape_ne_docs == []

        bool_shape_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"bool_shape": {"$ne": True}},
        )
        assert [document.id for document in bool_shape_ne_docs] == ["doc-b"]

        text_bool_shape_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"bool_shape": {"$ne": "true"}},
        )
        assert text_bool_shape_ne_docs == []

        numeric_shape_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$ne": 5}},
        )
        assert [document.id for document in numeric_shape_ne_docs] == ["doc-b"]

        numeric_shape_nin_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"shape": {"$nin": [5]}},
        )
        assert [document.id for document in numeric_shape_nin_docs] == ["doc-b"]

        scientific_eq_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"scientific": {"$eq": 1e-7}},
        )
        assert [document.id for document in scientific_eq_docs] == ["doc-a"]

        scientific_in_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"scientific": {"$in": [1e20]}},
        )
        assert [document.id for document in scientific_in_docs] == ["doc-b"]

        scientific_ne_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"scientific": {"$ne": -0.0}},
        )
        assert [document.id for document in scientific_ne_docs] == ["doc-a", "doc-b"]

        scientific_nin_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"scientific": {"$nin": [1e-7, 1e20]}},
        )
        assert [document.id for document in scientific_nin_docs] == ["doc-c"]

        date_range_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"event_date": {"$gte": dt.date(2026, 7, 10)}},
        )
        assert [document.id for document in date_range_docs] == ["doc-b", "doc-c"]

        special_eq_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"special": {"$eq": '第一行\n"引号"\\路径'}},
        )
        assert [document.id for document in special_eq_docs] == ["doc-a"]

        special_in_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"special": {"$in": ["missing", '第一行\n"引号"\\路径']}},
        )
        assert [document.id for document in special_in_docs] == ["doc-a"]

        json_null_docs = store.similarity_search(
            "needle",
            k=3,
            filter={"optional": {"$contains": None}},
        )
        assert [document.id for document in json_null_docs] == ["doc-b"]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_metadata_filter_cast_failure_is_clear():
    engine = _engine()
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(
            ["bad-score"],
            metadatas=[{"score": "not-a-number"}],
            ids=["doc-bad"],
        )

        with pytest.raises(GaussDBFilterError) as exc_info:
            store.similarity_search(
                "needle",
                k=1,
                filter={"score": {"$gte": 0.5}},
            )

        message = str(exc_info.value)
        assert "metadata filter" in message
        assert "score" in message
        assert "not-a-number" not in message
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)
