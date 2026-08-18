from __future__ import annotations

import pytest

from langchain_gaussdb import GaussDBFilterError, GaussDBSQLError, GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
    mark_vectorstore_initialized,
)


def _store(
    *,
    embeddings: DeterministicEmbeddings | None = None,
    engine: RecordingEngine | None = None,
) -> tuple[GaussDBVectorStore, DeterministicEmbeddings, RecordingEngine]:
    embeddings = embeddings or DeterministicEmbeddings(query_vector=[1.0, 0.0, 0.0])
    engine = engine or RecordingEngine()
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )
    mark_vectorstore_initialized(store)
    return store, embeddings, engine


def _sql_repr(compiled) -> str:
    return repr(compiled.statement)


def test_similarity_search_applies_metadata_filter_before_ordering():
    engine = RecordingEngine(
        fetch_results=[[("doc-1", "alpha", {"tenant": "t1"}, 0.25)]]
    )
    store, embeddings, _engine = _store(engine=engine)

    documents = store.similarity_search("needle", k=1, filter={"tenant": "t1"})

    assert embeddings.query_calls == ["needle"]
    assert [document.id for document in documents] == ["doc-1"]
    compiled, operation = engine.fetched[0]
    statement = _sql_repr(compiled)
    assert operation == "similarity search vectorstore documents with score"
    assert "WHERE" in statement
    assert "ORDER BY" in statement
    assert "->>" not in statement
    assert "::text" in statement
    assert compiled.params == ("[1.0,0.0,0.0]", '"t1"', 1)


def test_similarity_search_by_vector_applies_metadata_filter_without_embedding():
    engine = RecordingEngine(
        fetch_results=[[("doc-1", "alpha", {"tenant": "t1"}, 0.25)]]
    )
    store, embeddings, _engine = _store(engine=engine)

    store.similarity_search_by_vector([0.0, 1.0, 0.0], k=1, filter={"tenant": "t1"})

    assert embeddings.query_calls == []
    compiled, _operation = engine.fetched[0]
    assert "WHERE" in _sql_repr(compiled)
    assert compiled.params == ("[0.0,1.0,0.0]", '"t1"', 1)


def test_similarity_search_with_score_param_order_keeps_vector_first():
    engine = RecordingEngine(
        fetch_results=[[("doc-1", "alpha", {"tenant": "t1"}, 0.0)]]
    )
    store, _embeddings, _engine = _store(engine=engine)

    store.similarity_search_with_score("needle", k=1, filter={"tenant": "t1"})

    compiled, operation = engine.fetched[0]
    statement = _sql_repr(compiled)
    assert operation == "similarity search vectorstore documents with score"
    assert "AS distance" in statement
    assert "WHERE" in statement
    assert compiled.params == ("[1.0,0.0,0.0]", '"t1"', 1)


def test_similarity_search_with_score_by_vector_applies_filter():
    engine = RecordingEngine(
        fetch_results=[[("doc-1", "alpha", {"tenant": "t1"}, 0.25)]]
    )
    store, _embeddings, _engine = _store(engine=engine)

    results = store.similarity_search_with_score_by_vector(
        [0.0, 1.0, 0.0],
        k=1,
        filter={"score": {"$gte": 0.5}},
    )

    assert results[0][0].id == "doc-1"
    compiled, _operation = engine.fetched[0]
    assert "WHERE" in _sql_repr(compiled)
    assert compiled.params == ("[0.0,1.0,0.0]", 0.5, 1)


def test_mmr_candidate_sql_applies_filter_before_python_rerank():
    engine = RecordingEngine(
        fetch_results=[
            [
                ("doc-1", "alpha", {"tenant": "t1"}, "[1.0,0.0,0.0]", 0.0),
                ("doc-2", "beta", {"tenant": "t1"}, "[0.0,1.0,0.0]", 1.0),
            ]
        ]
    )
    store, _embeddings, _engine = _store(engine=engine)

    documents = store.max_marginal_relevance_search(
        "needle",
        k=1,
        fetch_k=2,
        filter={"tenant": "t1"},
    )

    assert [document.id for document in documents] == ["doc-1"]
    compiled, operation = engine.fetched[0]
    assert operation == "max marginal relevance search vectorstore documents"
    assert "WHERE" in _sql_repr(compiled)
    assert compiled.params == ("[1.0,0.0,0.0]", '"t1"', 2)


def test_mmr_by_vector_applies_filter():
    engine = RecordingEngine(
        fetch_results=[
            [
                ("doc-1", "alpha", {"tenant": "t1"}, "[1.0,0.0,0.0]", 0.0),
            ]
        ]
    )
    store, embeddings, _engine = _store(engine=engine)

    store.max_marginal_relevance_search_by_vector(
        [1.0, 0.0, 0.0],
        k=1,
        fetch_k=1,
        filter={"tenant": "t1"},
    )

    assert embeddings.query_calls == []
    compiled, _operation = engine.fetched[0]
    assert "WHERE" in _sql_repr(compiled)
    assert compiled.params == ("[1.0,0.0,0.0]", '"t1"', 1)


def test_as_retriever_passes_filter_to_similarity_search():
    engine = RecordingEngine(
        fetch_results=[[("doc-1", "alpha", {"tenant": "t1"}, 0.25)]]
    )
    store, _embeddings, _engine = _store(engine=engine)

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1, "filter": {"tenant": "t1"}},
    )
    documents = retriever.invoke("needle")

    assert [document.id for document in documents] == ["doc-1"]
    assert engine.fetched[0][0].params == (
        "[1.0,0.0,0.0]",
        '"t1"',
        1,
    )


@pytest.mark.asyncio
async def test_async_similarity_compiles_filter_through_sync_executor_path():
    store, _embeddings, engine = _store()

    await store.asimilarity_search("needle", k=1, filter={"tenant": "t1"})

    assert engine.fetched[0][0].params == ("[1.0,0.0,0.0]", '"t1"', 1)


@pytest.mark.asyncio
async def test_async_score_compiles_filter_through_sync_executor_path():
    store, _embeddings, engine = _store()

    await store.asimilarity_search_with_score(
        "needle",
        k=1,
        filter={"tenant": "t1"},
    )

    assert engine.fetched[0][0].params == ("[1.0,0.0,0.0]", '"t1"', 1)


@pytest.mark.asyncio
async def test_async_mmr_compiles_filter_through_sync_executor_path():
    store, _embeddings, engine = _store()

    await store.amax_marginal_relevance_search(
        "needle",
        k=1,
        fetch_k=2,
        filter={"tenant": "t1"},
    )

    assert engine.fetched[0][0].params == ("[1.0,0.0,0.0]", '"t1"', 2)


def test_invalid_filter_fails_before_embedding_query():
    store, embeddings, engine = _store()

    with pytest.raises(GaussDBFilterError):
        store.similarity_search("needle", k=1, filter={"$exists": {"tenant": True}})

    assert embeddings.query_calls == []
    assert engine.fetched == []


def test_invalid_filter_fails_before_fetch_by_vector():
    store, _embeddings, engine = _store()

    with pytest.raises(GaussDBFilterError):
        store.similarity_search_by_vector(
            [1.0, 0.0, 0.0],
            k=1,
            filter={"$exists": {"tenant": True}},
        )

    assert engine.fetched == []


@pytest.mark.parametrize("filter_value", [None, {}])
def test_empty_filter_preserves_no_where_sql(filter_value):
    store, _embeddings, engine = _store()

    store.similarity_search_by_vector([1.0, 0.0, 0.0], k=1, filter=filter_value)

    compiled, _operation = engine.fetched[0]
    assert "WHERE" not in _sql_repr(compiled)
    assert compiled.params == ("[1.0,0.0,0.0]", 1)


def test_constructor_only_search_kwargs_are_rejected():
    store, _embeddings, engine = _store()

    with pytest.raises(ValueError, match="Unsupported search kwargs: retrieval_mode"):
        store.similarity_search("needle", k=1, retrieval_mode="hybrid")

    assert engine.fetched == []


def test_unknown_kwargs_are_still_rejected():
    store, _embeddings, engine = _store()

    with pytest.raises(ValueError, match="unexpected"):
        store.similarity_search("needle", k=1, unexpected=True)

    assert engine.fetched == []


class CastFailingEngine(RecordingEngine):
    def fetch_all(self, compiled, *, operation: str = "fetch_all"):
        raise GaussDBSQLError("bad cast", sqlstate="22P02")


def test_filter_cast_failure_is_wrapped_without_values():
    secret_value = "private-tenant-value"
    store, _embeddings, _engine = _store(engine=CastFailingEngine())

    with pytest.raises(GaussDBFilterError) as exc_info:
        store.similarity_search_by_vector(
            [1.0, 0.0, 0.0],
            k=1,
            filter={"tenant": secret_value},
        )

    message = str(exc_info.value)
    assert "metadata filter" in message
    assert "tenant" in message
    assert secret_value not in message
    assert exc_info.value.__cause__ is None
