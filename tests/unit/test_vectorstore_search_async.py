from __future__ import annotations

import asyncio

from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
    mark_vectorstore_initialized,
)


def _store(fetch_results=None) -> tuple[GaussDBVectorStore, RecordingEngine]:
    engine = RecordingEngine(fetch_results=fetch_results)
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(query_vector=[1.0, 0.0, 0.0]),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )
    mark_vectorstore_initialized(store)
    return store, engine


def test_asimilarity_search_uses_sync_query_and_engine_in_executor():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    documents = asyncio.run(store.asimilarity_search("needle", k=1))

    assert [document.id for document in documents] == ["doc-1"]
    assert len(engine.fetched) == 1
    assert store.embeddings.async_query_calls == []
    assert store.embeddings.query_calls == ["needle"]


def test_asimilarity_search_by_vector_uses_sync_engine_without_embedding():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    documents = asyncio.run(store.asimilarity_search_by_vector([1.0, 0.0, 0.0], k=1))

    assert [document.id for document in documents] == ["doc-1"]
    assert len(engine.fetched) == 1
    assert store.embeddings.async_query_calls == []
    assert store.embeddings.query_calls == []


def test_asimilarity_search_with_score_uses_sync_engine_in_executor():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    results = asyncio.run(store.asimilarity_search_with_score("needle", k=1))

    assert [(document.id, score) for document, score in results] == [("doc-1", 0.0)]
    assert len(engine.fetched) == 1
    assert store.embeddings.async_query_calls == []
    assert store.embeddings.query_calls == ["needle"]


def test_asimilarity_search_with_relevance_scores_uses_sync_engine_in_executor():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    results = asyncio.run(
        store.asimilarity_search_with_relevance_scores(
            "needle",
            k=1,
            score_threshold=0.8,
        )
    )

    assert [(document.id, score) for document, score in results] == [("doc-1", 1.0)]
    assert len(engine.fetched) == 1
    assert store.embeddings.async_query_calls == []
    assert store.embeddings.query_calls == ["needle"]


def test_amax_marginal_relevance_search_uses_sync_query_and_engine_in_executor():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, "[1,0,0]", 0.0)]])

    documents = asyncio.run(
        store.amax_marginal_relevance_search("needle", k=1, fetch_k=1)
    )

    assert [document.id for document in documents] == ["doc-1"]
    assert len(engine.fetched) == 1
    assert store.embeddings.async_query_calls == []
    assert store.embeddings.query_calls == ["needle"]


def test_amax_marginal_relevance_search_by_vector_skips_embedding():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, "[1,0,0]", 0.0)]])

    documents = asyncio.run(
        store.amax_marginal_relevance_search_by_vector(
            [1.0, 0.0, 0.0],
            k=1,
            fetch_k=1,
        )
    )

    assert [document.id for document in documents] == ["doc-1"]
    assert len(engine.fetched) == 1
    assert store.embeddings.async_query_calls == []
    assert store.embeddings.query_calls == []


def test_asearch_dispatches_similarity_async():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    documents = asyncio.run(store.asearch("needle", "similarity", k=1))

    assert [document.id for document in documents] == ["doc-1"]
    assert len(engine.fetched) == 1


def test_asearch_dispatches_similarity_score_threshold_async():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    documents = asyncio.run(
        store.asearch(
            "needle",
            "similarity_score_threshold",
            k=1,
            score_threshold=0.8,
        )
    )

    assert [document.id for document in documents] == ["doc-1"]
    assert len(engine.fetched) == 1


def test_asearch_dispatches_mmr_async():
    store, engine = _store(fetch_results=[[("doc-1", "same", {}, "[1,0,0]", 0.0)]])

    documents = asyncio.run(store.asearch("needle", "mmr", k=1, fetch_k=1))

    assert [document.id for document in documents] == ["doc-1"]
    assert len(engine.fetched) == 1


def test_as_retriever_ainvoke_dispatches_similarity_threshold_and_mmr():
    similarity_store, similarity_engine = _store(
        fetch_results=[[("doc-sim", "same", {}, 0.0)]]
    )
    threshold_store, threshold_engine = _store(
        fetch_results=[[("doc-threshold", "same", {}, 0.0)]]
    )
    mmr_store, mmr_engine = _store(
        fetch_results=[[("doc-mmr", "same", {}, "[1,0,0]", 0.0)]]
    )

    similarity_docs = asyncio.run(
        similarity_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1},
        ).ainvoke("needle")
    )
    threshold_docs = asyncio.run(
        threshold_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.8},
        ).ainvoke("needle")
    )
    mmr_docs = asyncio.run(
        mmr_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "fetch_k": 1},
        ).ainvoke("needle")
    )

    assert [document.id for document in similarity_docs] == ["doc-sim"]
    assert [document.id for document in threshold_docs] == ["doc-threshold"]
    assert [document.id for document in mmr_docs] == ["doc-mmr"]
    assert len(similarity_engine.fetched) == 1
    assert len(threshold_engine.fetched) == 1
    assert len(mmr_engine.fetched) == 1
