from __future__ import annotations

import pytest
from langchain_core.documents import Document

from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
    mark_vectorstore_initialized,
)


def _store(
    *,
    fetch_results=None,
    distance_strategy: str = "cosine",
) -> tuple[GaussDBVectorStore, RecordingEngine]:
    engine = RecordingEngine(fetch_results=fetch_results)
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(query_vector=[1.0, 0.0, 0.0]),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        distance_strategy=distance_strategy,
    )
    mark_vectorstore_initialized(store)
    return store, engine


def test_cosine_relevance_score_clamps_to_zero_and_one():
    store, _engine = _store()
    relevance = store._select_relevance_score_fn()

    assert relevance(0.0) == 1.0
    assert relevance(1.0) == 0.5
    assert relevance(2.0) == 0.0
    assert relevance(3.0) == 0.0


def test_l2_relevance_score_clamps_to_zero_and_one():
    store, _engine = _store(distance_strategy="l2")
    relevance = store._select_relevance_score_fn()

    assert relevance(0.0) == 1.0
    assert relevance(1.0) == 0.5
    assert relevance(3.0) == 0.25
    assert relevance(-0.5) == 1.0


def test_similarity_search_with_relevance_scores_applies_threshold():
    store, _engine = _store(
        fetch_results=[
            [
                ("doc-1", "same", {"rank": 1}, 0.0),
                ("doc-2", "near", {"rank": 2}, 1.0),
                ("doc-3", "opposite", {"rank": 3}, 2.0),
            ]
        ]
    )

    results = store.similarity_search_with_relevance_scores(
        "needle",
        k=3,
        score_threshold=0.5,
    )

    assert [(document.id, score) for document, score in results] == [
        ("doc-1", 1.0),
        ("doc-2", 0.5),
    ]


def test_search_dispatches_similarity():
    store, _engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    documents = store.search("needle", "similarity", k=1)

    assert [document.id for document in documents] == ["doc-1"]


def test_search_dispatches_similarity_score_threshold():
    store, _engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])

    documents = store.search(
        "needle",
        "similarity_score_threshold",
        k=1,
        score_threshold=0.8,
    )

    assert [document.id for document in documents] == ["doc-1"]


def test_search_dispatches_mmr(monkeypatch):
    store, _engine = _store()
    calls = []

    def fake_mmr(query, **kwargs):
        calls.append((query, kwargs))
        return [Document(id="doc-mmr", page_content="mmr")]

    monkeypatch.setattr(store, "max_marginal_relevance_search", fake_mmr)

    documents = store.search("needle", "mmr", k=1, fetch_k=1)

    assert [document.id for document in documents] == ["doc-mmr"]
    assert calls == [("needle", {"k": 1, "fetch_k": 1})]


def test_as_retriever_similarity_invoke_returns_documents():
    store, _engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    documents = retriever.invoke("needle")

    assert [document.id for document in documents] == ["doc-1"]


def test_as_retriever_score_threshold_invoke_returns_documents():
    store, _engine = _store(fetch_results=[[("doc-1", "same", {}, 0.0)]])
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.8},
    )

    documents = retriever.invoke("needle")

    assert [document.id for document in documents] == ["doc-1"]


def test_as_retriever_mmr_invoke_returns_documents(monkeypatch):
    store, _engine = _store()
    calls = []

    def fake_mmr(query, **kwargs):
        calls.append((query, kwargs))
        return [Document(id="doc-mmr", page_content="mmr")]

    monkeypatch.setattr(store, "max_marginal_relevance_search", fake_mmr)
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1, "fetch_k": 1},
    )

    documents = retriever.invoke("needle")

    assert [document.id for document in documents] == ["doc-mmr"]
    assert calls == [("needle", {"k": 1, "fetch_k": 1})]


def test_as_retriever_rejects_non_standard_search_type():
    store, _engine = _store()

    with pytest.raises(ValueError, match="search_type"):
        store.as_retriever(search_type="hybrid")


def test_as_retriever_threshold_requires_score_threshold():
    store, _engine = _store()

    with pytest.raises(ValueError, match="score_threshold"):
        store.as_retriever(search_type="similarity_score_threshold")
