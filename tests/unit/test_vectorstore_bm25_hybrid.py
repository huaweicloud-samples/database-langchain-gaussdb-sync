from __future__ import annotations

import asyncio
import inspect

import pytest
from langchain_core.documents import Document

import langchain_gaussdb.vectorstore as vectorstore_module
from langchain_gaussdb import BM25Config, GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
    mark_vectorstore_initialized,
)


def test_bm25_runtime_path_has_no_capability_probe_api() -> None:
    assert not hasattr(GaussDBVectorStore, "supports_bm25_operator")
    assert not hasattr(GaussDBVectorStore, "has_bm25_index")
    parameters = inspect.signature(
        GaussDBVectorStore._bm25_search_with_score
    ).parameters
    assert "initialize" not in parameters
    assert "check_capability" not in parameters


def _store(
    mode: str,
    *,
    fetch_results=None,
    bm25_config: BM25Config | None = None,
) -> tuple[GaussDBVectorStore, RecordingEngine, DeterministicEmbeddings]:
    engine = RecordingEngine(fetch_results=fetch_results)
    embeddings = DeterministicEmbeddings(query_vector=[0.0, 1.0, 2.0])
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        retrieval_mode=mode,
        bm25_config=bm25_config,
        engine=engine,
    )
    mark_vectorstore_initialized(store)
    return store, engine, embeddings


def test_bm25_search_does_not_embed_query() -> None:
    store, engine, embeddings = _store(
        "bm25",
        fetch_results=[[("doc-1", "GaussDB BM25", {"source": "unit"}, 0.9)]],
    )

    documents = store.similarity_search("GaussDB", k=1)

    assert documents == [
        Document(id="doc-1", page_content="GaussDB BM25", metadata={"source": "unit"})
    ]
    assert embeddings.query_calls == []
    assert [operation for _compiled, operation in engine.fetched] == [
        "bm25 search vectorstore documents"
    ]


def test_bm25_search_returns_database_score_without_readiness_probes() -> None:
    store, engine, _embeddings = _store(
        "bm25",
        fetch_results=[[("doc-1", "alpha", {}, 1.25)]],
    )

    results = store.similarity_search_with_score("alpha", k=1)

    assert results[0][1] == 1.25
    assert len(engine.fetched) == 1


def test_bm25_query_override_is_forwarded_to_sql() -> None:
    store, engine, _embeddings = _store("bm25", fetch_results=[[]])

    assert store.similarity_search("raw", bm25_query="normalized") == []

    assert engine.fetched[0][0].params[0] == "normalized"


def test_dense_search_rejects_bm25_query_instead_of_ignoring_it() -> None:
    store, engine, embeddings = _store("dense")

    with pytest.raises(ValueError, match="bm25_query is not supported for dense"):
        store.similarity_search("raw", k=1, bm25_query="normalized")

    assert embeddings.query_calls == []
    assert engine.fetched == []


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_lexical_modes_validate_bm25_query_even_when_k_is_zero(mode: str) -> None:
    store, engine, embeddings = _store(mode)

    with pytest.raises(ValueError, match="bm25_query must be a string"):
        store.similarity_search("raw", k=0, bm25_query=7)

    assert embeddings.query_calls == []
    assert engine.fetched == []


def test_projection_column_is_used_for_bm25_query() -> None:
    store, engine, _embeddings = _store(
        "bm25",
        fetch_results=[[]],
        bm25_config=BM25Config(column="content_lexical"),
    )

    store.similarity_search("normalized")

    assert "Identifier('content_lexical')" in repr(engine.fetched[0][0].statement)


def test_hybrid_search_runs_dense_and_bm25_then_fuses() -> None:
    store, engine, embeddings = _store(
        "hybrid",
        fetch_results=[
            [("dense", "dense result", {}, 0.1)],
            [("lexical", "lexical result", {}, 2.0)],
        ],
    )

    documents = store.similarity_search("needle", k=2)

    assert {document.id for document in documents} == {"dense", "lexical"}
    assert embeddings.query_calls == ["needle"]
    assert [operation for _compiled, operation in engine.fetched] == [
        "hybrid dense search vectorstore documents with score",
        "hybrid bm25 search vectorstore documents with score",
    ]


def test_hybrid_search_serializes_query_embedding_once(monkeypatch) -> None:
    store, _engine, _embeddings = _store(
        "hybrid",
        fetch_results=[
            [("doc-1", "shared", {}, 0.1)],
            [("doc-1", "shared", {}, 2.0)],
        ],
    )
    original = vectorstore_module._embedding_to_vector_string
    calls: list[list[float]] = []

    def tracked(embedding, *, expected_dimension, index):
        calls.append(list(embedding))
        return original(
            embedding,
            expected_dimension=expected_dimension,
            index=index,
        )

    monkeypatch.setattr(vectorstore_module, "_embedding_to_vector_string", tracked)

    store.similarity_search("needle", k=1)

    assert calls == [[0.0, 1.0, 2.0]]


def test_hybrid_search_with_score_returns_fusion_scores() -> None:
    store, _engine, _embeddings = _store(
        "hybrid",
        fetch_results=[
            [("doc-1", "shared", {}, 0.1)],
            [("doc-1", "shared", {}, 2.0)],
        ],
    )

    results = store.similarity_search_with_score("needle", k=1)

    assert results[0][0].id == "doc-1"
    assert results[0][1] > 0


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_empty_search_skips_embedding_and_database(mode: str) -> None:
    store, engine, embeddings = _store(mode)

    assert store.similarity_search("needle", k=0) == []

    assert embeddings.query_calls == []
    assert engine.fetched == []


def test_bm25_retriever_uses_standard_vectorstore_path() -> None:
    store, _engine, _embeddings = _store(
        "bm25", fetch_results=[[("doc-1", "alpha", {}, 1.0)]]
    )

    documents = store.as_retriever(search_kwargs={"k": 1}).invoke("alpha")

    assert [document.id for document in documents] == ["doc-1"]


def test_hybrid_retriever_uses_standard_vectorstore_path() -> None:
    store, _engine, _embeddings = _store(
        "hybrid",
        fetch_results=[
            [("doc-1", "alpha", {}, 0.1)],
            [("doc-1", "alpha", {}, 1.0)],
        ],
    )

    documents = store.as_retriever(search_kwargs={"k": 1}).invoke("alpha")

    assert [document.id for document in documents] == ["doc-1"]


def test_async_bm25_search_uses_sync_path_in_executor() -> None:
    store, _engine, embeddings = _store(
        "bm25", fetch_results=[[("doc-1", "alpha", {}, 1.0)]]
    )

    documents = asyncio.run(store.asimilarity_search("alpha", k=1))

    assert [document.id for document in documents] == ["doc-1"]
    assert embeddings.query_calls == []
