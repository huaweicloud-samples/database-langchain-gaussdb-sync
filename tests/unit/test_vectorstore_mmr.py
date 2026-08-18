from __future__ import annotations

import pytest

import langchain_gaussdb.vectorstore as vectorstore_module
from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
    mark_vectorstore_initialized,
)


def _store(
    *,
    engine: RecordingEngine | None = None,
) -> tuple[GaussDBVectorStore, DeterministicEmbeddings, RecordingEngine]:
    embeddings = DeterministicEmbeddings(query_vector=[1.0, 0.0, 0.0])
    engine = engine or RecordingEngine()
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )
    mark_vectorstore_initialized(store)
    return store, embeddings, engine


def test_mmr_search_fetches_candidates_with_embedding_text_and_returns_k_documents():
    store, embeddings, engine = _store(
        engine=RecordingEngine(
            fetch_results=[
                [
                    ("doc-1", "same", {"rank": 1}, "[1,0,0]", 0.0),
                    ("doc-2", "orthogonal", {"rank": 2}, "[0,1,0]", 1.0),
                    ("doc-3", "other", {"rank": 3}, "[0,0,1]", 1.0),
                ]
            ]
        )
    )

    documents = store.max_marginal_relevance_search(
        "needle",
        k=2,
        fetch_k=3,
        lambda_mult=1.0,
    )

    assert embeddings.query_calls == ["needle"]
    assert [document.id for document in documents] == ["doc-1", "doc-2"]
    compiled, operation = engine.fetched[0]
    assert operation == "max marginal relevance search vectorstore documents"
    statement = repr(compiled.statement)
    assert "embedding" in statement
    assert "::text" in statement
    assert "AS distance" in statement
    assert "ORDER BY" in statement
    assert compiled.params == ("[1.0,0.0,0.0]", 3)


def test_mmr_search_by_vector_does_not_embed_query():
    store, embeddings, engine = _store(
        engine=RecordingEngine(
            fetch_results=[
                [
                    ("doc-1", "same", {}, "[1,0,0]", 0.0),
                ]
            ]
        )
    )

    documents = store.max_marginal_relevance_search_by_vector(
        [1.0, 0.0, 0.0],
        k=1,
        fetch_k=1,
    )

    assert embeddings.query_calls == []
    assert [document.id for document in documents] == ["doc-1"]


def test_mmr_only_parses_candidate_embeddings_from_database(monkeypatch):
    store, _embeddings, _engine = _store(
        engine=RecordingEngine(
            fetch_results=[
                [
                    ("doc-1", "same", {}, "[1,0,0]", 0.0),
                ]
            ]
        )
    )
    original = vectorstore_module._vector_string_to_floats
    parsed_values = []

    def tracked(value, *, expected_dimension):
        parsed_values.append(value)
        return original(value, expected_dimension=expected_dimension)

    monkeypatch.setattr(vectorstore_module, "_vector_string_to_floats", tracked)

    store.max_marginal_relevance_search_by_vector(
        [1.0, 0.0, 0.0],
        k=1,
        fetch_k=1,
    )

    assert parsed_values == ["[1,0,0]"]


def test_mmr_k_zero_returns_empty_without_sql():
    store, embeddings, engine = _store()

    documents = store.max_marginal_relevance_search("needle", k=0, fetch_k=3)

    assert documents == []
    assert embeddings.query_calls == []
    assert engine.fetched == []


def test_mmr_fetch_k_zero_returns_empty_without_sql():
    store, embeddings, engine = _store()

    documents = store.max_marginal_relevance_search("needle", k=2, fetch_k=0)

    assert documents == []
    assert embeddings.query_calls == []
    assert engine.fetched == []


@pytest.mark.parametrize("fetch_k", [-1, True, 1.5, "4"])
def test_mmr_rejects_invalid_fetch_k(fetch_k):
    store, embeddings, engine = _store()

    with pytest.raises(ValueError, match="fetch_k"):
        store.max_marginal_relevance_search("needle", k=1, fetch_k=fetch_k)

    assert embeddings.query_calls == []
    assert engine.fetched == []


@pytest.mark.parametrize("value", ["not-a-vector", "[1,0]", "[1,nan,0]", None])
def test_mmr_rejects_invalid_database_embedding_text(value):
    store, _embeddings, _engine = _store(
        engine=RecordingEngine(
            fetch_results=[
                [
                    ("doc-1", "same", {"secret": "do-not-print"}, value, 0.0),
                ]
            ]
        )
    )

    with pytest.raises(ValueError) as exc_info:
        store.max_marginal_relevance_search("needle", k=1, fetch_k=1)

    message = str(exc_info.value)
    assert "embedding" in message
    assert "do-not-print" not in message
    if isinstance(value, str):
        assert value not in message


def test_mmr_binds_vector_and_fetch_k_as_params():
    store, _embeddings, engine = _store(
        engine=RecordingEngine(
            fetch_results=[
                [
                    ("doc-1", "same", {}, "[1,0,0]", 0.0),
                ]
            ]
        )
    )

    store.max_marginal_relevance_search("x'); DROP TABLE documents; --", k=1, fetch_k=1)

    compiled, _operation = engine.fetched[0]
    statement = repr(compiled.statement)
    assert compiled.params == ("[1.0,0.0,0.0]", 1)
    assert "[1.0,0.0,0.0]" not in statement
    assert "DROP TABLE" not in statement
