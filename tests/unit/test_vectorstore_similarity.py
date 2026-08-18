from __future__ import annotations

from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
    mark_vectorstore_initialized,
)


def _store(
    *,
    embeddings: DeterministicEmbeddings | None = None,
    engine: RecordingEngine | None = None,
    distance_strategy: str = "cosine",
) -> tuple[GaussDBVectorStore, DeterministicEmbeddings, RecordingEngine]:
    embeddings = embeddings or DeterministicEmbeddings(query_vector=[1.0, 0.0, 0.0])
    engine = engine or RecordingEngine(
        fetch_results=[
            [
                ("doc-1", "same", {"rank": 1}, 0.0),
                ("doc-2", "near", '{"rank":2}', 1.5),
            ]
        ]
    )
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        distance_strategy=distance_strategy,
    )
    mark_vectorstore_initialized(store)
    return store, embeddings, engine


def test_similarity_search_embeds_query_and_returns_documents():
    store, embeddings, engine = _store()

    documents = store.similarity_search("needle", k=2)

    assert embeddings.query_calls == ["needle"]
    assert [document.id for document in documents] == ["doc-1", "doc-2"]
    assert [document.page_content for document in documents] == ["same", "near"]
    assert documents[0].metadata == {"rank": 1}
    assert documents[1].metadata == {"rank": 2}
    compiled, operation = engine.fetched[0]
    assert operation == "similarity search vectorstore documents with score"
    statement = repr(compiled.statement)
    assert "AS distance" in statement
    assert "ORDER BY" in statement
    assert "<+>" in statement
    assert "floatvector" in statement
    assert "3" in statement
    assert compiled.params == ("[1.0,0.0,0.0]", 2)


def test_similarity_search_by_vector_does_not_embed_query():
    store, embeddings, engine = _store()

    documents = store.similarity_search_by_vector([0.0, 1.0, 0.0], k=1)

    assert embeddings.query_calls == []
    assert [document.id for document in documents] == ["doc-1", "doc-2"]
    compiled, _operation = engine.fetched[0]
    assert "AS distance" in repr(compiled.statement)
    assert compiled.params == ("[0.0,1.0,0.0]", 1)


def test_similarity_search_with_score_returns_raw_distance():
    engine = RecordingEngine(
        fetch_results=[
            [
                ("doc-1", "same", {"rank": 1}, 0),
                ("doc-2", "near", {"rank": 2}, "1.5"),
            ]
        ]
    )
    store, embeddings, _engine = _store(engine=engine)

    results = store.similarity_search_with_score("needle", k=2)

    assert embeddings.query_calls == ["needle"]
    assert [(document.id, score) for document, score in results] == [
        ("doc-1", 0.0),
        ("doc-2", 1.5),
    ]
    compiled, operation = engine.fetched[0]
    assert operation == "similarity search vectorstore documents with score"
    statement = repr(compiled.statement)
    assert "AS distance" in statement
    assert "ORDER BY" in statement
    assert compiled.params == ("[1.0,0.0,0.0]", 2)


def test_similarity_search_with_score_by_vector_returns_raw_distance():
    engine = RecordingEngine(
        fetch_results=[
            [
                ("doc-1", "same", {"rank": 1}, 0.25),
            ]
        ]
    )
    store, embeddings, _engine = _store(engine=engine)

    results = store.similarity_search_with_score_by_vector([0.0, 1.0, 0.0], k=1)

    assert embeddings.query_calls == []
    assert len(results) == 1
    document, score = results[0]
    assert document.id == "doc-1"
    assert score == 0.25
    compiled, _operation = engine.fetched[0]
    assert compiled.params == ("[0.0,1.0,0.0]", 1)


def test_l2_similarity_search_uses_l2_operator():
    store, _embeddings, engine = _store(distance_strategy="l2")

    store.similarity_search_by_vector([0.0, 1.0, 0.0], k=1)

    statement = repr(engine.fetched[0][0].statement)
    assert "<->" in statement
    assert "<+>" not in statement


def test_similarity_search_binds_vector_and_limit_as_params():
    malicious_query_vector = [9.0, 8.0, 7.0]
    store, _embeddings, engine = _store(
        embeddings=DeterministicEmbeddings(query_vector=malicious_query_vector)
    )

    store.similarity_search("x'); DROP TABLE documents; --", k=2)

    compiled, _operation = engine.fetched[0]
    statement = repr(compiled.statement)
    assert compiled.params == ("[9.0,8.0,7.0]", 2)
    assert "[9.0,8.0,7.0]" not in statement
    assert "DROP TABLE" not in statement
