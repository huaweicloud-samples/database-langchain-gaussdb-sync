from __future__ import annotations

import math

import pytest

from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
)


def _store(
    *,
    embeddings: DeterministicEmbeddings | None = None,
    engine: RecordingEngine | None = None,
    distance_strategy: str = "cosine",
    retrieval_mode: str = "dense",
) -> tuple[GaussDBVectorStore, DeterministicEmbeddings, RecordingEngine]:
    embeddings = embeddings or DeterministicEmbeddings()
    engine = engine or RecordingEngine()
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        distance_strategy=distance_strategy,
        retrieval_mode=retrieval_mode,
    )
    return store, embeddings, engine


def test_constructor_defaults_to_cosine_distance_strategy():
    store, _embeddings, _engine = _store()

    assert store._distance_strategy == "cosine"
    assert store._distance_operator == "<+>"


def test_constructor_accepts_l2_distance_strategy():
    store, _embeddings, _engine = _store(distance_strategy="l2")

    assert store._distance_strategy == "l2"
    assert store._distance_operator == "<->"


@pytest.mark.parametrize("distance_strategy", ["cosine; DROP TABLE documents", []])
def test_constructor_rejects_unknown_distance_strategy(distance_strategy):
    with pytest.raises(ValueError) as exc_info:
        _store(distance_strategy=distance_strategy)

    message = str(exc_info.value)
    assert "distance_strategy" in message
    assert "cosine" in message
    assert "l2" in message


def test_prepare_search_extracts_filter_compiles_it_and_validates_k():
    store, _embeddings, _engine = _store()
    kwargs = {"filter": {"source": "unit"}}
    expected_filter = store._compile_metadata_filter(kwargs["filter"])

    result_count, compiled_filter = store._prepare_search(2, kwargs)

    assert result_count == 2
    assert compiled_filter == expected_filter
    assert kwargs == {}


@pytest.mark.parametrize(
    ("k", "kwargs", "message"),
    [
        (-1, {}, "k must be a non-negative integer"),
        (
            1,
            {"retrieval_mode": "hybrid"},
            "Unsupported search kwargs: retrieval_mode",
        ),
        (
            1,
            {"drop_old": True},
            "Unsupported search kwargs: drop_old",
        ),
        (
            1,
            {"unexpected": True},
            "Unsupported search kwargs: unexpected",
        ),
    ],
)
def test_prepare_search_preserves_validation_errors(k, kwargs, message):
    store, _embeddings, _engine = _store()

    with pytest.raises(ValueError, match=message):
        store._prepare_search(k, kwargs)


@pytest.mark.parametrize("retrieval_mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"drop_old": True}, "Unsupported search kwargs: drop_old"),
        ({"unexpected": True}, "Unsupported search kwargs: unexpected"),
    ],
)
def test_prepare_search_uses_mode_neutral_validation_errors(
    retrieval_mode,
    kwargs,
    message,
):
    store, _embeddings, _engine = _store(retrieval_mode=retrieval_mode)

    with pytest.raises(ValueError, match=message):
        store._prepare_search(1, kwargs)


def test_similarity_search_k_zero_returns_empty_without_sql():
    store, embeddings, engine = _store()

    documents = store.similarity_search("alpha", k=0)

    assert documents == []
    assert embeddings.query_calls == []
    assert engine.fetched == []


@pytest.mark.parametrize("k", [-1, True, 1.5, "4"])
def test_similarity_search_rejects_invalid_k(k):
    store, embeddings, engine = _store()

    with pytest.raises(ValueError, match="k"):
        store.similarity_search("alpha", k=k)

    assert embeddings.query_calls == []
    assert engine.fetched == []


@pytest.mark.parametrize(
    "vector",
    [
        [0.0, 1.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, math.nan, 2.0],
        [0.0, math.inf, 2.0],
    ],
)
def test_similarity_search_rejects_bad_query_embedding(vector):
    store, embeddings, engine = _store(
        embeddings=DeterministicEmbeddings(query_vector=vector)
    )

    with pytest.raises(ValueError, match="embedding"):
        store.similarity_search("alpha", k=1)

    assert embeddings.query_calls == ["alpha"]
    assert engine.fetched == []


@pytest.mark.parametrize("filter_value", [None, {}])
@pytest.mark.parametrize(
    "method_name,args",
    [
        ("similarity_search", ("alpha",)),
        ("similarity_search_by_vector", ([0.0, 1.0, 2.0],)),
        ("similarity_search_with_score", ("alpha",)),
        ("similarity_search_with_score_by_vector", ([0.0, 1.0, 2.0],)),
        ("max_marginal_relevance_search", ("alpha",)),
        ("max_marginal_relevance_search_by_vector", ([0.0, 1.0, 2.0],)),
    ],
)
def test_search_paths_accept_empty_filter(method_name, args, filter_value):
    store, embeddings, engine = _store()
    method = getattr(store, method_name)

    result = method(*args, k=0, filter=filter_value)

    assert result == []
    assert embeddings.query_calls == []
    assert engine.fetched == []


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("similarity_search", ("alpha",)),
        ("similarity_search_by_vector", ([0.0, 1.0, 2.0],)),
        ("similarity_search_with_score", ("alpha",)),
        ("similarity_search_with_score_by_vector", ([0.0, 1.0, 2.0],)),
        ("max_marginal_relevance_search", ("alpha",)),
        ("max_marginal_relevance_search_by_vector", ([0.0, 1.0, 2.0],)),
    ],
)
def test_search_paths_accept_valid_non_empty_filter_at_k_zero(method_name, args):
    store, embeddings, engine = _store()
    method = getattr(store, method_name)

    result = method(*args, k=0, filter={"source": "unit"})

    assert result == []
    assert embeddings.query_calls == []
    assert engine.fetched == []


@pytest.mark.parametrize("lambda_mult", [-0.1, 1.1, math.nan, True, "0.5"])
def test_mmr_rejects_invalid_lambda_mult(lambda_mult):
    store, embeddings, engine = _store()

    with pytest.raises(ValueError, match="lambda_mult"):
        store.max_marginal_relevance_search(
            "alpha",
            k=1,
            fetch_k=1,
            lambda_mult=lambda_mult,
        )

    assert embeddings.query_calls == []
    assert engine.fetched == []


def test_search_paths_use_user_facing_unknown_kwarg_message():
    store, _embeddings, engine = _store()

    with pytest.raises(ValueError) as exc_info:
        store.similarity_search("alpha", k=1, unexpected=True)

    message = str(exc_info.value)
    assert "search" in message
    assert "unexpected" in message
    assert "SR-" not in message
    assert engine.fetched == []


def test_search_paths_reject_constructor_only_retrieval_mode():
    store, _embeddings, engine = _store()

    with pytest.raises(ValueError) as exc_info:
        store.similarity_search("alpha", k=1, retrieval_mode="hybrid")

    message = str(exc_info.value)
    assert "retrieval_mode" in message
    assert "Unsupported search kwargs" in message
    assert "SR-" not in message
    assert engine.fetched == []
