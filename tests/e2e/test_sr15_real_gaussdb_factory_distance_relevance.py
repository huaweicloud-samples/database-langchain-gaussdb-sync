"""SR-15 real-DB coverage for automatic factories and distance/relevance."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.indexes import build_create_vector_index
from langchain_gaussdb.sql import CompiledSQL

pytestmark = pytest.mark.gaussdb_e2e


class _DistinctEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        return [list(vectors[index % 3]) for index, _text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _options(engine: GaussDBEngine, table: Any) -> dict[str, Any]:
    return {
        "embedding": _DistinctEmbeddings(),
        "engine": engine,
        "schema_name": table.schema,
        "table_name": table.name,
        "embedding_dimension": 3,
    }


def _index_exists(engine: GaussDBEngine, schema: str, name: str) -> bool:
    return bool(
        engine.fetch_all(
            CompiledSQL(
                sql.SQL(
                    "SELECT 1 FROM pg_class AS idx "
                    "JOIN pg_namespace AS ns ON ns.oid = idx.relnamespace "
                    "WHERE ns.nspname = %s AND idx.relname = %s"
                ),
                (schema, name),
            ),
            operation="check automatic factory vector index",
        )
    )


def test_from_texts_automatically_creates_vector_index_and_searches(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = GaussDBVectorStore.from_texts(
        ["alpha", "beta", "gamma"],
        ids=["d1", "d2", "d3"],
        **_options(writable_engine, temporary_vector_table),
    )
    index_name, _ = build_create_vector_index(
        temporary_vector_table.schema,
        temporary_vector_table.name,
        "embedding",
        3,
    )

    assert _index_exists(writable_engine, temporary_vector_table.schema, index_name)
    assert store.similarity_search("alpha", k=1)[0].id == "d1"


def test_from_documents_uses_the_same_automatic_contract(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = GaussDBVectorStore.from_documents(
        [Document(id="document-id", page_content="alpha")],
        **_options(writable_engine, temporary_vector_table),
    )

    assert store.get_by_ids(["document-id"])[0].id == "document-id"


@pytest.mark.parametrize("factory", ["from_texts", "from_documents"])
def test_factories_reject_drop_old(
    factory: str,
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
) -> None:
    kwargs = _options(writable_engine, temporary_vector_table)

    with pytest.raises(ValueError, match="drop_old"):
        if factory == "from_texts":
            GaussDBVectorStore.from_texts(["alpha"], drop_old=True, **kwargs)
        else:
            GaussDBVectorStore.from_documents(
                [Document(page_content="alpha")], drop_old=True, **kwargs
            )


@pytest.mark.parametrize("distance_strategy", ["cosine", "l2"])
def test_distance_strategy_returns_nearest(
    distance_strategy: str,
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = GaussDBVectorStore(
        **_options(writable_engine, temporary_vector_table),
        distance_strategy=distance_strategy,
    )
    store.add_texts(["alpha", "beta", "gamma"], ids=["d1", "d2", "d3"])

    assert store.similarity_search("alpha", k=1)[0].id == "d1"


def test_unsupported_distance_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="distance_strategy"):
        GaussDBVectorStore(
            embedding=_DistinctEmbeddings(),
            engine=object(),
            table_name="documents",
            embedding_dimension=3,
            distance_strategy="unsupported",
        )


def test_relevance_scores_dense_mode_works(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = GaussDBVectorStore(**_options(writable_engine, temporary_vector_table))
    store.add_texts(["alpha", "beta"], ids=["d1", "d2"])

    scored = store.similarity_search_with_relevance_scores("alpha", k=2)

    assert scored[0][0].id == "d1"
    assert all(0.0 <= score <= 1.0 for _document, score in scored)


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_relevance_scores_gate_rejects_non_dense(mode: str) -> None:
    store = GaussDBVectorStore(
        embedding=_DistinctEmbeddings(),
        engine=object(),
        table_name="documents",
        embedding_dimension=3,
        retrieval_mode=mode,
    )

    with pytest.raises(ValueError, match="dense"):
        store.similarity_search_with_relevance_scores("alpha", k=1)
