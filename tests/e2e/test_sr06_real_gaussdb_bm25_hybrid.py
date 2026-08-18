from __future__ import annotations

from typing import Any

import pytest

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings

pytestmark = [pytest.mark.gaussdb_e2e, pytest.mark.gaussdb_e2e_fast]


def _store(engine: GaussDBEngine, table: Any, mode: str) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        engine=engine,
        embedding=DeterministicEmbeddings(dimension=3),
        schema_name=table.schema,
        table_name=table.name,
        embedding_dimension=3,
        retrieval_mode=mode,
    )


def test_bm25_and_hybrid_use_mode_configured_standard_retrievers(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
    requires_bm25: bool,
) -> None:
    dense_store = _store(writable_engine, temporary_vector_table, "dense")
    dense_store.add_texts(
        ["lexicalneedle lexicalneedle", "unrelated text"],
        ids=["match", "control"],
    )
    bm25_store = _store(writable_engine, temporary_vector_table, "bm25")
    hybrid_store = _store(writable_engine, temporary_vector_table, "hybrid")
    bm25_store.setup()
    hybrid_store.setup()

    bm25_documents = bm25_store.as_retriever(search_kwargs={"k": 2}).invoke(
        "lexicalneedle"
    )
    hybrid_documents = hybrid_store.as_retriever(search_kwargs={"k": 2}).invoke(
        "lexicalneedle"
    )

    assert bm25_documents[0].id == "match"
    assert {document.id for document in hybrid_documents} >= {"match"}
    assert bm25_store.retrieval_mode == "bm25"
    assert hybrid_store.retrieval_mode == "hybrid"
