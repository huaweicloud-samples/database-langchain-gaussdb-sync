from __future__ import annotations

from typing import Any

import pytest
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.indexes import build_create_vector_index
from langchain_gaussdb.sql import CompiledSQL
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings

pytestmark = [
    pytest.mark.gaussdb_e2e,
    pytest.mark.gaussdb_e2e_fast,
    pytest.mark.gaussdb_e2e_full,
]


def _store(engine: GaussDBEngine, table: Any, mode: str) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        engine=engine,
        embedding=DeterministicEmbeddings(dimension=3),
        schema_name=table.schema,
        table_name=table.name,
        embedding_dimension=3,
        retrieval_mode=mode,
    )


def test_default_schema_configuration_is_public(
    writable_engine: GaussDBEngine,
) -> None:
    store = GaussDBVectorStore(
        engine=writable_engine,
        embedding=DeterministicEmbeddings(dimension=3),
        table_name="schema_contract_only",
        embedding_dimension=3,
    )

    assert store._schema_name == "public"


def test_each_mode_prepares_its_indexes_and_queries_the_same_table(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
    requires_bm25: bool,
) -> None:
    dense_store = _store(writable_engine, temporary_vector_table, "dense")
    assert dense_store.add_texts(["lifecycle lexicalneedle"], ids=["shared-id"]) == [
        "shared-id"
    ]
    dense_store.setup()

    dense_documents = dense_store.as_retriever(search_kwargs={"k": 1}).invoke(
        "lifecycle"
    )
    bm25_store = _store(writable_engine, temporary_vector_table, "bm25")
    hybrid_store = _store(writable_engine, temporary_vector_table, "hybrid")
    bm25_store.setup()
    hybrid_store.setup()
    bm25_documents = bm25_store.as_retriever(search_kwargs={"k": 1}).invoke(
        "lexicalneedle"
    )
    hybrid_documents = hybrid_store.as_retriever(search_kwargs={"k": 1}).invoke(
        "lexicalneedle"
    )

    assert dense_documents[0].id == "shared-id"
    assert bm25_documents[0].id == "shared-id"
    assert hybrid_documents[0].id == "shared-id"


def test_query_only_store_does_not_need_setup_when_objects_are_prepared(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    writer = _store(writable_engine, temporary_vector_table, "dense")
    writer.add_texts(["prepared"], ids=["prepared-id"])

    reader = _store(writable_engine, temporary_vector_table, "dense")

    assert reader._initialized is False
    assert reader.similarity_search("prepared", k=1)[0].id == "prepared-id"
    assert reader._initialized is False


def test_setup_positive_cache_does_not_detect_external_ddl(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _store(writable_engine, temporary_vector_table, "dense")
    store.add_texts(["cache contract"], ids=["cache-id"])
    index_name, _ = build_create_vector_index(
        temporary_vector_table.schema,
        temporary_vector_table.name,
        "embedding",
        3,
    )
    writable_engine.execute(
        CompiledSQL(
            sql.SQL("DROP INDEX {}.{}").format(
                sql.Identifier(temporary_vector_table.schema),
                sql.Identifier(index_name),
            )
        ),
        operation="drop vector index outside VectorStore",
    )

    store.setup()
    rows = writable_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT 1 FROM pg_class AS idx "
                "JOIN pg_namespace AS ns ON ns.oid = idx.relnamespace "
                "WHERE ns.nspname = %s AND idx.relname = %s"
            ),
            (temporary_vector_table.schema, index_name),
        ),
        operation="verify setup positive cache does not recreate external DDL",
    )

    assert rows == []
    assert store.similarity_search("cache contract", k=1)[0].id == "cache-id"
