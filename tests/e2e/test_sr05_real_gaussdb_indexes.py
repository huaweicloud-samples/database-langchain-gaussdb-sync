from __future__ import annotations

from typing import Any

import pytest
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.indexes import (
    build_create_metadata_index,
    build_create_vector_index,
)
from langchain_gaussdb.sql import CompiledSQL
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings

pytestmark = [
    pytest.mark.gaussdb_e2e,
    pytest.mark.gaussdb_e2e_fast,
    pytest.mark.gaussdb_e2e_full,
]


def _catalog_names(engine: GaussDBEngine, table: Any) -> set[str]:
    rows = engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = %s AND tablename = %s"
            ),
            (table.schema, table.name),
        ),
        operation="inspect setup-created indexes",
    )
    return {str(row[0]) for row in rows}


def test_dense_setup_creates_only_its_required_internal_indexes(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = GaussDBVectorStore(
        engine=writable_engine,
        embedding=DeterministicEmbeddings(dimension=3),
        schema_name=temporary_vector_table.schema,
        table_name=temporary_vector_table.name,
        embedding_dimension=3,
        metadata_indexes={"tenant": "text"},
    )

    store.setup()
    first_names = _catalog_names(writable_engine, temporary_vector_table)
    store.setup()
    second_names = _catalog_names(writable_engine, temporary_vector_table)

    vector_name, _ = build_create_vector_index(
        temporary_vector_table.schema,
        temporary_vector_table.name,
        "embedding",
        3,
    )
    metadata_name, _ = build_create_metadata_index(
        temporary_vector_table.schema,
        temporary_vector_table.name,
        "metadata",
        "tenant",
        cast="text",
    )
    assert {vector_name, metadata_name} <= first_names
    assert not any("bm25" in name.lower() for name in first_names)
    assert second_names == first_names


def test_vectorstore_does_not_expose_database_administration_methods(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
) -> None:
    store = GaussDBVectorStore(
        engine=writable_engine,
        embedding=DeterministicEmbeddings(dimension=3),
        schema_name=temporary_vector_table.schema,
        table_name=temporary_vector_table.name,
        embedding_dimension=3,
    )

    assert not hasattr(store, "create_vector_index")
    assert not hasattr(store, "drop_index")
    assert not hasattr(store, "reindex")
    assert not hasattr(store, "drop_table")
