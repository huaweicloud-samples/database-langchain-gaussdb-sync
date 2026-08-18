from __future__ import annotations

from typing import Any

import pytest

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings

pytestmark = [pytest.mark.gaussdb_e2e, pytest.mark.gaussdb_e2e_fast]


def _store(engine: GaussDBEngine, table: Any) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        engine=engine,
        embedding=DeterministicEmbeddings(dimension=3),
        schema_name=table.schema,
        table_name=table.name,
        embedding_dimension=3,
    )


def test_setup_write_read_and_delete_contract(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _store(writable_engine, temporary_vector_table)

    store.setup()
    assert store.add_texts(["alpha"], ids=["doc-1"]) == ["doc-1"]
    assert [document.id for document in store.get_by_ids(["doc-1", "missing"])] == [
        "doc-1"
    ]
    assert store.delete(ids=["doc-1"])
    assert store.get_by_ids(["doc-1"]) == []


def test_factory_uses_the_same_automatic_initialization_contract(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = GaussDBVectorStore.from_texts(
        ["factory"],
        embedding=DeterministicEmbeddings(dimension=3),
        ids=["factory-id"],
        engine=writable_engine,
        schema_name=temporary_vector_table.schema,
        table_name=temporary_vector_table.name,
        embedding_dimension=3,
    )

    assert [document.id for document in store.get_by_ids(["factory-id"])] == [
        "factory-id"
    ]


@pytest.mark.parametrize("removed_flag", ["create_table", "create_index"])
def test_factory_rejects_removed_lifecycle_switches(
    removed_flag: str,
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
) -> None:
    with pytest.raises(ValueError, match=removed_flag):
        GaussDBVectorStore.from_texts(
            ["factory"],
            embedding=DeterministicEmbeddings(dimension=3),
            engine=writable_engine,
            schema_name=temporary_vector_table.schema,
            table_name=temporary_vector_table.name,
            embedding_dimension=3,
            **{removed_flag: True},
        )
