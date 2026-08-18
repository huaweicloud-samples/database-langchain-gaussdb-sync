from __future__ import annotations

import pytest

from langchain_gaussdb.vectorstore import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings, RecordingEngine


def _store(*, metadata_indexes=None) -> tuple[GaussDBVectorStore, RecordingEngine]:
    engine = RecordingEngine()
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        schema_name="public",
        embedding_dimension=3,
        metadata_indexes=metadata_indexes,
        engine=engine,
    )
    return store, engine


def _statement(engine: RecordingEngine, index: int = -1) -> str:
    return repr(engine.executed[index][0].statement)


def test_internal_index_setup_executes_complete_idempotent_ddl() -> None:
    store, engine = _store(metadata_indexes={"tenant_id": "text"})
    store._prepare_table_if_needed = lambda: None

    store.setup()

    assert len(engine.executed) == 2
    assert "CREATE INDEX IF NOT EXISTS" in _statement(engine, 0)
    assert "tenant_id" in _statement(engine, 0)
    assert "USING gsdiskann" in _statement(engine, 1)
    assert engine.fetched == []


@pytest.mark.parametrize(
    ("mode", "expected_methods"),
    [
        ("bm25", ["USING bm25"]),
        ("hybrid", ["USING gsdiskann", "USING bm25"]),
    ],
)
def test_internal_index_setup_matches_retrieval_mode(
    mode: str, expected_methods: list[str]
) -> None:
    engine = RecordingEngine(fetch_results=[[(False,)]])
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        schema_name="public",
        embedding_dimension=3,
        retrieval_mode=mode,
        engine=engine,
    )
    store._prepare_table_if_needed = lambda: None

    store.setup()

    assert len(engine.executed) == len(expected_methods)
    for index, expected_method in enumerate(expected_methods):
        assert expected_method in _statement(engine, index)


@pytest.mark.parametrize(
    "name",
    [
        "create_vector_index",
        "create_metadata_index",
        "create_bm25_index",
        "create_ugin_trgm_index",
        "drop_index",
        "reindex",
        "drop_table",
        "supports_index_access_method",
        "supports_index_opclass",
    ],
)
def test_database_administration_helpers_are_not_public(name: str) -> None:
    store, _engine = _store()

    assert not hasattr(store, name)
