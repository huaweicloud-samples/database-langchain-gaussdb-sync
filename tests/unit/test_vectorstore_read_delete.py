from __future__ import annotations

import pytest

from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
)


def _store(
    *,
    engine: RecordingEngine | None = None,
) -> tuple[GaussDBVectorStore, RecordingEngine]:
    engine = engine or RecordingEngine()
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )
    return store, engine


def test_get_by_ids_returns_existing_documents_and_ignores_missing_ids():
    store, engine = _store(
        engine=RecordingEngine(
            fetch_results=[
                [
                    ("doc-1", "alpha", {"source": "unit", "rank": 1}),
                ]
            ]
        )
    )

    documents = store.get_by_ids(["doc-1", "missing"])

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "doc-1"
    assert document.page_content == "alpha"
    assert document.metadata == {"source": "unit", "rank": 1}
    compiled, operation = engine.fetched[0]
    assert operation == "get vectorstore documents by ids"
    assert compiled.params == (["doc-1", "missing"],)
    statement = repr(compiled.statement)
    assert "ANY" in statement
    assert "embedding" not in statement


def test_get_by_ids_returns_empty_for_empty_ids_without_sql():
    store, engine = _store()

    documents = store.get_by_ids([])

    assert documents == []
    assert engine.fetched == []


@pytest.mark.parametrize("ids", ["doc-1", b"doc-1"])
def test_get_by_ids_rejects_string_like_ids(ids):
    store, engine = _store()

    with pytest.raises(ValueError, match="ids"):
        store.get_by_ids(ids)

    assert engine.fetched == []


def test_get_by_ids_parses_json_string_and_none_metadata():
    store, _engine = _store(
        engine=RecordingEngine(
            fetch_results=[
                [
                    ("doc-json", "alpha", '{"tags":["json"],"rank":1}'),
                    ("doc-none", "beta", None),
                ]
            ]
        )
    )

    documents = store.get_by_ids(["doc-json", "doc-none"])

    by_id = {document.id: document for document in documents}
    assert by_id["doc-json"].metadata == {"tags": ["json"], "rank": 1}
    assert by_id["doc-none"].metadata == {}


@pytest.mark.parametrize("metadata", ["{bad", "[]"])
def test_get_by_ids_rejects_invalid_database_metadata(metadata):
    store, _engine = _store(
        engine=RecordingEngine(fetch_results=[[("doc-1", "alpha", metadata)]])
    )

    with pytest.raises(ValueError, match="metadata from database"):
        store.get_by_ids(["doc-1"])


def test_delete_by_ids_uses_any_param_and_returns_true():
    store, engine = _store()
    ids = ["doc-1", "missing"]

    result = store.delete(ids=ids)

    assert result is True
    assert len(engine.executed) == 1
    compiled, operation = engine.executed[0]
    assert operation == "delete vectorstore documents by ids"
    assert compiled.params == (ids,)
    statement = repr(compiled.statement)
    assert "DELETE FROM" in statement
    assert "ANY" in statement


def test_delete_empty_ids_returns_true_without_sql():
    store, engine = _store()

    result = store.delete(ids=[])

    assert result is True
    assert engine.executed == []


@pytest.mark.parametrize("ids", ["doc-1", b"doc-1"])
def test_delete_rejects_string_like_ids_without_sql(ids):
    store, engine = _store()

    with pytest.raises(ValueError, match="ids"):
        store.delete(ids=ids)

    assert engine.executed == []


def test_delete_none_requires_delete_all_flag():
    store, engine = _store()

    with pytest.raises(ValueError, match="delete_all"):
        store.delete(ids=None)

    assert engine.executed == []


@pytest.mark.parametrize("delete_all", ["true", "false", 1])
def test_delete_all_requires_literal_true(delete_all):
    store, engine = _store()

    with pytest.raises(ValueError, match="delete_all"):
        store.delete(ids=None, delete_all=delete_all)

    assert engine.executed == []


def test_delete_rejects_delete_all_when_ids_are_provided():
    store, engine = _store()

    with pytest.raises(ValueError, match="delete_all"):
        store.delete(ids=["doc-1"], delete_all=True)

    assert engine.executed == []


def test_delete_all_requires_explicit_flag_and_has_no_where_clause():
    store, engine = _store()

    result = store.delete(ids=None, delete_all=True)

    assert result is True
    assert len(engine.executed) == 1
    compiled, operation = engine.executed[0]
    assert operation == "delete all vectorstore documents"
    assert compiled.params == ()
    statement = repr(compiled.statement)
    assert "DELETE FROM" in statement
    assert "WHERE" not in statement


def test_delete_filter_is_later_sr_and_fails_fast():
    store, engine = _store()

    with pytest.raises(ValueError) as exc_info:
        store.delete(filter={"source": "unit"})

    message = str(exc_info.value)
    assert "filter" in message
    assert "storage API" in message
    assert engine.executed == []


def test_delete_rejects_unknown_kwargs():
    store, engine = _store()

    with pytest.raises(ValueError, match="unexpected"):
        store.delete(ids=["doc-1"], unexpected=True)

    assert engine.executed == []
