from __future__ import annotations

import datetime as dt
import math
import uuid

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
    embeddings: DeterministicEmbeddings | None = None,
    engine: RecordingEngine | None = None,
) -> tuple[GaussDBVectorStore, DeterministicEmbeddings, RecordingEngine]:
    embeddings = embeddings or DeterministicEmbeddings()
    engine = engine or RecordingEngine()
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )
    mark_vectorstore_initialized(store)
    return store, embeddings, engine


def _statement_repr(engine: RecordingEngine) -> str:
    compiled, _operation = engine.executed[0]
    return repr(compiled.statement)


def test_add_texts_writes_rows_and_returns_ids():
    store, _embeddings, engine = _store()

    ids = store.add_texts(
        ["alpha"],
        metadatas=[{"source": "unit"}],
        ids=["doc-1"],
    )

    assert ids == ["doc-1"]
    assert len(engine.executed) == 1
    compiled, operation = engine.executed[0]
    assert operation == "add vectorstore texts"
    assert compiled.params == (
        "doc-1",
        "alpha",
        '{"source":"unit"}',
        "[0.0,1.0,2.0]",
    )


def test_add_texts_turns_generator_into_stable_list_once():
    class SingleUseTexts:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("texts iterable was consumed more than once")
            yield "alpha"
            yield "beta"

    texts = SingleUseTexts()
    store, embeddings, engine = _store()

    ids = store.add_texts(texts, ids=["doc-1", "doc-2"])

    assert ids == ["doc-1", "doc-2"]
    assert texts.iterations == 1
    assert embeddings.document_calls == [["alpha", "beta"]]
    compiled, _operation = engine.executed[0]
    assert compiled.params == (
        "doc-1",
        "alpha",
        "{}",
        "[0.0,1.0,2.0]",
        "doc-2",
        "beta",
        "{}",
        "[1.0,2.0,3.0]",
    )


def test_add_texts_generates_uuid_ids_when_missing():
    store, _embeddings, engine = _store()

    ids = store.add_texts(["alpha", "beta"])

    assert len(ids) == 2
    assert [str(uuid.UUID(value)) for value in ids] == ids
    compiled, _operation = engine.executed[0]
    assert compiled.params[0] == ids[0]
    assert compiled.params[4] == ids[1]


def test_add_texts_returns_empty_list_without_embedding_or_sql():
    store, embeddings, engine = _store()

    ids = store.add_texts([])

    assert ids == []
    assert embeddings.document_calls == []
    assert engine.executed == []


@pytest.mark.parametrize("texts", ["abc", b"abc"])
def test_add_texts_rejects_string_like_texts(texts):
    store, embeddings, engine = _store()

    with pytest.raises(ValueError, match="texts"):
        store.add_texts(texts, ids=["doc-1", "doc-2", "doc-3"])

    assert embeddings.document_calls == []
    assert engine.executed == []


def test_add_texts_empty_texts_still_validate_ids_length():
    store, _embeddings, _engine = _store()

    with pytest.raises(ValueError, match="ids"):
        store.add_texts([], ids=["doc-1"])


def test_add_texts_empty_texts_still_validate_metadata_length():
    store, _embeddings, _engine = _store()

    with pytest.raises(ValueError, match="metadatas"):
        store.add_texts([], metadatas=[{}])


def test_add_texts_rejects_metadata_length_mismatch():
    store, _embeddings, _engine = _store()

    with pytest.raises(ValueError, match="metadatas"):
        store.add_texts(["alpha", "beta"], metadatas=[{}])


def test_add_texts_rejects_ids_length_mismatch():
    store, _embeddings, _engine = _store()

    with pytest.raises(ValueError, match="ids"):
        store.add_texts(["alpha", "beta"], ids=["doc-1"])


@pytest.mark.parametrize("ids", ["abc", b"abc"])
def test_add_texts_rejects_string_like_ids(ids):
    store, embeddings, engine = _store()

    with pytest.raises(ValueError, match="ids"):
        store.add_texts(["alpha", "beta", "gamma"], ids=ids)

    assert embeddings.document_calls == []
    assert engine.executed == []


def test_add_texts_allows_database_to_resolve_duplicate_batch_ids():
    store, _embeddings, engine = _store()

    ids = store.add_texts(["alpha", "beta"], ids=["doc-1", "doc-1"])

    assert ids == ["doc-1", "doc-1"]
    compiled, _operation = engine.executed[0]
    assert compiled.params[0] == "doc-1"
    assert compiled.params[4] == "doc-1"


def test_add_texts_splits_large_insert_statements(monkeypatch):
    monkeypatch.setattr(vectorstore_module, "_WRITE_BATCH_SIZE", 2)
    store, embeddings, engine = _store()

    ids = store.add_texts(
        ["alpha", "beta", "gamma"],
        ids=["doc-1", "doc-2", "doc-3"],
    )

    assert ids == ["doc-1", "doc-2", "doc-3"]
    assert embeddings.document_calls == [["alpha", "beta", "gamma"]]
    assert [len(compiled.params) for compiled, _operation in engine.executed] == [8, 4]
    assert engine.executed[1][0].params[0] == "doc-3"


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_add_texts_rejects_empty_id_value(bad_id):
    store, _embeddings, _engine = _store()

    with pytest.raises(ValueError, match="empty"):
        store.add_texts(["alpha"], ids=[bad_id])


def test_add_texts_rejects_embedding_count_mismatch():
    store, _embeddings, _engine = _store(
        embeddings=DeterministicEmbeddings(vectors=[[0.0, 1.0, 2.0]])
    )

    with pytest.raises(ValueError, match="embedding"):
        store.add_texts(["alpha", "beta"], ids=["doc-1", "doc-2"])


def test_add_texts_rejects_embedding_dimension_mismatch():
    store, _embeddings, _engine = _store(
        embeddings=DeterministicEmbeddings(vectors=[[0.0, 1.0]])
    )

    with pytest.raises(ValueError, match="dimension"):
        store.add_texts(["alpha"], ids=["doc-1"])


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_add_texts_rejects_non_finite_embedding_value(bad_value):
    store, _embeddings, _engine = _store(
        embeddings=DeterministicEmbeddings(vectors=[[0.0, bad_value, 2.0]])
    )

    with pytest.raises(ValueError, match="finite"):
        store.add_texts(["alpha"], ids=["doc-1"])


def test_add_texts_rejects_unserializable_metadata():
    store, _embeddings, _engine = _store()
    secret_value = "super-sensitive-full-metadata-value"

    with pytest.raises(ValueError) as exc_info:
        store.add_texts(
            ["alpha"],
            metadatas=[{"secret": secret_value, "bad": object()}],
            ids=["doc-1"],
        )

    message = str(exc_info.value)
    assert "metadata" in message
    assert "0" in message
    assert secret_value not in message


def test_add_texts_rejects_metadata_with_non_finite_number():
    store, _embeddings, _engine = _store()
    secret_value = "super-sensitive-full-metadata-value"

    with pytest.raises(ValueError) as exc_info:
        store.add_texts(
            ["alpha"],
            metadatas=[{"secret": secret_value, "score": math.nan}],
            ids=["doc-1"],
        )

    message = str(exc_info.value)
    assert "metadata" in message
    assert "0" in message
    assert secret_value not in message


def test_add_texts_serializes_temporal_metadata_values_to_json_iso_strings():
    store, _embeddings, engine = _store()

    store.add_texts(
        ["alpha"],
        metadatas=[
            {
                "event_date": dt.date(2026, 7, 1),
                "created_at": dt.datetime(2026, 7, 1, 12, 30, 45),
                "start_time": dt.time(9, 30, 0),
            }
        ],
        ids=["doc-1"],
    )

    compiled, _operation = engine.executed[0]
    assert compiled.params[2] == (
        '{"event_date":"2026-07-01",'
        '"created_at":"2026-07-01T12:30:45",'
        '"start_time":"09:30:00"}'
    )


def test_add_texts_rejects_non_dict_metadata():
    store, _embeddings, _engine = _store()

    with pytest.raises(ValueError, match="metadata"):
        store.add_texts(["alpha"], metadatas=[["not", "a", "dict"]], ids=["doc-1"])


def test_add_texts_uses_odku_not_on_conflict_or_merge():
    store, _embeddings, engine = _store()

    store.add_texts(["alpha"], ids=["doc-1"])

    statement = _statement_repr(engine)
    assert "ON DUPLICATE KEY UPDATE" in statement
    assert "ON CONFLICT" not in statement
    assert "MERGE" not in statement


def test_add_texts_binds_content_metadata_and_vector_as_params():
    malicious_id = "doc-1'); DROP TABLE documents; --"
    malicious_content = "x'); DROP TABLE documents; --"
    malicious_metadata = {"note": "y'); DELETE FROM documents; --"}
    store, _embeddings, engine = _store(
        embeddings=DeterministicEmbeddings(vectors=[[0.1, 0.2, 3.0]])
    )

    store.add_texts(
        [malicious_content],
        metadatas=[malicious_metadata],
        ids=[malicious_id],
    )

    compiled, _operation = engine.executed[0]
    metadata_json = '{"note":"y\'); DELETE FROM documents; --"}'
    assert compiled.params == (
        malicious_id,
        malicious_content,
        metadata_json,
        "[0.1,0.2,3.0]",
    )
    statement = repr(compiled.statement)
    assert malicious_id not in statement
    assert malicious_content not in statement
    assert metadata_json not in statement
    assert "[0.1,0.2,3.0]" not in statement
    assert "DROP TABLE" not in statement


def test_add_texts_stores_metadata_once_as_jsonb_with_metadata_indexes():
    engine = RecordingEngine()
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(vectors=[[0.1, 0.2, 0.3]]),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"tenant_id": "text", "score": "float"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(
        ["alpha"],
        metadatas=[{"tenant_id": "t1", "score": 0.8, "source": "json-only"}],
        ids=["doc-1"],
    )

    compiled, operation = engine.executed[0]
    statement = repr(compiled.statement)
    assert operation == "add vectorstore texts"
    assert "Identifier('tenant_id')" not in statement
    assert "Identifier('score')" not in statement
    assert "VALUES(" in statement
    assert compiled.params == (
        "doc-1",
        "alpha",
        '{"tenant_id":"t1","score":0.8,"source":"json-only"}',
        "[0.1,0.2,0.3]",
    )


def test_add_texts_does_not_apply_index_cast_validation_to_jsonb_writes():
    engine = RecordingEngine()
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"score": "float"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(["alpha"], metadatas=[{"score": "high"}], ids=["doc-1"])

    assert engine.executed[0][0].params[2] == '{"score":"high"}'


def test_add_texts_keeps_numeric_metadata_inside_jsonb():
    engine = RecordingEngine()
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(vectors=[[0.1, 0.2, 0.3]]),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"priority": "bigint"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(
        ["alpha"],
        metadatas=[{"priority": 7}],
        ids=["doc-1"],
    )

    assert store.metadata_indexes == {"priority": "bigint"}
    assert engine.executed[0][0].params[2] == '{"priority":7}'
    assert len(engine.executed[0][0].params) == 4


@pytest.mark.parametrize("invalid_value", [True, 1.5, "7"])
def test_add_texts_allows_json_types_that_do_not_match_index_cast(
    invalid_value,
):
    engine = RecordingEngine()
    embeddings = DeterministicEmbeddings()
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"priority": "bigint"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(
        ["alpha"],
        metadatas=[{"priority": invalid_value}],
        ids=["doc-1"],
    )

    assert embeddings.document_calls == [["alpha"]]
    assert len(engine.executed[0][0].params) == 4


def test_add_texts_keeps_iso_date_string_inside_jsonb():
    engine = RecordingEngine()
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(vectors=[[0.1, 0.2, 0.3]]),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"event_date": "date"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(
        ["alpha"],
        metadatas=[{"event_date": "2026-07-01"}],
        ids=["doc-1"],
    )

    assert engine.executed[0][0].params == (
        "doc-1",
        "alpha",
        '{"event_date":"2026-07-01"}',
        "[0.1,0.2,0.3]",
    )


def test_add_texts_serializes_python_temporal_values_only_inside_jsonb():
    engine = RecordingEngine()
    embeddings = DeterministicEmbeddings(vectors=[[0.1, 0.2, 0.3]])
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"event_date": "date"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(
        ["alpha"],
        metadatas=[
            {
                "event_date": dt.date(2026, 7, 1),
                "created_at": dt.datetime(2026, 7, 1, 12, 30, 45),
                "start_time": dt.time(9, 30, 0),
            }
        ],
        ids=["doc-1"],
    )

    compiled, _operation = engine.executed[0]
    assert compiled.params == (
        "doc-1",
        "alpha",
        '{"event_date":"2026-07-01","created_at":"2026-07-01T12:30:45",'
        '"start_time":"09:30:00"}',
        "[0.1,0.2,0.3]",
    )
    statement = repr(compiled.statement)
    assert "to_date" not in statement
    assert "to_timestamp" not in statement


@pytest.mark.parametrize(
    "timestamp_value",
    [
        "2026-07-01 12:30:45",
        "2026-07-01T12:30:45",
    ],
)
def test_add_texts_preserves_temporal_strings_inside_jsonb(timestamp_value):
    engine = RecordingEngine()
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(vectors=[[0.1, 0.2, 0.3]]),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"event_date": "date"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(
        ["alpha"],
        metadatas=[
            {
                "event_date": "2026-07-01",
                "created_at": timestamp_value,
                "start_time": "09:30:00",
            }
        ],
        ids=["doc-1"],
    )

    assert engine.executed[0][0].params == (
        "doc-1",
        "alpha",
        '{"event_date":"2026-07-01","created_at":'
        f'"{timestamp_value}","start_time":"09:30:00"}}',
        "[0.1,0.2,0.3]",
    )


@pytest.mark.parametrize(
    "value",
    [
        "2026-7-01",
        "2026-02-30",
        "2026-07-01 12:30:45",
        dt.datetime(2026, 7, 1, 12, 30, 45),
        dt.date(2026, 7, 1),
        dt.datetime(
            2026,
            7,
            1,
            12,
            30,
            45,
            tzinfo=dt.timezone(dt.timedelta(hours=8)),
        ),
        dt.time(9, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))),
    ],
)
def test_add_texts_leaves_date_index_cast_compatibility_to_queries(value):
    engine = RecordingEngine()
    embeddings = DeterministicEmbeddings()
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={"temporal_value": "date"},
    )
    mark_vectorstore_initialized(store)

    store.add_texts(
        ["alpha"],
        metadatas=[{"temporal_value": value}],
        ids=["doc-1"],
    )

    assert embeddings.document_calls == [["alpha"]]
    assert len(engine.executed) == 1


def test_embedding_serializer_outputs_gaussdb_vector_string():
    store, _embeddings, engine = _store(
        embeddings=DeterministicEmbeddings(vectors=[[0.1, 0.2, 3.0]])
    )

    store.add_texts(["alpha"], ids=["doc-1"])

    compiled, _operation = engine.executed[0]
    assert compiled.params[-1] == "[0.1,0.2,3.0]"
