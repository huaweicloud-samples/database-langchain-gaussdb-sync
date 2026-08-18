from __future__ import annotations

import pytest

import langchain_gaussdb.vectorstore as vectorstore_module
from langchain_gaussdb import BM25Config, GaussDBVectorStore
from langchain_gaussdb.errors import (
    GaussDBConnectionError,
    GaussDBSQLBuildError,
)
from tests.helpers.vectorstore_fakes import (
    DeterministicEmbeddings,
    RecordingEngine,
)


def test_constructor_accepts_external_engine_without_sql():
    embeddings = DeterministicEmbeddings()
    engine = RecordingEngine()

    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )

    assert store.embeddings is embeddings
    assert engine.executed == []
    assert engine.fetched == []


def test_constructor_builds_engine_from_dsn_source(monkeypatch):
    created_engines = []

    class FakeGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            created_engines.append(self)

    monkeypatch.setattr(vectorstore_module, "GaussDBEngine", FakeGaussDBEngine)

    dsn = "host=127.0.0.1 dbname=unit"
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        dsn=dsn,
    )

    assert store.embeddings is not None
    assert len(created_engines) == 1
    assert created_engines[0].kwargs == {
        "dsn": dsn,
        "connection_kwargs": None,
    }
    assert created_engines[0].executed == []
    assert created_engines[0].fetched == []


@pytest.mark.parametrize(
    "source_kwargs",
    [
        {"dsn": "host=127.0.0.1 dbname=unit"},
        {"connection_kwargs": {"host": "127.0.0.1", "dbname": "unit"}},
    ],
)
def test_constructor_rejects_engine_with_other_connection_sources(source_kwargs):
    with pytest.raises(GaussDBConnectionError, match="engine"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=RecordingEngine(),
            **source_kwargs,
        )


def test_constructor_rejects_missing_connection_source():
    with pytest.raises(GaussDBConnectionError, match="connection source"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
        )


@pytest.mark.parametrize("removed_argument", ["pool", "connection"])
def test_constructor_rejects_removed_connection_injection(removed_argument):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            **{removed_argument: object()},
        )


@pytest.mark.parametrize("dimension", [0, -1, "3", 1.5, True, None])
def test_constructor_rejects_invalid_embedding_dimension(dimension):
    with pytest.raises(ValueError, match="embedding_dimension"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=dimension,
            engine=RecordingEngine(),
        )


def test_constructor_rejects_dotted_table_name():
    with pytest.raises(GaussDBSQLBuildError, match="single identifier"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="public.docs",
            embedding_dimension=3,
            engine=RecordingEngine(),
        )


@pytest.mark.parametrize(
    "duplicate_columns",
    [
        {"content_column": "id"},
        {"metadata_column": "id"},
        {"embedding_column": "id"},
        {"metadata_column": "content"},
        {"embedding_column": "content"},
        {"embedding_column": "metadata"},
    ],
)
def test_constructor_rejects_duplicate_core_columns(duplicate_columns):
    engine = RecordingEngine()

    with pytest.raises(ValueError, match="core columns must be distinct"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=engine,
            **duplicate_columns,
        )

    assert engine.executed == []
    assert engine.fetched == []


@pytest.mark.parametrize("column", ["metadata", "embedding"])
def test_constructor_rejects_bm25_on_non_text_core_column(column):
    with pytest.raises(GaussDBSQLBuildError, match="text column"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=RecordingEngine(),
            retrieval_mode="bm25",
            bm25_config=BM25Config(column=column),
        )


def test_constructor_reports_invalid_bm25_identifier_before_reserved_name_use():
    with pytest.raises(GaussDBSQLBuildError, match="bm25_config.column"):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=RecordingEngine(),
            bm25_config=BM25Config(column=[]),  # type: ignore[arg-type]
        )


def test_embeddings_property_returns_embedding_object():
    embeddings = DeterministicEmbeddings()
    store = GaussDBVectorStore(
        embedding=embeddings,
        table_name="documents",
        embedding_dimension=3,
        engine=RecordingEngine(),
    )

    assert store.embeddings is embeddings


def test_constructor_accepts_metadata_expression_indexes_without_sql():
    engine = RecordingEngine()

    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes={
            "tenant_id": "text",
            "event_date": "date",
            "exact_value": None,
        },
    )

    assert store.metadata_indexes == {
        "tenant_id": "text",
        "event_date": "date",
        "exact_value": None,
    }
    assert engine.executed == []
    assert engine.fetched == []


def test_constructor_accepts_arbitrary_json_keys_for_metadata_indexes_without_sql():
    engine = RecordingEngine()
    metadata_indexes = {
        "元 数据": "text",
        "索引列": "date",
        'Meta"Value': "bigint",
        "bad-name": "text",
        "1source": "text",
        "source.name": None,
    }

    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        metadata_indexes=metadata_indexes,
    )

    assert store.metadata_indexes == metadata_indexes
    assert engine.executed == []
    assert engine.fetched == []


def test_constructor_normalizes_metadata_index_cast_aliases():
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=RecordingEngine(),
        metadata_indexes={"priority": "integer", "score": "double precision"},
    )

    assert store.metadata_indexes == {
        "priority": "bigint",
        "score": "float",
    }


@pytest.mark.parametrize(
    "metadata_indexes, expected",
    [
        ({"": "text"}, "metadata index key"),
        ({"bad\0name": "text"}, "metadata index key"),
        ({"$operator": "text"}, "metadata index key"),
        ({"tenant_id": "jsonb"}, "metadata index cast"),
        ({"created_at": "timestamp"}, "metadata index cast"),
        ({"start_time": "time"}, "metadata index cast"),
        ({"tenant_id": 1}, "metadata index cast"),
    ],
)
def test_constructor_rejects_invalid_metadata_indexes(metadata_indexes, expected):
    with pytest.raises(ValueError, match=expected):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=RecordingEngine(),
            metadata_indexes=metadata_indexes,
        )


def test_close_closes_owned_engine_only(monkeypatch):
    external_engine = RecordingEngine()
    external_store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=external_engine,
    )

    external_store.close()

    assert external_engine.closed is False

    created_engines = []

    class FakeGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            created_engines.append(self)

    monkeypatch.setattr(vectorstore_module, "GaussDBEngine", FakeGaussDBEngine)
    owned_store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        dsn="host=127.0.0.1 dbname=unit",
    )

    owned_store.close()

    assert created_engines[0].closed is True
