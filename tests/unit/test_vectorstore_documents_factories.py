from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document

import langchain_gaussdb.vectorstore as vectorstore_module
from langchain_gaussdb import BM25Config, GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings, RecordingEngine

SCHEMA_ROWS = [("id",), ("content",), ("metadata",), ("embedding",)]


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
    store._initialized = True
    return store, embeddings, engine


def test_add_documents_id_precedence_and_no_input_mutation() -> None:
    document = Document(
        id="document-id",
        page_content="alpha",
        metadata={"source": "document"},
    )
    store, _embeddings, engine = _store()

    ids = store.add_documents([document], ids=["explicit-id"])

    assert ids == ["explicit-id"]
    assert document.id == "document-id"
    assert engine.executed[0][0].params[0] == "explicit-id"


def test_add_documents_uses_document_id_or_generates_uuid() -> None:
    store, _embeddings, engine = _store()
    documents = [
        Document(id="document-id", page_content="alpha"),
        Document(page_content="beta"),
    ]

    ids = store.add_documents(iter(documents))

    assert ids[0] == "document-id"
    assert str(uuid.UUID(ids[1])) == ids[1]
    assert documents[1].id is None
    assert engine.executed[0][0].params[::4] == tuple(ids)


def test_factory_always_uses_public_write_and_complete_initialization() -> None:
    events: list[str] = []

    class ProbeStore(GaussDBVectorStore):
        def _ensure_initialized(self) -> None:
            events.append("initialize")

        def add_texts(self, texts, **kwargs):
            events.append("add_texts")
            return super().add_texts(texts, **kwargs)

    store = ProbeStore.from_texts(
        ["alpha"],
        embedding=DeterministicEmbeddings(),
        ids=["doc-1"],
        table_name="documents",
        embedding_dimension=3,
        engine=RecordingEngine(),
    )

    assert isinstance(store, ProbeStore)
    assert events == ["add_texts", "initialize"]


@pytest.mark.parametrize("option", ["create_table", "create_index"])
def test_removed_factory_switches_fail_before_construction(option: str) -> None:
    class ConstructionProbeStore(GaussDBVectorStore):
        construction_count = 0

        def __init__(self, **kwargs) -> None:
            type(self).construction_count += 1
            super().__init__(**kwargs)

    with pytest.raises(ValueError, match=option):
        ConstructionProbeStore.from_texts(
            ["alpha"],
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=RecordingEngine(),
            **{option: True},
        )

    assert ConstructionProbeStore.construction_count == 0


def test_from_texts_initializes_table_indexes_then_writes() -> None:
    engine = RecordingEngine(fetch_results=[[], SCHEMA_ROWS])

    store = GaussDBVectorStore.from_texts(
        ["alpha"],
        embedding=DeterministicEmbeddings(),
        metadatas=[{"source": "factory"}],
        ids=["doc-1"],
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )

    assert isinstance(store, GaussDBVectorStore)
    assert [operation for _compiled, operation in engine.executed] == [
        "create vectorstore table",
        "create gsdiskann vector index",
        "add vectorstore texts",
    ]
    assert engine.executed[-1][0].params[:3] == (
        "doc-1",
        "alpha",
        '{"source":"factory"}',
    )


def test_from_documents_preserves_document_ids_and_explicit_override() -> None:
    engine = RecordingEngine(fetch_results=[SCHEMA_ROWS])
    document = Document(id="document-id", page_content="alpha")

    store = GaussDBVectorStore.from_documents(
        [document],
        embedding=DeterministicEmbeddings(),
        ids=["explicit-id"],
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )

    assert isinstance(store, GaussDBVectorStore)
    assert document.id == "document-id"
    assert engine.executed[-1][0].params[0] == "explicit-id"


def test_from_texts_closes_owned_engine_when_initialization_fails(monkeypatch) -> None:
    created_engines = []

    class FakeGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__(fetch_results=[[]])
            self.fail_execute_operations.add("create vectorstore table")
            created_engines.append(self)

    monkeypatch.setattr(vectorstore_module, "GaussDBEngine", FakeGaussDBEngine)

    with pytest.raises(RuntimeError, match="create vectorstore table"):
        GaussDBVectorStore.from_texts(
            ["alpha"],
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            dsn="host=127.0.0.1 dbname=unit",
        )

    assert created_engines[0].closed is True


def test_factory_failure_does_not_close_external_engine() -> None:
    engine = RecordingEngine(fetch_results=[SCHEMA_ROWS])
    engine.fail_execute_operations.add("add vectorstore texts")

    with pytest.raises(RuntimeError, match="add vectorstore texts"):
        GaussDBVectorStore.from_texts(
            ["alpha"],
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=engine,
        )

    assert engine.closed is False


@pytest.mark.parametrize("bad_kwarg", ["drop_old", "filter"])
def test_later_storage_kwargs_fail_fast_in_factories(bad_kwarg: str) -> None:
    with pytest.raises(ValueError, match=bad_kwarg):
        GaussDBVectorStore.from_texts(
            ["alpha"],
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=RecordingEngine(),
            **{bad_kwarg: object()},
        )


def test_from_texts_passes_lexical_projection_values_to_write() -> None:
    engine = RecordingEngine(fetch_results=[[*SCHEMA_ROWS, ("text_lemmatized",)]])

    store = GaussDBVectorStore.from_texts(
        ["original"],
        embedding=DeterministicEmbeddings(),
        ids=["doc-1"],
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        bm25_config=BM25Config(column="text_lemmatized"),
        text_lemmatized_values=["processed"],
    )

    assert store.bm25_config.column == "text_lemmatized"
    assert "processed" in engine.executed[-1][0].params


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_factory_accepts_retrieval_mode_constructor_option(mode: str) -> None:
    engine = RecordingEngine(fetch_results=[[(False,)], SCHEMA_ROWS])

    store = GaussDBVectorStore.from_documents(
        [Document(page_content="alpha")],
        embedding=DeterministicEmbeddings(),
        ids=["doc-1"],
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        retrieval_mode=mode,
    )

    assert store.retrieval_mode == mode
    assert engine.executed[-1][1] == "add vectorstore texts"
