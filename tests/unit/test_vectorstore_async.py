from __future__ import annotations

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

import langchain_gaussdb.vectorstore as vectorstore_module
from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings, RecordingEngine

SCHEMA_ROWS = [("id",), ("content",), ("metadata",), ("embedding",)]


def _store(
    *, engine: RecordingEngine | None = None
) -> tuple[GaussDBVectorStore, RecordingEngine, DeterministicEmbeddings]:
    engine = engine or RecordingEngine()
    embedding = DeterministicEmbeddings()
    store = GaussDBVectorStore(
        embedding=embedding,
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
    )
    store._initialized = True
    return store, engine, embedding


@pytest.mark.parametrize(
    "name",
    [
        "aadd_texts",
        "aadd_documents",
        "adelete",
        "aget_by_ids",
        "asimilarity_search",
        "asimilarity_search_by_vector",
        "asimilarity_search_with_score",
        "asimilarity_search_with_relevance_scores",
        "amax_marginal_relevance_search",
        "amax_marginal_relevance_search_by_vector",
        "afrom_texts",
        "afrom_documents",
    ],
)
def test_standard_async_methods_are_inherited_from_langchain(name: str) -> None:
    assert name not in GaussDBVectorStore.__dict__
    assert inspect.getattr_static(GaussDBVectorStore, name) is inspect.getattr_static(
        VectorStore, name
    )


def test_aadd_texts_runs_the_synchronous_path_in_executor() -> None:
    store, engine, embedding = _store()
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []
    original = store.add_texts

    def record_sync_call(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    store.add_texts = record_sync_call  # type: ignore[method-assign]
    ids = asyncio.run(
        store.aadd_texts(
            ["alpha"],
            metadatas=[{"source": "executor"}],
            ids=["doc-1"],
        )
    )

    assert ids == ["doc-1"]
    assert worker_threads and worker_threads[0] != caller_thread
    assert embedding.document_calls == [["alpha"]]
    assert embedding.async_document_calls == []
    assert [operation for _compiled, operation in engine.executed] == [
        "add vectorstore texts"
    ]


def test_aadd_documents_preserves_document_and_explicit_ids() -> None:
    store, engine, _embedding = _store()
    document = Document(
        id="document-id",
        page_content="alpha",
        metadata={"source": "document"},
    )

    ids = asyncio.run(store.aadd_documents([document], ids=["explicit-id"]))

    assert ids == ["explicit-id"]
    assert document.id == "document-id"
    assert engine.executed[0][0].params[0] == "explicit-id"


def test_adelete_and_aget_by_ids_delegate_to_sync_methods() -> None:
    store, engine, _embedding = _store(
        engine=RecordingEngine(
            fetch_results=[[("doc-1", "alpha", {"source": "executor"})]]
        )
    )

    deleted = asyncio.run(store.adelete(ids=["doc-1", "missing"]))
    documents = asyncio.run(store.aget_by_ids(["doc-1", "missing"]))

    assert deleted is True
    assert [document.id for document in documents] == ["doc-1"]
    assert [operation for _compiled, operation in engine.executed] == [
        "delete vectorstore documents by ids"
    ]
    assert [operation for _compiled, operation in engine.fetched] == [
        "get vectorstore documents by ids"
    ]


def test_adelete_requires_explicit_delete_all() -> None:
    store, engine, _embedding = _store()

    with pytest.raises(ValueError, match="delete_all"):
        asyncio.run(store.adelete(ids=None))

    assert engine.executed == []
    assert asyncio.run(store.adelete(ids=None, delete_all=True)) is True
    assert engine.executed[-1][1] == "delete all vectorstore documents"


def test_async_factory_uses_sync_factory_and_sync_embedding() -> None:
    engine = RecordingEngine(fetch_results=[[], SCHEMA_ROWS])
    embedding = DeterministicEmbeddings()

    store = asyncio.run(
        GaussDBVectorStore.afrom_texts(
            ["alpha"],
            embedding=embedding,
            metadatas=[{"source": "factory"}],
            ids=["doc-1"],
            table_name="documents",
            embedding_dimension=3,
            engine=engine,
        )
    )

    assert isinstance(store, GaussDBVectorStore)
    assert embedding.document_calls == [["alpha"]]
    assert embedding.async_document_calls == []
    assert engine.calls == [
        ("fetch", "check vectorstore required columns"),
        ("execute", "create vectorstore table"),
        ("fetch", "check vectorstore required columns"),
        ("execute", "create gsdiskann vector index"),
        ("execute", "add vectorstore texts"),
    ]


def test_afrom_documents_preserves_document_ids() -> None:
    engine = RecordingEngine(fetch_results=[[], SCHEMA_ROWS])
    document = Document(id="doc-id", page_content="alpha")

    store = asyncio.run(
        GaussDBVectorStore.afrom_documents(
            [document],
            embedding=DeterministicEmbeddings(),
            table_name="documents",
            embedding_dimension=3,
            engine=engine,
        )
    )

    assert isinstance(store, GaussDBVectorStore)
    assert document.id == "doc-id"
    assert engine.executed[-1][0].params[0] == "doc-id"


def test_async_factory_closes_owned_engine_when_sync_factory_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[RecordingEngine] = []

    class FakeEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__(fetch_results=[[]])
            self.fail_execute_operations.add("create vectorstore table")
            created.append(self)

    monkeypatch.setattr(vectorstore_module, "GaussDBEngine", FakeEngine)

    with pytest.raises(RuntimeError, match="create vectorstore table"):
        asyncio.run(
            GaussDBVectorStore.afrom_texts(
                ["alpha"],
                embedding=DeterministicEmbeddings(),
                table_name="documents",
                embedding_dimension=3,
                dsn="host=127.0.0.1 dbname=unit",
            )
        )

    assert len(created) == 1
    assert created[0].closed is True


def test_executor_fallback_does_not_block_event_loop() -> None:
    store, _engine, _embedding = _store()
    entered = threading.Event()
    release = threading.Event()

    def slow_similarity_search(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return []

    store.similarity_search = slow_similarity_search  # type: ignore[method-assign]

    async def run() -> None:
        task = asyncio.create_task(store.asimilarity_search("query"))
        await asyncio.to_thread(entered.wait, 1)
        heartbeat = 0
        for _ in range(3):
            await asyncio.sleep(0)
            heartbeat += 1
        release.set()
        assert await task == []
        assert heartbeat == 3

    asyncio.run(run())


def test_multiple_async_calls_can_use_distinct_executor_workers() -> None:
    store, _engine, _embedding = _store()
    barrier = threading.Barrier(2)
    worker_threads: set[int] = set()

    def concurrent_search(*args, **kwargs):
        worker_threads.add(threading.get_ident())
        barrier.wait(timeout=2)
        return []

    store.similarity_search = concurrent_search  # type: ignore[method-assign]

    async def run() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        await asyncio.gather(
            store.asimilarity_search("one"),
            store.asimilarity_search("two"),
        )

    asyncio.run(run())
    assert len(worker_threads) == 2


def test_vectorstore_does_not_expose_nonstandard_asetup() -> None:
    store, _engine, _embedding = _store()
    assert not hasattr(store, "asetup")
