from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from langchain_gaussdb.errors import GaussDBCapabilityError
from langchain_gaussdb.vectorstore import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings, RecordingEngine


def _store(
    *,
    mode: str = "dense",
    metadata_indexes=None,
    embedding_dimension: int = 3,
    distributed: bool | None = None,
) -> GaussDBVectorStore:
    fetch_results = []
    if distributed is not None:
        fetch_results.append([(distributed,)])
    elif mode != "dense" or embedding_dimension > 1024:
        fetch_results.append([(False,)])
    return GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        schema_name="public",
        embedding_dimension=embedding_dimension,
        retrieval_mode=mode,
        metadata_indexes=metadata_indexes,
        engine=RecordingEngine(fetch_results=fetch_results),
    )


def _record_initialization(store: GaussDBVectorStore, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        store, "_prepare_table_if_needed", lambda: calls.append("table")
    )
    monkeypatch.setattr(
        store,
        "_create_metadata_index",
        lambda key, cast=None: calls.append(f"metadata:{key}:{cast}"),
    )
    monkeypatch.setattr(
        store,
        "_create_vector_index",
        lambda: calls.append("vector"),
    )
    monkeypatch.setattr(store, "_create_bm25_index", lambda: calls.append("bm25"))
    return calls


@pytest.mark.parametrize(
    ("mode", "expected_indexes"),
    [
        ("dense", ["vector"]),
        ("bm25", ["bm25"]),
        ("hybrid", ["vector", "bm25"]),
    ],
)
def test_initialization_creates_only_indexes_required_by_mode(
    monkeypatch, mode: str, expected_indexes: list[str]
) -> None:
    store = _store(mode=mode, metadata_indexes={"tenant_id": "text"})
    calls = _record_initialization(store, monkeypatch)

    store._ensure_initialized()

    assert calls == ["table", "metadata:tenant_id:text", *expected_indexes]
    assert store._initialized is True


def test_initialization_runs_once(monkeypatch) -> None:
    store = _store()
    calls = _record_initialization(store, monkeypatch)

    store._ensure_initialized()
    store._ensure_initialized()

    assert calls == ["table", "vector"]


def test_setup_is_idempotent(monkeypatch) -> None:
    store = _store()
    calls = _record_initialization(store, monkeypatch)

    store.setup()
    store.setup()

    assert calls == ["table", "vector"]


def test_failed_initialization_is_not_cached(monkeypatch) -> None:
    store = _store()
    calls: list[str] = []
    attempts = 0

    def prepare_table() -> None:
        nonlocal attempts
        attempts += 1
        calls.append("table")
        if attempts == 1:
            raise RuntimeError("ddl failed")

    monkeypatch.setattr(store, "_prepare_table_if_needed", prepare_table)
    monkeypatch.setattr(store, "_create_vector_index", lambda: calls.append("vector"))
    monkeypatch.setattr(store, "_create_bm25_index", lambda: calls.append("bm25"))

    with pytest.raises(RuntimeError, match="ddl failed"):
        store._ensure_initialized()
    store._ensure_initialized()

    assert calls == ["table", "table", "vector"]
    assert store._initialized is True


def test_initialization_lock_serializes_concurrent_callers(monkeypatch) -> None:
    store = _store()
    entered = Event()
    release = Event()
    calls = 0
    calls_lock = Lock()

    def prepare_table() -> None:
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(store, "_prepare_table_if_needed", prepare_table)
    monkeypatch.setattr(store, "_create_vector_index", lambda: None)
    monkeypatch.setattr(store, "_create_bm25_index", lambda: None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(store._ensure_initialized)
        assert entered.wait(timeout=5)
        second = executor.submit(store._ensure_initialized)
        release.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert calls == 1


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_distributed_gaussdb_rejects_lexical_modes_before_ddl(
    monkeypatch, mode: str
) -> None:
    store = _store(mode=mode, distributed=True)
    monkeypatch.setattr(
        store,
        "_prepare_table_if_needed",
        lambda: pytest.fail("deployment validation must run before table DDL"),
    )

    with pytest.raises(
        GaussDBCapabilityError,
        match=rf"retrieval_mode '{mode}'.*distributed.*dense",
    ):
        store.setup()

    assert store._initialized is False
    assert store._engine.executed == []


def test_distributed_gaussdb_rejects_dimension_above_1024_before_ddl(
    monkeypatch,
) -> None:
    store = _store(embedding_dimension=1025, distributed=True)
    monkeypatch.setattr(
        store,
        "_prepare_table_if_needed",
        lambda: pytest.fail("deployment validation must run before table DDL"),
    )

    with pytest.raises(GaussDBCapabilityError, match="up to 1024"):
        store.setup()

    assert store._initialized is False
    assert store._engine.executed == []


def test_centralized_gaussdb_allows_global_maximum_dimension(monkeypatch) -> None:
    store = _store(embedding_dimension=4096, distributed=False)
    calls = _record_initialization(store, monkeypatch)

    store.setup()

    assert calls == ["table", "vector"]
    assert store._initialized is True


def test_dense_dimension_1024_needs_no_topology_probe(monkeypatch) -> None:
    store = _store(embedding_dimension=1024)
    calls = _record_initialization(store, monkeypatch)

    store.setup()

    assert calls == ["table", "vector"]
    assert store._engine.fetched == []


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_similarity_search_never_runs_initialization(monkeypatch, mode: str) -> None:
    fetch_results = [[], []] if mode == "hybrid" else [[]]
    engine = RecordingEngine(fetch_results=fetch_results)
    store = _store(mode=mode)
    store._engine = engine

    monkeypatch.setattr(
        store,
        "_ensure_initialized",
        lambda: pytest.fail("query path must not initialize storage"),
    )

    assert store.similarity_search("needle") == []
    assert engine.executed == []


def test_mmr_search_never_runs_initialization(monkeypatch) -> None:
    engine = RecordingEngine(fetch_results=[[]])
    store = _store()
    store._engine = engine

    monkeypatch.setattr(
        store,
        "_ensure_initialized",
        lambda: pytest.fail("query path must not initialize storage"),
    )

    assert store.max_marginal_relevance_search("needle", k=1, fetch_k=1) == []
    assert engine.executed == []
