from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.sql import CompiledSQL

STANDARD_SEARCH_CASES = (
    {
        "search_type": "similarity",
        "search_kwargs": {"k": 2},
        "expected_ids": ("alpha", "beta"),
    },
    {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {"k": 3, "score_threshold": 0.99},
        "expected_ids": ("alpha",),
    },
    {
        "search_type": "mmr",
        "search_kwargs": {"k": 2, "fetch_k": 3, "lambda_mult": 0.7},
        "expected_ids": ("alpha", "beta"),
    },
)

HYBRID_DISTANCE_CASES = (
    {"distance": "cosine", "expected_ids": ("both", "both-secondary")},
    {"distance": "l2", "expected_ids": ("both", "both-secondary")},
)

HYBRID_SEED_CASES = (
    {
        "id": "both",
        "text": "alpha lexicalneedle lexicalneedle",
        "tenant": "keep",
        "branch": "both",
    },
    {
        "id": "both-secondary",
        "text": "beta lexicalneedle",
        "tenant": "keep",
        "branch": "both",
    },
    {
        "id": "sparse",
        "text": "lexicalneedle lexicalneedle sparse",
        "tenant": "drop",
        "branch": "sparse-only",
    },
    {
        "id": "dense",
        "text": "alpha denseonly",
        "tenant": "drop",
        "branch": "dense-only",
    },
)

NON_DENSE_GATE_CASES = (
    {"mode": "bm25", "search_type": "similarity_score_threshold"},
    {"mode": "bm25", "search_type": "mmr"},
    {"mode": "hybrid", "search_type": "similarity_score_threshold"},
    {"mode": "hybrid", "search_type": "mmr"},
)

CALLBACK_ENTRYPOINT_CASES = (
    {"entrypoint": "invoke", "query_count": 1},
    {"entrypoint": "ainvoke", "query_count": 1},
    {"entrypoint": "batch", "query_count": 2},
    {"entrypoint": "abatch", "query_count": 2},
)


class _RetrieverEmbeddings(Embeddings):
    _VECTORS = {
        "alpha exact": [1.0, 0.0, 0.0],
        "beta nearby": [0.8, 0.6, 0.0],
        "gamma remote": [0.0, 1.0, 0.0],
        "lexicalneedle": [1.0, 0.0, 0.0],
        "alpha lexicalneedle lexicalneedle": [1.0, 0.0, 0.0],
        "beta lexicalneedle": [0.8, 0.6, 0.0],
        "lexicalneedle lexicalneedle sparse": [0.0, 1.0, 0.0],
        "alpha denseonly": [1.0, 0.0, 0.0],
    }

    def __init__(self) -> None:
        self.calls = {
            "embed_documents": 0,
            "embed_query": 0,
        }
        self._lock = threading.Lock()

    def _record(self, name: str) -> None:
        with self._lock:
            self.calls[name] += 1

    def _vector(self, text: str) -> list[float]:
        return list(self._VECTORS.get(text, [1.0, 0.0, 0.0]))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._record("embed_documents")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self._record("embed_query")
        return self._vector(text)


class _RetrieverCallbackRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.starts.append(
                {
                    "query": query,
                    "run_id": str(run_id),
                    "parent_run_id": (
                        None if parent_run_id is None else str(parent_run_id)
                    ),
                    "tags": tuple(tags or ()),
                    "metadata": dict(metadata or {}),
                }
            )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.ends.append(
                {
                    "run_id": str(run_id),
                    "parent_run_id": (
                        None if parent_run_id is None else str(parent_run_id)
                    ),
                    "ids": _ids(documents),
                }
            )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.errors.append(
                {
                    "run_id": str(run_id),
                    "parent_run_id": (
                        None if parent_run_id is None else str(parent_run_id)
                    ),
                    "type": type(error).__name__,
                }
            )


class _NoDatabaseEngine:
    def __init__(self) -> None:
        self.capability_probe_count = 0

    def __getattr__(self, name: str) -> Any:
        self.capability_probe_count += 1
        raise AssertionError(f"database access preceded non-Dense gate: {name}")


def _store(
    engine: GaussDBEngine,
    table: Any,
    embeddings: Embeddings,
    *,
    mode: str = "dense",
    distance: str = "cosine",
) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        engine=engine,
        embedding=embeddings,
        schema_name=table.schema,
        table_name=table.name,
        embedding_dimension=3,
        retrieval_mode=mode,
        distance_strategy=distance,
    )


def _ids(documents: Sequence[Document]) -> list[str]:
    return [str(document.id) for document in documents]


def _seed_dense(store: GaussDBVectorStore) -> None:
    assert store.add_texts(
        ["alpha exact", "beta nearby", "gamma remote"],
        metadatas=[
            {"tenant": "keep", "rank": 1},
            {"tenant": "keep", "rank": 2},
            {"tenant": "drop", "rank": 3},
        ],
        ids=["alpha", "beta", "gamma"],
    ) == ["alpha", "beta", "gamma"]


def _seed_mode(store: GaussDBVectorStore) -> None:
    assert store.add_texts(
        [case["text"] for case in HYBRID_SEED_CASES],
        metadatas=[
            {"tenant": case["tenant"], "branch": case["branch"]}
            for case in HYBRID_SEED_CASES
        ],
        ids=[case["id"] for case in HYBRID_SEED_CASES],
    ) == [case["id"] for case in HYBRID_SEED_CASES]


def _assert_single_and_batch_results(
    sync_single: Sequence[Document],
    async_single: Sequence[Document],
    sync_batch: Sequence[Sequence[Document]],
    async_batch: Sequence[Sequence[Document]],
    expected_ids: Sequence[str],
) -> None:
    expected = list(expected_ids)
    assert _ids(sync_single) == expected
    assert _ids(async_single) == expected
    assert [_ids(result) for result in sync_batch] == [expected, expected]
    assert [_ids(result) for result in async_batch] == [expected, expected]


@pytest.mark.parametrize(
    "case", STANDARD_SEARCH_CASES, ids=lambda case: case["search_type"]
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_standard_retriever_invoke_ainvoke_batch_abatch_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    embeddings = _RetrieverEmbeddings()
    store = _store(writable_engine, temporary_vector_table, embeddings)
    _seed_dense(store)
    retriever = store.as_retriever(
        search_type=case["search_type"],
        search_kwargs=dict(case["search_kwargs"]),
    )

    sync_single = retriever.invoke("alpha exact")
    async_single = await retriever.ainvoke("alpha exact")
    sync_batch = retriever.batch(["alpha exact", "alpha exact"])
    async_batch = await retriever.abatch(["alpha exact", "alpha exact"])

    _assert_single_and_batch_results(
        sync_single,
        async_single,
        sync_batch,
        async_batch,
        case["expected_ids"],
    )
    assert embeddings.calls["embed_query"] == 6


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_bm25_retriever_invoke_ainvoke_batch_abatch(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_bm25: bool,
) -> None:
    embeddings = _RetrieverEmbeddings()
    store = _store(
        writable_engine,
        temporary_vector_table,
        embeddings,
        mode="bm25",
    )
    _seed_mode(store)
    before_query_calls = embeddings.calls["embed_query"]
    retriever = store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"tenant": {"$eq": "keep"}},
        }
    )

    sync_single = retriever.invoke("lexicalneedle")
    async_single = await retriever.ainvoke("lexicalneedle")
    sync_batch = retriever.batch(["lexicalneedle", "lexicalneedle"])
    async_batch = await retriever.abatch(["lexicalneedle", "lexicalneedle"])

    _assert_single_and_batch_results(
        sync_single,
        async_single,
        sync_batch,
        async_batch,
        ("both", "both-secondary"),
    )
    assert embeddings.calls["embed_query"] == before_query_calls


@pytest.mark.parametrize(
    "case",
    HYBRID_DISTANCE_CASES,
    ids=lambda case: case["distance"],
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_hybrid_retriever_invoke_ainvoke_batch_abatch(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_bm25: bool,
    requires_gsdiskann: bool,
) -> None:
    embeddings = _RetrieverEmbeddings()
    store = _store(
        writable_engine,
        temporary_vector_table,
        embeddings,
        mode="hybrid",
        distance=case["distance"],
    )
    _seed_mode(store)
    filter_value = {"tenant": {"$eq": "keep"}}
    search_kwargs = {"k": 3, "filter": filter_value}
    retriever = store.as_retriever(search_kwargs=search_kwargs)

    sync_single = retriever.invoke("lexicalneedle")
    async_single = await retriever.ainvoke("lexicalneedle")
    sync_batch = retriever.batch(["lexicalneedle", "lexicalneedle"])
    async_batch = await retriever.abatch(["lexicalneedle", "lexicalneedle"])

    _assert_single_and_batch_results(
        sync_single,
        async_single,
        sync_batch,
        async_batch,
        case["expected_ids"],
    )
    scored = store.similarity_search_with_score(
        "lexicalneedle",
        k=3,
        filter=filter_value,
    )
    scored_ids = [str(document.id) for document, _ in scored]
    scores = [score for _, score in scored]
    expected_ids = list(case["expected_ids"])
    assert scored_ids == expected_ids
    assert len(scored_ids) == len(set(scored_ids))
    assert scores and all(score > 0.0 for score in scores)
    assert all(left >= right for left, right in zip(scores, scores[1:]))
    assert scored_ids[0] == expected_ids[0]


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_mode_retrievers_do_not_mutate_shared_store(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_bm25: bool,
    requires_gsdiskann: bool,
) -> None:
    embeddings = _RetrieverEmbeddings()
    store = _store(writable_engine, temporary_vector_table, embeddings, mode="dense")
    _seed_mode(store)
    filter_value = {"tenant": {"$eq": "keep"}}
    dense = store.as_retriever(
        search_kwargs={"k": 1, "filter": filter_value},
    )
    bm25_store = _store(
        writable_engine, temporary_vector_table, embeddings, mode="bm25"
    )
    hybrid_store = _store(
        writable_engine, temporary_vector_table, embeddings, mode="hybrid"
    )
    bm25_store.setup()
    hybrid_store.setup()
    bm25 = bm25_store.as_retriever(
        search_kwargs={"k": 1, "filter": filter_value},
    )
    hybrid = hybrid_store.as_retriever(
        search_kwargs={"k": 1, "filter": filter_value},
    )

    dense_results = dense.invoke("alpha lexicalneedle")
    bm25_results, hybrid_results = await asyncio.gather(
        bm25.ainvoke("lexicalneedle"),
        hybrid.ainvoke("lexicalneedle"),
    )

    assert _ids(dense_results) == ["both"]
    assert _ids(bm25_results) == ["both"]
    assert _ids(hybrid_results) == ["both"]
    assert store.retrieval_mode == "dense"


@pytest.mark.parametrize(
    "case",
    CALLBACK_ENTRYPOINT_CASES,
    ids=lambda case: case["entrypoint"],
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_retriever_callbacks_and_metadata_survive_executor_async(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    embeddings = _RetrieverEmbeddings()
    store = _store(writable_engine, temporary_vector_table, embeddings)
    _seed_dense(store)
    retriever = store.as_retriever(
        search_kwargs={"k": 1},
        tags=["retriever-constructor-tag"],
    )
    recorder = _RetrieverCallbackRecorder()
    config = {
        "callbacks": [recorder],
        "tags": ["invoke-tag"],
        "metadata": {"tenant": "callback-tenant", "trace": "executor-async"},
    }
    entrypoint = case["entrypoint"]
    if entrypoint == "invoke":
        results: Any = retriever.invoke("alpha exact", config=config)
        assert _ids(results) == ["alpha"]
    elif entrypoint == "ainvoke":
        results = await retriever.ainvoke("alpha exact", config=config)
        assert _ids(results) == ["alpha"]
    elif entrypoint == "batch":
        results = retriever.batch(["alpha exact", "alpha exact"], config=config)
        assert [_ids(result) for result in results] == [["alpha"], ["alpha"]]
    else:
        assert entrypoint == "abatch"
        results = await retriever.abatch(
            ["alpha exact", "alpha exact"],
            config=config,
        )
        assert [_ids(result) for result in results] == [["alpha"], ["alpha"]]

    expected_count = int(case["query_count"])
    assert len(recorder.starts) == expected_count
    assert len(recorder.ends) == expected_count
    assert recorder.errors == []
    start_run_ids = [event["run_id"] for event in recorder.starts]
    end_run_ids = [event["run_id"] for event in recorder.ends]
    error_run_ids = [event["run_id"] for event in recorder.errors]
    assert len(start_run_ids) == len(set(start_run_ids))
    assert len(end_run_ids) == len(set(end_run_ids))
    assert len(error_run_ids) == len(set(error_run_ids))
    assert set(end_run_ids).isdisjoint(error_run_ids)
    assert set(start_run_ids) == set(end_run_ids) | set(error_run_ids)
    assert all(start["query"] == "alpha exact" for start in recorder.starts)
    assert all(
        {"retriever-constructor-tag", "invoke-tag"} <= set(start["tags"])
        for start in recorder.starts
    )
    assert all(
        start["metadata"]["tenant"] == "callback-tenant"
        and start["metadata"]["trace"] == "executor-async"
        for start in recorder.starts
    )
    assert [event["ids"] for event in recorder.ends] == [["alpha"]] * expected_count


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_exhaustive
@pytest.mark.asyncio
async def test_sync_async_retrievers_share_pool_and_heartbeat_progresses(
    writable_engine_factory: Any,
    temporary_vector_table: Any,
    requires_bm25: bool,
    requires_gsdiskann: bool,
    resource_registry: Any,
    e2e_namespace: Any,
) -> None:
    engine = writable_engine_factory(minconn=2, maxconn=2)
    embeddings = _RetrieverEmbeddings()
    store = _store(engine, temporary_vector_table, embeddings, mode="dense")
    _seed_mode(store)
    filter_value = {"tenant": {"$eq": "keep"}}
    dense = store.as_retriever(search_kwargs={"k": 1, "filter": filter_value})
    bm25_store = _store(engine, temporary_vector_table, embeddings, mode="bm25")
    hybrid_store = _store(engine, temporary_vector_table, embeddings, mode="hybrid")
    bm25_store.setup()
    hybrid_store.setup()
    bm25 = bm25_store.as_retriever(search_kwargs={"k": 1, "filter": filter_value})
    hybrid = hybrid_store.as_retriever(search_kwargs={"k": 1, "filter": filter_value})

    release_connections = threading.Event()
    held_ready: queue.Queue[tuple[str, int]] = queue.Queue()
    holder_failures: queue.Queue[BaseException] = queue.Queue()
    holders: list[threading.Thread] = []
    held: list[tuple[str, int]] = []

    def hold_connection(role: str) -> None:
        try:
            with engine.connection() as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute("SELECT pg_backend_pid()")
                    pid = int(cursor.fetchone()[0])
                finally:
                    cursor.close()
                held_ready.put((role, pid))
                if not release_connections.wait(timeout=30.0):
                    raise TimeoutError("pool holder release timed out")
        except BaseException as exc:
            holder_failures.put(exc)

    for role in ("pool_hold_one", "pool_hold_two"):
        holder = threading.Thread(
            target=hold_connection,
            args=(role,),
            name=f"gaussdb-e2e-{role}",
            daemon=True,
        )
        holder.start()
        holders.append(holder)

    for _ in holders:
        role, pid = held_ready.get(timeout=10.0)
        label = e2e_namespace.name(role)
        resource_registry.register_session(label, pid)
        held.append((label, pid))
    held_pids = {pid for _label, pid in held}

    sync_outcome: queue.Queue[Any] = queue.Queue()
    sync_started = threading.Event()

    def invoke_dense() -> None:
        sync_started.set()
        try:
            sync_outcome.put(("result", dense.invoke("alpha lexicalneedle")))
        except BaseException as exc:
            sync_outcome.put(("error", exc))

    worker = threading.Thread(
        target=invoke_dense,
        name="gaussdb-e2e-sync-retriever",
        daemon=True,
    )
    bm25_task: asyncio.Task[list[Document]] | None = None
    hybrid_task: asyncio.Task[list[Document]] | None = None
    try:
        worker.start()
        assert sync_started.wait(timeout=2.0)
        bm25_task = asyncio.create_task(bm25.ainvoke("lexicalneedle"))
        hybrid_task = asyncio.create_task(hybrid.ainvoke("lexicalneedle"))
        await asyncio.sleep(0)

        heartbeat = 0
        for _ in range(20):
            assert not bm25_task.done()
            assert not hybrid_task.done()
            assert worker.is_alive()
            heartbeat += 1
            await asyncio.sleep(0)
        assert heartbeat == 20

        release_connections.set()
        for holder in holders:
            holder.join(timeout=15.0)
            assert not holder.is_alive()
        assert holder_failures.empty(), repr(holder_failures.get_nowait())
        while held:
            label, pid = held.pop()
            resource_registry.release_session(label, pid)

        bm25_results, hybrid_results = await asyncio.wait_for(
            asyncio.gather(bm25_task, hybrid_task),
            timeout=15.0,
        )
        worker.join(timeout=15.0)
        assert not worker.is_alive()
        outcome, payload = sync_outcome.get_nowait()
        assert outcome == "result", repr(payload)
        assert _ids(payload) == ["both"]
        assert _ids(bm25_results) == ["both"]
        assert _ids(hybrid_results) == ["both"]
        assert store.retrieval_mode == "dense"

        rows = engine.fetch_all(
            CompiledSQL(sql.SQL("SELECT pg_backend_pid(), 1")),
            operation="verify shared Retriever pool remains reusable",
        )
        assert rows[0][1] == 1
        assert int(rows[0][0]) in held_pids
    finally:
        release_connections.set()
        for holder in holders:
            if holder.is_alive():
                holder.join(timeout=15.0)
        if held:
            while held:
                label, pid = held.pop()
                resource_registry.release_session(label, pid)
        pending = [
            task
            for task in (bm25_task, hybrid_task)
            if task is not None and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if worker.is_alive():
            worker.join(timeout=15.0)


@pytest.mark.parametrize(
    "case",
    NON_DENSE_GATE_CASES,
    ids=lambda case: f"{case['mode']}-{case['search_type']}",
)
@pytest.mark.asyncio
async def test_non_dense_relevance_and_mmr_gates_precede_capability(
    case: dict[str, str],
) -> None:
    engine = _NoDatabaseEngine()
    embeddings = _RetrieverEmbeddings()
    store = GaussDBVectorStore(
        engine=engine,
        embedding=embeddings,
        table_name="synthetic_non_dense_gate",
        embedding_dimension=3,
        retrieval_mode=case["mode"],
    )
    retriever = store.as_retriever(
        search_type=case["search_type"],
        search_kwargs={"k": 1, "fetch_k": 2, "score_threshold": 0.5},
    )

    with pytest.raises(ValueError, match="only supported for dense retrieval"):
        retriever.invoke("must-not-reach-database")
    with pytest.raises(ValueError, match="only supported for dense retrieval"):
        await retriever.ainvoke("must-not-reach-database")

    assert engine.capability_probe_count == 0
    assert embeddings.calls == {
        "embed_documents": 0,
        "embed_query": 0,
    }
