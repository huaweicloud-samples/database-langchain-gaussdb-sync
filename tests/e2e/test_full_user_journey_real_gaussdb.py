from __future__ import annotations

import asyncio
import datetime as dt
import queue
import threading
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, message_to_dict
from langchain_core.retrievers import BaseRetriever
from psycopg2 import sql

from langchain_gaussdb import (
    BM25Config,
    CompiledSQL,
    GaussDBChatMessageHistory,
    GaussDBConnectionError,
    GaussDBEngine,
    GaussDBVectorStore,
)
from langchain_gaussdb.indexes import (
    build_create_bm25_index,
    build_create_metadata_index,
    build_create_vector_index,
)

JOURNEY_STEPS = (
    {"id": "J01_root_import", "description": "provider root imports"},
    {
        "id": "J02_typed_automatic_dense_store",
        "description": "typed automatic dense store on maxconn=2 Engine",
    },
    {
        "id": "J03_mode_setup_and_sync_write",
        "description": "mode setup prepares required indexes before sync write",
    },
    {
        "id": "J04_async_upsert_projection",
        "description": "LangChain async compatibility keeps ODKU projections aligned",
    },
    {"id": "J05_sync_async_get_ids", "description": "sync/async ID reads"},
    {
        "id": "J06_dense_cosine_search_matrix",
        "description": "cosine query/vector/score/relevance/MMR matrix",
    },
    {
        "id": "J07_nested_typed_null_filter",
        "description": "nested typed and explicit JSON null/empty controls",
    },
    {
        "id": "J08_mode_retriever_entrypoints",
        "description": "mode-configured stores use the standard Retriever entrypoint",
    },
    {
        "id": "J09_hybrid_branch_and_embedding_contract",
        "description": "BM25 embedding isolation and Hybrid branch controls",
    },
    {
        "id": "J10_sync_async_pool_heartbeat",
        "description": "sync Dense plus async Hybrid share bounded pool",
    },
    {
        "id": "J11_chat_sessions_order_isolation",
        "description": "two ChatHistory sessions preserve order and isolation",
    },
    {"id": "J15_delete_contract", "description": "partial, guarded and full delete"},
    {
        "id": "J16_external_owner_close",
        "description": "store/history close does not close external Engine",
    },
    {
        "id": "J17_engine_close_contract",
        "description": "Engine close rejects new work and is idempotent",
    },
    {
        "id": "J18_registry_catalog_cleanup",
        "description": "registry cleanup leaves no random-prefix catalog objects",
    },
)


class _JourneyLedger:
    def __init__(self) -> None:
        self._expected = [item["id"] for item in JOURNEY_STEPS]
        self._completed: list[str] = []

    def complete(self, step_id: str, *, evidence: Any) -> None:
        assert bool(evidence), f"journey step {step_id} has no positive evidence"
        assert self._expected[len(self._completed)] == step_id
        self._completed.append(step_id)

    def assert_complete(self) -> None:
        assert self._completed == self._expected


class _JourneyEmbeddings(Embeddings):
    _VECTORS = {
        "journeyneedle": [1.0, 0.0, 0.0],
        "journeyneedle journeyneedle target": [1.0, 0.0, 0.0],
        "journeyneedle journeyneedle target updated": [1.0, 0.0, 0.0],
        "journeyneedle secondary": [0.8, 0.6, 0.0],
        "journeyneedle sparse control": [0.0, 1.0, 0.0],
        "dense control": [0.9, 0.435889894, 0.0],
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
        return list(self._VECTORS.get(text, [0.0, 0.0, 1.0]))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._record("embed_documents")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self._record("embed_query")
        return self._vector(text)


def _ids(documents: Any) -> list[str]:
    return [str(document.id) for document in documents]


def _automatic_index_names(
    *,
    schema_name: str,
    table_name: str,
    metadata_indexes: dict[str, str | None],
) -> dict[str, str]:
    vector_name, _vector_sql = build_create_vector_index(
        schema_name,
        table_name,
        "embedding",
        3,
        distance_strategy="cosine",
    )
    bm25_name, _bm25_sql = build_create_bm25_index(
        schema_name,
        table_name,
        "content",
    )
    names = {"vector": vector_name, "bm25": bm25_name}
    for field, cast in metadata_indexes.items():
        names[f"metadata:{field}"] = build_create_metadata_index(
            schema_name,
            table_name,
            "metadata",
            field,
            cast=cast,
        )[0]
    return names


def _index_catalog(
    engine: GaussDBEngine,
    schema_name: str,
    table_name: str,
) -> list[tuple[Any, ...]]:
    return engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT idx.relname, am.amname, ind.indisvalid, ind.indisready, "
                "ind.indisusable, pg_get_indexdef(ind.indexrelid) "
                "FROM pg_index AS ind "
                "JOIN pg_class AS idx ON idx.oid = ind.indexrelid "
                "JOIN pg_class AS tab ON tab.oid = ind.indrelid "
                "JOIN pg_namespace AS ns ON ns.oid = tab.relnamespace "
                "JOIN pg_am AS am ON am.oid = idx.relam "
                "WHERE ns.nspname = %s AND tab.relname = %s "
                "ORDER BY idx.relname"
            ),
            (schema_name, table_name),
        ),
        operation="inspect full journey index catalog",
    )


def _index_ready(rows: list[tuple[Any, ...]], name: str) -> bool:
    matches = [row for row in rows if str(row[0]) == name]
    return len(matches) == 1 and matches[0][2:5] == (True, True, True)


def _chat_index_name(table_name: str) -> str:
    name = f"{table_name}_session_id_id_idx"
    assert len(name.encode("utf-8")) <= 63
    return name


@dataclass
class _HeldTableLock:
    connection_context: Any
    connection: Any
    cursor: Any


def _hold_table_lock(
    engine: GaussDBEngine,
    schema_name: str,
    table_name: str,
) -> tuple[_HeldTableLock, int]:
    connection_context = engine.connection()
    connection = connection_context.__enter__()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            sql.SQL("LOCK TABLE {}.{} IN ACCESS EXCLUSIVE MODE").format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
            )
        )
        cursor.execute("SELECT pg_backend_pid()")
        pid = int(cursor.fetchone()[0])
        return _HeldTableLock(connection_context, connection, cursor), pid
    except BaseException:
        try:
            connection.rollback()
        finally:
            if cursor is not None:
                cursor.close()
            connection_context.__exit__(None, None, None)
        raise


def _release_table_lock(
    held_lock: _HeldTableLock,
    *,
    rollback: bool = False,
) -> None:
    primary_error: BaseException | None = None
    try:
        try:
            if rollback:
                held_lock.connection.rollback()
            else:
                held_lock.connection.commit()
        except BaseException as exc:
            primary_error = exc
            try:
                held_lock.connection.rollback()
            except BaseException:
                pass
            raise
    finally:
        try:
            held_lock.cursor.close()
        finally:
            try:
                held_lock.connection_context.__exit__(None, None, None)
            except BaseException:
                if primary_error is None:
                    raise


async def _wait_for_lock_waiters(
    control_engine: GaussDBEngine,
    schema_name: str,
    table_name: str,
    *,
    expected: int,
    timeout: float = 8.0,
) -> list[int]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        rows = await asyncio.to_thread(
            control_engine.fetch_all,
            CompiledSQL(
                sql.SQL(
                    "SELECT DISTINCT activity.pid "
                    "FROM pg_locks AS locks "
                    "JOIN pg_stat_activity AS activity "
                    "ON activity.pid = locks.pid "
                    "JOIN pg_class AS relation ON relation.oid = locks.relation "
                    "JOIN pg_namespace AS ns ON ns.oid = relation.relnamespace "
                    "WHERE activity.datname = current_database() "
                    "AND activity.pid <> pg_backend_pid() "
                    "AND locks.granted IS FALSE "
                    "AND ns.nspname = %s "
                    "AND relation.relname = %s "
                    "ORDER BY activity.pid"
                ),
                (schema_name, table_name),
            ),
            operation="observe full journey retrieval lock waiters",
        )
        pids = sorted({int(row[0]) for row in rows})
        if len(pids) >= expected:
            return pids
        if loop.time() >= deadline:
            raise TimeoutError(
                f"observed {len(pids)} retrieval lock waiter(s); expected {expected}"
            )
        await asyncio.sleep(0.02)


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_fast
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_full_sync_async_langchain_user_journey_on_one_shared_engine(
    writable_engine_factory: Any,
    control_engine: GaussDBEngine,
    capability_snapshot: dict[str, object],
    requires_gsdiskann: bool,
    requires_bm25: bool,
    e2e_namespace: Any,
    resource_registry: Any,
) -> None:
    journey = _JourneyLedger()

    root_objects = (
        GaussDBEngine,
        GaussDBVectorStore,
        GaussDBChatMessageHistory,
    )
    assert all(
        value.__module__.startswith("langchain_gaussdb") for value in root_objects
    )
    journey.complete(
        "J01_root_import",
        evidence=all(value is not None for value in root_objects),
    )

    assert requires_gsdiskann and requires_bm25
    assert "gsdiskann" in capability_snapshot["access_methods"]
    bm25_operator_available = capability_snapshot.get("bm25") is True
    bm25_access_method_available = "bm25" in capability_snapshot["access_methods"]
    if not (bm25_operator_available and bm25_access_method_available):
        pytest.skip(
            "owned full journey requires both BM25 operator and bm25 access method"
        )
    floatvector_available = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = %s "
                "AND pg_type_is_visible(oid))"
            ),
            ("floatvector",),
        ),
        operation="verify full journey floatvector prerequisite",
    )
    if floatvector_available != [(True,)]:
        pytest.skip("GaussDB capability floatvector is explicitly unavailable")
    writer_maxconn = 2
    writer = resource_registry.register_engine(
        e2e_namespace.name("journey_writer"),
        writable_engine_factory(minconn=2, maxconn=writer_maxconn),
    )
    schema_name = e2e_namespace.schema("journey_schema")
    writer.execute(
        CompiledSQL(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))),
        operation="create full journey schema",
    )
    vector_table = e2e_namespace.table(schema_name, "journey_vector")
    chat_table = e2e_namespace.table(schema_name, "journey_chat")
    embeddings = _JourneyEmbeddings()
    store = GaussDBVectorStore(
        engine=writer,
        embedding=embeddings,
        schema_name=schema_name,
        table_name=vector_table.name,
        embedding_dimension=3,
        distance_strategy="cosine",
        retrieval_mode="dense",
        metadata_indexes={
            "tenant": "text",
            "priority": "bigint",
            "event_date": "date",
        },
        bm25_config=BM25Config(column="content"),
    )
    automatic_names = _automatic_index_names(
        schema_name=schema_name,
        table_name=vector_table.name,
        metadata_indexes=store.metadata_indexes,
    )
    for index_name in automatic_names.values():
        resource_registry.register_index(schema_name, index_name)
    mode_store_options = {
        "engine": writer,
        "embedding": embeddings,
        "schema_name": schema_name,
        "table_name": vector_table.name,
        "embedding_dimension": 3,
        "metadata_indexes": store.metadata_indexes,
    }
    bm25_store = GaussDBVectorStore(
        **mode_store_options,
        retrieval_mode="bm25",
    )
    hybrid_store = GaussDBVectorStore(
        **mode_store_options,
        retrieval_mode="hybrid",
    )
    store.setup()
    bm25_store.setup()
    hybrid_store.setup()
    journey.complete(
        "J02_typed_automatic_dense_store",
        evidence=(
            store.metadata_indexes
            == {"tenant": "text", "priority": "bigint", "event_date": "date"}
            and store.retrieval_mode == "dense"
        ),
    )

    seed_documents = [
        Document(
            id="target",
            page_content="journeyneedle journeyneedle target",
            metadata={
                "tenant": "tenant-a",
                "priority": 10,
                "event_date": dt.date(2026, 7, 1),
                "status": "active",
                "null_gate": None,
                "empty_gate": "value",
                "branch": "both",
            },
        ),
        Document(
            id="both-secondary",
            page_content="journeyneedle secondary",
            metadata={
                "tenant": "tenant-a",
                "priority": 20,
                "event_date": dt.date(2026, 7, 2),
                "status": "active",
                "branch": "both",
            },
        ),
        Document(
            id="sparse-only",
            page_content="journeyneedle sparse control",
            metadata={"tenant": "tenant-drop", "priority": 30, "branch": "sparse-only"},
        ),
        Document(
            id="dense-only",
            page_content="dense control",
            metadata={"tenant": "tenant-drop", "priority": 40, "branch": "dense-only"},
        ),
        Document(
            id="missing-control",
            page_content="missing state control",
            metadata={
                "tenant": "tenant-a",
                "priority": 11,
                "event_date": dt.date(2026, 7, 2),
            },
        ),
        Document(
            id="null-control",
            page_content="null state control",
            metadata={
                "tenant": "tenant-a",
                "priority": 11,
                "event_date": dt.date(2026, 7, 2),
                "status": None,
            },
        ),
        Document(
            id="empty-control",
            page_content="empty state control",
            metadata={
                "tenant": "tenant-a",
                "priority": 11,
                "event_date": dt.date(2026, 7, 2),
                "status": "",
                "empty_gate": "",
            },
        ),
    ]
    assert store.add_documents(seed_documents) == [
        "target",
        "both-secondary",
        "sparse-only",
        "dense-only",
        "missing-control",
        "null-control",
        "empty-control",
    ]
    dense_catalog = _index_catalog(control_engine, schema_name, vector_table.name)
    required_dense_indexes = {
        automatic_names["vector"],
        automatic_names["bm25"],
        automatic_names["metadata:tenant"],
        automatic_names["metadata:priority"],
        automatic_names["metadata:event_date"],
    }
    journey.complete(
        "J03_mode_setup_and_sync_write",
        evidence=all(
            _index_ready(dense_catalog, name) for name in required_dense_indexes
        ),
    )

    sync_embed_calls_before_async = embeddings.calls["embed_documents"]
    assert await store.aadd_texts(
        ["journeyneedle journeyneedle target updated"],
        ids=["target"],
        metadatas=[
            {
                "tenant": "tenant-a",
                "priority": 11,
                "event_date": dt.date(2026, 7, 2),
                "status": "active",
                "null_gate": None,
                "empty_gate": "value",
                "branch": "both",
            }
        ],
    ) == ["target"]
    projection = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT count(*), min(content), min(metadata->>'tenant'), "
                "min(metadata->>'priority'), min(metadata->>'event_date') "
                "FROM {}.{} WHERE id = %s"
            ).format(sql.Identifier(schema_name), sql.Identifier(vector_table.name)),
            ("target",),
        ),
        operation="verify full journey ODKU projection",
    )
    assert len(projection) == 1
    (
        row_count,
        updated_content,
        json_tenant,
        json_priority,
        json_event_date,
    ) = projection[0]
    assert row_count == 1
    assert updated_content == "journeyneedle journeyneedle target updated"
    assert json_tenant == "tenant-a"
    assert json_priority == "11"
    assert json_event_date == "2026-07-02"
    assert embeddings.calls["embed_documents"] == sync_embed_calls_before_async + 1
    journey.complete(
        "J04_async_upsert_projection",
        evidence=(
            updated_content == "journeyneedle journeyneedle target updated"
            and json_tenant == "tenant-a"
            and json_priority == "11"
            and json_event_date == "2026-07-02"
            and embeddings.calls["embed_documents"] == sync_embed_calls_before_async + 1
        ),
    )

    sync_get = store.get_by_ids(["missing-id", "target"])
    async_get = await store.aget_by_ids(["both-secondary", "missing-id"])
    empty_sync_get = store.get_by_ids([])
    empty_async_get = await store.aget_by_ids([])
    journey.complete(
        "J05_sync_async_get_ids",
        evidence=(
            _ids(sync_get) == ["target"]
            and _ids(async_get) == ["both-secondary"]
            and empty_sync_get == []
            and empty_async_get == []
            and sync_get[0].id == "target"
        ),
    )

    dense_no_score = store.similarity_search("journeyneedle", k=4)
    raw_scores = store.similarity_search_with_score("journeyneedle", k=4)
    relevance_scores = store.similarity_search_with_relevance_scores(
        "journeyneedle",
        k=4,
    )
    query_vector = embeddings.embed_query("journeyneedle")
    by_vector_scores = store.similarity_search_with_score_by_vector(query_vector, k=4)
    sync_mmr = store.max_marginal_relevance_search(
        "journeyneedle",
        k=2,
        fetch_k=4,
    )
    async_vector_mmr = await store.amax_marginal_relevance_search_by_vector(
        query_vector,
        k=2,
        fetch_k=4,
    )
    raw_values = [score for _, score in raw_scores]
    relevance_values = [score for _, score in relevance_scores]
    journey.complete(
        "J06_dense_cosine_search_matrix",
        evidence=(
            dense_no_score[0].id == "target"
            and raw_scores[0][0].id == "target"
            and raw_values == sorted(raw_values)
            and relevance_scores[0][0].id == "target"
            and relevance_values == sorted(relevance_values, reverse=True)
            and by_vector_scores[0][0].id == "target"
            and sync_mmr[0].id == "target"
            and async_vector_mmr[0].id == "target"
        ),
    )

    nested_filter = {
        "$and": [
            {"tenant": {"$eq": "tenant-a"}},
            {"priority": {"$between": [10, 15]}},
            {"event_date": {"$eq": dt.date(2026, 7, 2)}},
            {"status": {"$eq": "active"}},
            {"null_gate": {"$contains": None}},
            {"empty_gate": {"$ne": ""}},
        ]
    }
    filtered = store.similarity_search(
        "journeyneedle",
        k=8,
        filter=nested_filter,
    )
    journey.complete(
        "J07_nested_typed_null_filter",
        evidence=(
            _ids(filtered) == ["target"]
            and not {"missing-control", "null-control", "empty-control"}
            & set(_ids(filtered))
        ),
    )

    mode_filter = {"tenant": {"$eq": "tenant-a"}}
    bm25 = bm25_store.as_retriever(search_kwargs={"k": 4, "filter": mode_filter})
    hybrid = hybrid_store.as_retriever(search_kwargs={"k": 4, "filter": mode_filter})
    before_bm25 = dict(embeddings.calls)
    bm25_documents = bm25.invoke("journeyneedle")
    after_bm25 = dict(embeddings.calls)
    hybrid_documents = await hybrid.ainvoke("journeyneedle")
    hybrid_batches = await hybrid.abatch(["journeyneedle", "journeyneedle"])
    journey.complete(
        "J08_mode_retriever_entrypoints",
        evidence=(
            isinstance(bm25, BaseRetriever)
            and isinstance(hybrid, BaseRetriever)
            and _ids(bm25_documents)[:2] == ["target", "both-secondary"]
            and _ids(hybrid_documents)[:2] == ["target", "both-secondary"]
            and len(hybrid_batches) == 2
        ),
    )

    hybrid_ids = _ids(hybrid_documents)
    negative_controls = {"sparse-only", "dense-only"}
    journey.complete(
        "J09_hybrid_branch_and_embedding_contract",
        evidence=(
            before_bm25 == after_bm25
            and len(hybrid_ids) == len(set(hybrid_ids))
            and not negative_controls & set(hybrid_ids)
            and {"target", "both-secondary"} <= set(hybrid_ids)
        ),
    )

    blocker_engine = resource_registry.register_engine(
        e2e_namespace.name("journey_lock_engine"),
        writable_engine_factory(minconn=1, maxconn=1),
    )
    blocker_label = e2e_namespace.name("journey_lock_session")
    blocker: Any | None = None
    blocker_pid: int | None = None
    blocker_registered = False
    waiter_sessions: list[tuple[str, int]] = []
    sync_outcome: queue.Queue[Any] = queue.Queue()
    sync_started = threading.Event()

    def run_sync_dense() -> None:
        sync_started.set()
        try:
            sync_outcome.put(("result", store.similarity_search("journeyneedle", k=2)))
        except BaseException as exc:
            sync_outcome.put(("error", exc))

    worker = threading.Thread(
        target=run_sync_dense, name="journey-sync-dense", daemon=True
    )
    hybrid_task: asyncio.Task[list[Document]] | None = None
    concurrent_hybrid: list[Document] = []
    concurrent_dense: list[Document] = []
    heartbeat = 0
    observed_waiter_pids: list[int] = []
    pool_was_saturated = False
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        blocker, blocker_pid = _hold_table_lock(
            blocker_engine,
            schema_name,
            vector_table.name,
        )
        resource_registry.register_session(blocker_label, blocker_pid)
        blocker_registered = True

        worker.start()
        assert sync_started.wait(timeout=2.0)
        hybrid_task = asyncio.create_task(hybrid.ainvoke("journeyneedle"))
        observed_waiter_pids = await _wait_for_lock_waiters(
            control_engine,
            schema_name,
            vector_table.name,
            expected=2,
        )
        assert len(observed_waiter_pids) == 2
        pool_was_saturated = len(observed_waiter_pids) == writer_maxconn
        assert pool_was_saturated
        for number, pid in enumerate(observed_waiter_pids):
            label = e2e_namespace.name(f"journey_waiter_{number}")
            resource_registry.register_session(label, pid)
            waiter_sessions.append((label, pid))

        loop = asyncio.get_running_loop()
        heartbeat_deadline = loop.time() + 0.1
        while loop.time() < heartbeat_deadline:
            assert worker.is_alive()
            assert not hybrid_task.done()
            heartbeat += 1
            await asyncio.sleep(0.005)

        _release_table_lock(blocker)
        blocker = None
        resource_registry.release_session(blocker_label, blocker_pid)
        blocker_registered = False

        concurrent_hybrid = await asyncio.wait_for(hybrid_task, timeout=15.0)
        worker.join(timeout=15.0)
        assert not worker.is_alive()
        outcome, payload = sync_outcome.get_nowait()
        assert outcome == "result", repr(payload)
        concurrent_dense = payload
        for label, pid in reversed(waiter_sessions):
            resource_registry.release_session(label, pid)
        waiter_sessions.clear()
    except BaseException as exc:
        primary_error = exc
    finally:
        if blocker is not None:
            try:
                _release_table_lock(blocker, rollback=primary_error is not None)
            except BaseException as exc:
                cleanup_errors.append(exc)
            else:
                if blocker_registered and blocker_pid is not None:
                    resource_registry.release_session(blocker_label, blocker_pid)
                    blocker_registered = False
        if hybrid_task is not None and not hybrid_task.done():
            hybrid_task.cancel()
            try:
                await asyncio.wait_for(hybrid_task, timeout=5.0)
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                cleanup_errors.append(exc)
        if worker.is_alive():
            worker.join(timeout=10.0)
        if worker.is_alive():
            cleanup_errors.append(
                TimeoutError("journey sync Dense worker did not settle during cleanup")
            )
        tasks_settled = hybrid_task is None or hybrid_task.done()
        if not worker.is_alive() and tasks_settled:
            for label, pid in reversed(waiter_sessions):
                try:
                    resource_registry.release_session(label, pid)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            waiter_sessions.clear()
    if primary_error is not None:
        for error in cleanup_errors:
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(
                    f"bounded J10 cleanup also failed with {type(error).__name__}"
                )
        raise primary_error
    if cleanup_errors:
        raise cleanup_errors[0]
    engine_reusable = await asyncio.to_thread(writer.check_connection)
    journey.complete(
        "J10_sync_async_pool_heartbeat",
        evidence=(
            heartbeat > 0
            and len(observed_waiter_pids) == 2
            and pool_was_saturated
            and _ids(concurrent_dense)[0] == "target"
            and _ids(concurrent_hybrid)[0] == "target"
            and engine_reusable
        ),
    )

    session_a = e2e_namespace.session_id("journey_chat_a")
    session_b = e2e_namespace.session_id("journey_chat_b")
    chat_index = resource_registry.register_index(
        schema_name,
        _chat_index_name(chat_table.name),
    )
    history_a = GaussDBChatMessageHistory(
        engine=writer,
        schema_name=schema_name,
        table_name=chat_table.name,
        session_id=session_a,
        create_table=True,
    )
    history_b = GaussDBChatMessageHistory(
        engine=writer,
        schema_name=schema_name,
        table_name=chat_table.name,
        session_id=session_b,
    )
    expected_a_sync = HumanMessage(
        content="a-sync",
        id="a-sync",
        name="journey-human-sync",
        additional_kwargs={"client": {"trace": "a-sync-trace"}},
        response_metadata={"locale": "zh-CN"},
    )
    expected_b_async = AIMessage(
        content="b-async",
        id="b-async",
        name="journey-ai-async",
        additional_kwargs={"vendor": {"trace": "b-async-trace"}},
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )
    expected_a_async = HumanMessage(
        content="a-async",
        id="a-async",
        name="journey-human-async",
        additional_kwargs={"client": {"trace": "a-async-trace"}},
        response_metadata={"locale": "en-US"},
    )
    history_a.add_messages([expected_a_sync])
    await history_b.aadd_messages([expected_b_async])
    await history_a.aadd_messages([expected_a_async])
    restored_a = await history_a.aget_messages()
    restored_b = await history_b.aget_messages()
    global_rows = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL("SELECT id, session_id FROM {}.{} ORDER BY id").format(
                sql.Identifier(schema_name),
                sql.Identifier(chat_table.name),
            )
        ),
        operation="verify full journey global chat order",
    )
    rich_fields = ("id", "name", "additional_kwargs", "response_metadata")
    assert [type(message) for message in restored_a] == [HumanMessage, HumanMessage]
    assert [type(message) for message in restored_b] == [AIMessage]
    assert [message_to_dict(message) for message in restored_a] == [
        message_to_dict(expected_a_sync),
        message_to_dict(expected_a_async),
    ]
    assert [message_to_dict(message) for message in restored_b] == [
        message_to_dict(expected_b_async)
    ]
    for field in rich_fields:
        assert (
            message_to_dict(restored_a[0])["data"][field]
            == message_to_dict(expected_a_sync)["data"][field]
        )
        assert (
            message_to_dict(restored_b[0])["data"][field]
            == message_to_dict(expected_b_async)["data"][field]
        )
    global_ids = [int(row[0]) for row in global_rows]
    global_sessions = [str(row[1]) for row in global_rows]
    assert global_ids == sorted(global_ids)
    assert global_sessions == [session_a, session_b, session_a]
    await history_b.aclear()
    remaining_a = await history_a.aget_messages()
    remaining_b = await history_b.aget_messages()
    journey.complete(
        "J11_chat_sessions_order_isolation",
        evidence=(
            [type(message) for message in restored_a] == [HumanMessage, HumanMessage]
            and [type(message) for message in restored_b] == [AIMessage]
            and [message_to_dict(message) for message in restored_a]
            == [message_to_dict(expected_a_sync), message_to_dict(expected_a_async)]
            and [message_to_dict(message) for message in restored_b]
            == [message_to_dict(expected_b_async)]
            and global_sessions == [session_a, session_b, session_a]
            and global_ids == sorted(global_ids)
            and remaining_b == []
            and [message_to_dict(message) for message in remaining_a]
            == [message_to_dict(expected_a_sync), message_to_dict(expected_a_async)]
            and _index_ready(
                _index_catalog(control_engine, schema_name, chat_table.name), chat_index
            )
        ),
    )

    assert store.delete(ids=["target", "missing-delete"])
    partial_absent = store.get_by_ids(["target"]) == []
    with pytest.raises(ValueError, match="delete_all"):
        store.delete(ids=None)
    assert store.delete(delete_all=True)
    after_delete_search = store.similarity_search("journeyneedle", k=4)
    after_delete_get = store.get_by_ids(["both-secondary"])
    journey.complete(
        "J15_delete_contract",
        evidence=partial_absent
        and after_delete_search == []
        and after_delete_get == [],
    )

    store.close()
    history_a.close()
    history_b.close()
    external_probe = writer.fetch_all(
        CompiledSQL(sql.SQL("SELECT %s::int"), (16,)),
        operation="verify external Engine ownership after store/history close",
    )
    journey.complete(
        "J16_external_owner_close",
        evidence=external_probe == [(16,)],
    )

    writer.close()
    with pytest.raises(GaussDBConnectionError) as closed_error:
        writer.fetch_all(
            CompiledSQL(sql.SQL("SELECT %s::int"), (17,)),
            operation="verify closed journey Engine rejects work",
        )
    second_close = writer.close()
    journey.complete(
        "J17_engine_close_contract",
        evidence=(
            isinstance(closed_error.value, GaussDBConnectionError)
            and second_close is None
        ),
    )

    resource_registry.cleanup()
    catalog_residue = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT count(*) FROM ("
                "SELECT nspname AS name FROM pg_namespace WHERE nspname LIKE %s "
                "UNION ALL "
                "SELECT rel.relname FROM pg_class AS rel "
                "JOIN pg_namespace AS ns ON ns.oid = rel.relnamespace "
                "WHERE rel.relname LIKE %s OR ns.nspname LIKE %s) AS residue"
            ),
            (
                f"{e2e_namespace.prefix}%",
                f"{e2e_namespace.prefix}%",
                f"{e2e_namespace.prefix}%",
            ),
        ),
        operation="verify full journey random-prefix catalog cleanup",
    )
    journey.complete(
        "J18_registry_catalog_cleanup",
        evidence=(
            catalog_residue == [(0,)]
            and resource_registry.evidence
            and all(item.absent_after_cleanup for item in resource_registry.evidence)
        ),
    )
    journey.assert_complete()
