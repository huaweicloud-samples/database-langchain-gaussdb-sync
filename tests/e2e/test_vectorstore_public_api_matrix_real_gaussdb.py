from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.sql import CompiledSQL

WRITE_OPERATIONS = (
    "add_texts",
    "add_documents",
    "get_by_ids",
    "delete",
    "upsert",
    "missing",
    "empty",
)
SEARCH_OPERATIONS = (
    "query",
    "by-vector",
    "no-score",
    "raw-score",
    "relevance",
    "mmr",
    "by-vector-mmr",
)
DISTANCE_STRATEGIES = ("cosine", "l2")
DENSE_EXPECTED_IDS = {
    "cosine": ["alpha", "beta", "gamma"],
    "l2": ["alpha", "beta", "gamma"],
}
MMR_EXPECTED_IDS = {
    "cosine": ["alpha", "beta"],
    "l2": ["alpha", "beta"],
}
SEARCH_TYPES = ("similarity", "similarity_score_threshold", "mmr")
FACTORY_APIS = (
    "from_texts",
    "from_documents",
    "afrom_texts",
    "afrom_documents",
)
FACTORY_LIFECYCLES = ("automatic",)
FACTORY_FAILURES = ("table-failure", "write-failure", "index-failure")
FACTORY_OWNERSHIP = ("owned", "external")


class _FixedPublicEmbeddings(Embeddings):
    _VECTORS = {
        "alpha exact": [1.0, 0.0, 0.0],
        "beta nearby": [0.8, 0.6, 0.0],
        "gamma remote": [0.0, 1.0, 0.0],
    }

    def __init__(self) -> None:
        self.calls = {
            "embed_documents": 0,
            "aembed_documents": 0,
            "embed_query": 0,
            "aembed_query": 0,
        }

    def _vector(self, text: str) -> list[float]:
        return list(self._VECTORS.get(text, [0.5, 0.5, 0.5]))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls["embed_documents"] += 1
        return [self._vector(text) for text in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls["aembed_documents"] += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls["embed_query"] += 1
        return self._vector(text)

    async def aembed_query(self, text: str) -> list[float]:
        self.calls["aembed_query"] += 1
        return self._vector(text)


def _store(
    engine: GaussDBEngine,
    embeddings: Any,
    table: Any,
    *,
    distance_strategy: str = "cosine",
) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        embedding=embeddings,
        engine=engine,
        schema_name=table.schema,
        table_name=table.name,
        embedding_dimension=3,
        distance_strategy=distance_strategy,
    )


def _ids(documents: list[Document]) -> list[str]:
    return [str(document.id) for document in documents]


def _seed_dense(store: GaussDBVectorStore) -> None:
    assert store.add_texts(
        ["alpha exact", "beta nearby", "gamma remote"],
        metadatas=[
            {"slot": "alpha"},
            {"slot": "beta"},
            {"slot": "gamma"},
        ],
        ids=["alpha", "beta", "gamma"],
    ) == ["alpha", "beta", "gamma"]


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_fast
def test_sync_write_read_delete_public_matrix(
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _store(
        writable_engine,
        deterministic_embeddings,
        temporary_vector_table,
    )
    assert store.add_texts(
        ["alpha", "beta"],
        metadatas=[{"revision": 1}, {"revision": 1}],
        ids=["a", "b"],
    ) == ["a", "b"]
    assert store.add_documents(
        [Document(id="c", page_content="gamma", metadata={"revision": 1})]
    ) == ["c"]

    assert sorted(_ids(store.get_by_ids(["missing", "c", "a"]))) == ["a", "c"]
    assert store.get_by_ids([]) == []

    assert store.add_texts(
        ["alpha-upserted"],
        metadatas=[{"revision": 2}],
        ids=["a"],
    ) == ["a"]
    updated = store.get_by_ids(["a"])
    assert [(doc.id, doc.page_content, doc.metadata) for doc in updated] == [
        ("a", "alpha-upserted", {"revision": 2})
    ]

    assert store.delete(ids=["missing"]) is True
    assert store.delete(ids=[]) is True
    assert store.delete(ids=["b"]) is True
    assert sorted(_ids(store.get_by_ids(["a", "b", "c"]))) == ["a", "c"]


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_async_write_read_delete_public_matrix(
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _store(
        writable_engine,
        deterministic_embeddings,
        temporary_vector_table,
    )
    assert await store.aadd_texts(
        ["alpha", "beta"],
        metadatas=[{"revision": 1}, {"revision": 1}],
        ids=["a", "b"],
    ) == ["a", "b"]
    assert await store.aadd_documents(
        [Document(id="c", page_content="gamma", metadata={"revision": 1})]
    ) == ["c"]
    assert sorted(_ids(await store.aget_by_ids(["missing", "c", "a"]))) == [
        "a",
        "c",
    ]
    assert await store.aget_by_ids([]) == []

    assert await store.aadd_texts(
        ["alpha-upserted"],
        metadatas=[{"revision": 2}],
        ids=["a"],
    ) == ["a"]
    updated = await store.aget_by_ids(["a"])
    assert [(doc.id, doc.page_content, doc.metadata) for doc in updated] == [
        ("a", "alpha-upserted", {"revision": 2})
    ]

    assert await store.adelete(ids=["missing"]) is True
    assert await store.adelete(ids=[]) is True
    assert await store.adelete(ids=["b"]) is True
    assert sorted(_ids(await store.aget_by_ids(["a", "b", "c"]))) == ["a", "c"]
    assert deterministic_embeddings.calls["embed_documents"] == 3
    assert deterministic_embeddings.calls["aembed_documents"] == 0


@pytest.mark.parametrize("distance_strategy", DISTANCE_STRATEGIES)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_fast
@pytest.mark.gaussdb_e2e_full
def test_sync_dense_search_public_matrix(
    distance_strategy: str,
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    fixed_embeddings = _FixedPublicEmbeddings()
    store = _store(
        writable_engine,
        fixed_embeddings,
        temporary_vector_table,
        distance_strategy=distance_strategy,
    )
    _seed_dense(store)

    expected_ids = DENSE_EXPECTED_IDS[distance_strategy]
    mmr_expected_ids = MMR_EXPECTED_IDS[distance_strategy]
    no_score = store.similarity_search("alpha exact", k=3)
    assert _ids(no_score) == expected_ids

    raw_score = store.similarity_search_with_score("alpha exact", k=3)
    assert [str(document.id) for document, _ in raw_score] == expected_ids
    assert [score for _, score in raw_score] == sorted(score for _, score in raw_score)
    assert raw_score[0][0].id == "alpha"
    assert raw_score[0][1] == pytest.approx(0.0, abs=1e-6)

    relevance = store.similarity_search_with_relevance_scores("alpha exact", k=3)
    assert [str(document.id) for document, _ in relevance] == expected_ids
    assert relevance[0][0].id == "alpha"
    assert all(0.0 <= score <= 1.0 for _, score in relevance)
    assert relevance[0][1] == pytest.approx(1.0, abs=1e-6)

    vector = fixed_embeddings.embed_query("alpha exact")
    before = dict(fixed_embeddings.calls)
    by_vector = store.similarity_search_by_vector(vector, k=3)
    by_vector_score = store.similarity_search_with_score_by_vector(vector, k=3)
    by_vector_mmr = store.max_marginal_relevance_search_by_vector(
        vector,
        k=2,
        fetch_k=3,
        lambda_mult=0.7,
    )
    assert fixed_embeddings.calls == before
    assert _ids(by_vector) == expected_ids
    assert [str(document.id) for document, _ in by_vector_score] == expected_ids
    assert _ids(by_vector_mmr) == mmr_expected_ids
    assert len(set(_ids(by_vector_mmr))) == len(mmr_expected_ids)

    mmr = store.max_marginal_relevance_search(
        "alpha exact",
        k=2,
        fetch_k=3,
        lambda_mult=0.7,
    )
    assert _ids(mmr) == mmr_expected_ids
    assert len(set(_ids(mmr))) == len(mmr_expected_ids)


@pytest.mark.parametrize("distance_strategy", DISTANCE_STRATEGIES)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_async_dense_search_public_matrix(
    distance_strategy: str,
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    fixed_embeddings = _FixedPublicEmbeddings()
    store = _store(
        writable_engine,
        fixed_embeddings,
        temporary_vector_table,
        distance_strategy=distance_strategy,
    )
    assert await store.aadd_texts(
        ["alpha exact", "beta nearby", "gamma remote"],
        metadatas=[{"slot": "alpha"}, {"slot": "beta"}, {"slot": "gamma"}],
        ids=["alpha", "beta", "gamma"],
    ) == ["alpha", "beta", "gamma"]

    expected_ids = DENSE_EXPECTED_IDS[distance_strategy]
    mmr_expected_ids = MMR_EXPECTED_IDS[distance_strategy]
    no_score = await store.asimilarity_search("alpha exact", k=3)
    assert _ids(no_score) == expected_ids
    raw_score = await store.asimilarity_search_with_score("alpha exact", k=3)
    assert [str(document.id) for document, _ in raw_score] == expected_ids
    assert [score for _, score in raw_score] == sorted(score for _, score in raw_score)
    assert raw_score[0][0].id == "alpha"
    assert raw_score[0][1] == pytest.approx(0.0, abs=1e-6)
    relevance = await store.asimilarity_search_with_relevance_scores(
        "alpha exact",
        k=3,
    )
    assert [str(document.id) for document, _ in relevance] == expected_ids
    assert relevance[0][0].id == "alpha"
    assert all(0.0 <= score <= 1.0 for _, score in relevance)
    assert relevance[0][1] == pytest.approx(1.0, abs=1e-6)

    vector = await fixed_embeddings.aembed_query("alpha exact")
    before = dict(fixed_embeddings.calls)
    by_vector = await store.asimilarity_search_by_vector(vector, k=3)
    by_vector_mmr = await store.amax_marginal_relevance_search_by_vector(
        vector,
        k=2,
        fetch_k=3,
        lambda_mult=0.7,
    )
    assert fixed_embeddings.calls == before
    assert _ids(by_vector) == expected_ids
    assert _ids(by_vector_mmr) == mmr_expected_ids
    assert len(set(_ids(by_vector_mmr))) == len(mmr_expected_ids)

    mmr = await store.amax_marginal_relevance_search(
        "alpha exact",
        k=2,
        fetch_k=3,
        lambda_mult=0.7,
    )
    assert _ids(mmr) == mmr_expected_ids
    assert len(set(_ids(mmr))) == len(mmr_expected_ids)
    assert fixed_embeddings.calls == {
        "embed_documents": 1,
        "aembed_documents": 0,
        "embed_query": 4,
        "aembed_query": 1,
    }


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_standard_search_and_asearch_dispatch_matrix(
    writable_engine: GaussDBEngine,
    temporary_vector_table: Any,
    requires_gsdiskann: bool,
) -> None:
    fixed_embeddings = _FixedPublicEmbeddings()
    store = _store(
        writable_engine,
        fixed_embeddings,
        temporary_vector_table,
    )
    _seed_dense(store)

    assert _ids(store.search("alpha exact", "similarity", k=2)) == ["alpha", "beta"]
    threshold = store.search(
        "alpha exact",
        "similarity_score_threshold",
        k=3,
        score_threshold=0.99,
    )
    assert _ids(threshold) == ["alpha"]
    assert (
        _ids(
            store.search(
                "alpha exact",
                "mmr",
                k=2,
                fetch_k=3,
                lambda_mult=0.7,
            )
        )
        == MMR_EXPECTED_IDS["cosine"]
    )

    before_async = dict(fixed_embeddings.calls)
    assert _ids(await store.asearch("alpha exact", "similarity", k=2)) == [
        "alpha",
        "beta",
    ]
    async_threshold = await store.asearch(
        "alpha exact",
        "similarity_score_threshold",
        k=3,
        score_threshold=0.99,
    )
    assert _ids(async_threshold) == ["alpha"]
    assert (
        _ids(
            await store.asearch(
                "alpha exact",
                "mmr",
                k=2,
                fetch_k=3,
                lambda_mult=0.7,
            )
        )
        == MMR_EXPECTED_IDS["cosine"]
    )
    assert fixed_embeddings.calls["embed_documents"] == before_async["embed_documents"]
    assert fixed_embeddings.calls["embed_query"] == before_async["embed_query"] + 3
    assert (
        fixed_embeddings.calls["aembed_documents"] == before_async["aembed_documents"]
    )
    assert fixed_embeddings.calls["aembed_query"] == before_async["aembed_query"]

    before = dict(fixed_embeddings.calls)
    with pytest.raises(ValueError, match="search_type"):
        store.search("must-not-embed", "unknown-search-type")
    with pytest.raises(ValueError, match="search_type"):
        await store.asearch("must-not-embed", "unknown-search-type")
    assert fixed_embeddings.calls == before


def _factory_options(lifecycle: str) -> dict[str, bool]:
    assert lifecycle == "automatic"
    return {}


def _registered_factory_table(
    resource_registry: Any,
    e2e_namespace: Any,
    temporary_schema: str,
    role: str,
) -> Any:
    return e2e_namespace.table(temporary_schema, role)


def _require_factory_capability(
    lifecycle: str,
    capability_snapshot: dict[str, object],
) -> None:
    assert lifecycle == "automatic"
    access_methods = capability_snapshot["access_methods"]
    missing = {"gsdiskann"}.difference(access_methods)
    if missing:
        pytest.skip(f"GaussDB capabilities are unavailable: {sorted(missing)}")


def _assert_factory_catalog_state(
    engine: GaussDBEngine,
    *,
    schema_name: str,
    table_name: str,
    lifecycle: str,
) -> None:
    columns = engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT cols.column_name, cols.is_nullable, cols.udt_name, "
                "format_type(attr.atttypid, attr.atttypmod) "
                "FROM information_schema.columns AS cols "
                "JOIN pg_class AS rel ON rel.relname = cols.table_name "
                "JOIN pg_namespace AS ns ON ns.oid = rel.relnamespace "
                "AND ns.nspname = cols.table_schema "
                "JOIN pg_attribute AS attr ON attr.attrelid = rel.oid "
                "AND attr.attname = cols.column_name "
                "WHERE cols.table_schema = %s AND cols.table_name = %s "
                "ORDER BY cols.ordinal_position"
            ),
            (schema_name, table_name),
        ),
        operation="verify factory table catalog state",
    )
    by_name = {str(row[0]): row for row in columns}
    assert set(by_name) == {"id", "content", "metadata", "embedding"}
    assert all(str(row[1]).upper() == "NO" for row in by_name.values())
    assert str(by_name["id"][2]).lower() == "text"
    assert str(by_name["content"][2]).lower() == "text"
    assert str(by_name["metadata"][2]).lower() == "jsonb"
    assert str(by_name["embedding"][3]).lower() == "floatvector(3)"

    indexes = engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT am.amname, idx.indisvalid, idx.indisready, idx.indisusable, "
                "pg_get_indexdef(idx.indexrelid) "
                "FROM pg_index AS idx "
                "JOIN pg_class AS tab ON tab.oid = idx.indrelid "
                "JOIN pg_namespace AS ns ON ns.oid = tab.relnamespace "
                "JOIN pg_class AS ind ON ind.oid = idx.indexrelid "
                "JOIN pg_am AS am ON am.oid = ind.relam "
                "WHERE ns.nspname = %s AND tab.relname = %s "
                "AND am.amname = 'gsdiskann'"
            ),
            (schema_name, table_name),
        ),
        operation="verify factory vector index catalog state",
    )
    assert lifecycle == "automatic"
    expected_method = "gsdiskann"
    assert len(indexes) == 1
    method, valid, ready, usable, definition = indexes[0]
    assert str(method).lower() == expected_method
    assert valid is True
    assert ready is True
    assert usable is True
    lowered_definition = str(definition).lower()
    assert table_name.lower() in lowered_definition
    assert "embedding" in lowered_definition


@pytest.mark.parametrize("factory", FACTORY_APIS[:2])
@pytest.mark.parametrize("lifecycle", FACTORY_LIFECYCLES)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_factory_sync_matrix(
    factory: str,
    lifecycle: str,
    writable_engine: GaussDBEngine,
    configure_gsdiskann_session_memory: Callable[[GaussDBEngine], None],
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    capability_snapshot: dict[str, object],
) -> None:
    _require_factory_capability(lifecycle, capability_snapshot)
    configure_gsdiskann_session_memory(writable_engine)
    table = _registered_factory_table(
        resource_registry,
        e2e_namespace,
        temporary_schema,
        f"sync_{factory}_{lifecycle}",
    )
    common = {
        "engine": writable_engine,
        "schema_name": table.schema,
        "table_name": table.name,
        "embedding_dimension": 3,
        **_factory_options(lifecycle),
    }
    if factory == "from_texts":
        store = GaussDBVectorStore.from_texts(
            ["factory alpha"],
            deterministic_embeddings,
            metadatas=[{"factory": factory}],
            ids=["factory-id"],
            **common,
        )
    else:
        store = GaussDBVectorStore.from_documents(
            [
                Document(
                    id="factory-id",
                    page_content="factory alpha",
                    metadata={"factory": factory},
                )
            ],
            deterministic_embeddings,
            **common,
        )
    assert _ids(store.get_by_ids(["factory-id"])) == ["factory-id"]
    _assert_factory_catalog_state(
        writable_engine,
        schema_name=table.schema,
        table_name=table.name,
        lifecycle=lifecycle,
    )


@pytest.mark.parametrize("factory", FACTORY_APIS[2:])
@pytest.mark.parametrize("lifecycle", FACTORY_LIFECYCLES)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_factory_async_matrix(
    factory: str,
    lifecycle: str,
    writable_engine: GaussDBEngine,
    configure_gsdiskann_session_memory: Callable[[GaussDBEngine], None],
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    capability_snapshot: dict[str, object],
) -> None:
    _require_factory_capability(lifecycle, capability_snapshot)
    configure_gsdiskann_session_memory(writable_engine)
    table = _registered_factory_table(
        resource_registry,
        e2e_namespace,
        temporary_schema,
        f"async_{factory}_{lifecycle}",
    )
    common = {
        "engine": writable_engine,
        "schema_name": table.schema,
        "table_name": table.name,
        "embedding_dimension": 3,
        **_factory_options(lifecycle),
    }
    if factory == "afrom_texts":
        store = await GaussDBVectorStore.afrom_texts(
            ["factory alpha"],
            deterministic_embeddings,
            metadatas=[{"factory": factory}],
            ids=["factory-id"],
            **common,
        )
    else:
        store = await GaussDBVectorStore.afrom_documents(
            [
                Document(
                    id="factory-id",
                    page_content="factory alpha",
                    metadata={"factory": factory},
                )
            ],
            deterministic_embeddings,
            **common,
        )
    assert _ids(await store.aget_by_ids(["factory-id"])) == ["factory-id"]
    _assert_factory_catalog_state(
        writable_engine,
        schema_name=table.schema,
        table_name=table.name,
        lifecycle=lifecycle,
    )
    assert deterministic_embeddings.calls["embed_documents"] == 1
    assert deterministic_embeddings.calls["aembed_documents"] == 0


class _SyntheticFactoryEngine:
    def __init__(self, *, close_failure: bool) -> None:
        self.closed = False
        self.close_failure = close_failure

    def close(self) -> None:
        self.closed = True
        if self.close_failure:
            raise RuntimeError("cleanup-failure")


class _SyntheticFactoryStore(GaussDBVectorStore):
    failure_phase = "write-failure"
    last_engine: _SyntheticFactoryEngine

    def __init__(self, **kwargs: Any) -> None:
        external = kwargs.get("engine")
        self._owns_engine = external is None
        self._engine = external or _SyntheticFactoryEngine(close_failure=True)
        type(self).last_engine = self._engine

    def _ensure_initialized(self) -> None:
        if self.failure_phase == "table-failure":
            raise RuntimeError("table-failure")
        if self.failure_phase == "index-failure":
            raise RuntimeError("index-failure")

    def add_texts(self, texts: Any, **kwargs: Any) -> list[str]:
        self._ensure_initialized()
        return self._synthetic_write(texts)

    def _synthetic_write(self, texts: Any) -> list[str]:
        if self.failure_phase == "write-failure":
            raise RuntimeError("write-failure")
        return ["synthetic-id" for _ in texts]


@pytest.mark.parametrize("failure_phase", FACTORY_FAILURES)
@pytest.mark.parametrize("ownership", FACTORY_OWNERSHIP)
def test_factory_primary_failure_cleanup_matrix(
    failure_phase: str,
    ownership: str,
) -> None:
    _SyntheticFactoryStore.failure_phase = failure_phase
    external = _SyntheticFactoryEngine(close_failure=True)
    kwargs: dict[str, Any] = {
        "table_name": "synthetic_table",
        "embedding_dimension": 3,
    }
    if ownership == "external":
        kwargs["engine"] = external
    with pytest.raises(RuntimeError, match=failure_phase) as caught:
        _SyntheticFactoryStore.from_texts(
            ["synthetic"],
            object(),
            ids=["synthetic-id"],
            **kwargs,
        )
    assert str(caught.value) == failure_phase
    if ownership == "owned":
        assert _SyntheticFactoryStore.last_engine.closed is True
    else:
        assert external.closed is False
        external.close_failure = False
        external.close()
        assert external.closed is True


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_factory_primary_failure_cleanup_real_smoke(
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
) -> None:
    table = _registered_factory_table(
        resource_registry,
        e2e_namespace,
        temporary_schema,
        "factory_failure_smoke",
    )
    with pytest.raises(ValueError, match="metadatas"):
        GaussDBVectorStore.from_texts(
            ["one", "two"],
            deterministic_embeddings,
            metadatas=[{"only": "one"}],
            ids=["one", "two"],
            engine=writable_engine,
            schema_name=table.schema,
            table_name=table.name,
            embedding_dimension=3,
        )
    assert writable_engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT 1")),
        operation="verify external Engine after factory failure",
    ) == [(1,)]
