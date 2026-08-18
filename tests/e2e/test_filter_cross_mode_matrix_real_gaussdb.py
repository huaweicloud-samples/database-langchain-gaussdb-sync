from __future__ import annotations

import copy
import datetime as dt
from typing import Any

import pytest
from _evidence import sanitize_exception
from langchain_core.embeddings import Embeddings

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.errors import GaussDBFilterError
from langchain_gaussdb.hybrid_search import BM25Config

FILTER_OPERATORS = (
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$nin",
    "$between",
    "$like",
    "$ilike",
    "$contains",
    "$exists",
)
LOGICAL_SHAPES = (
    "implicit-and",
    "$and",
    "$or",
    "$not",
    "double-$not",
    "nested",
    "empty",
    "unknown",
    "mixed-sibling",
    "multi-operator",
)
JSON_KINDS = (
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "time",
    "timestamp",
    "array",
    "object",
    "missing",
    "json-null",
    "empty-string",
)
TYPED_NULL_KIND = "sql-null"
RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "mmr")
HYBRID_CONTROLS = ("dense-only", "sparse-only")
ERROR_CONTRACT = ("GaussDBFilterError", "class22")

JSON_OPERATOR_CASES = (
    {
        "operator": "$eq",
        "kinds": ("string",),
        "filter": {"name": {"$eq": "Alpha"}},
        "expected_ids": {"d1"},
    },
    {
        "operator": "$ne",
        "filter": {"name": {"$ne": "Alpha"}},
        "expected_ids": {"d2", "d3"},
    },
    {
        "operator": "$gt",
        "kinds": ("integer",),
        "filter": {"rank": {"$gt": 1}},
        "expected_ids": {"d2", "d3"},
    },
    {
        "operator": "$gte",
        "filter": {"rank": {"$gte": 2}},
        "expected_ids": {"d2", "d3"},
    },
    {"operator": "$lt", "filter": {"rank": {"$lt": 3}}, "expected_ids": {"d1", "d2"}},
    {
        "operator": "$lte",
        "filter": {"rank": {"$lte": 2}},
        "expected_ids": {"d1", "d2"},
    },
    {
        "operator": "$in",
        "filter": {"name": {"$in": ["Alpha", "Beta"]}},
        "expected_ids": {"d1", "d2"},
    },
    {
        "operator": "$nin",
        "filter": {"name": {"$nin": ["Alpha"]}},
        "expected_ids": {"d2", "d3"},
    },
    {
        "operator": "$between",
        "filter": {"rank": {"$between": [1, 2]}},
        "expected_ids": {"d1", "d2"},
    },
    {
        "operator": "$like",
        "filter": {"name": {"$like": "Al%"}},
        "expected_ids": {"d1"},
    },
    {
        "operator": "$ilike",
        "filter": {"name": {"$ilike": "al%"}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "operator": "$contains",
        "kinds": ("array",),
        "filter": {"tags": {"$contains": ["red"]}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "operator": "$exists",
        "kinds": ("json-null",),
        "filter": {"name": {"$exists": True}},
        "expected_ids": {"d1", "d2", "d3", "d5"},
    },
    {
        "operator": "float-equality",
        "kinds": ("float",),
        "filter": {"score": {"$eq": 2.0}},
        "expected_ids": {"d2"},
    },
    {
        "operator": "boolean-equality",
        "kinds": ("boolean",),
        "filter": {"active": {"$eq": False}},
        "expected_ids": {"d2"},
    },
    {
        "operator": "date-python-equality",
        "kinds": ("date",),
        "filter": {"day": {"$eq": dt.date(2026, 1, 2)}},
        "expected_ids": {"d2"},
    },
    {
        "operator": "date-python-range",
        "filter": {
            "day": {
                "$between": [
                    dt.date(2026, 1, 1),
                    dt.date(2026, 1, 2),
                ]
            }
        },
        "expected_ids": {"d1", "d2"},
    },
    {
        "operator": "date-python-membership",
        "filter": {
            "day": {
                "$in": [
                    dt.date(2026, 1, 1),
                    dt.date(2026, 1, 3),
                ]
            }
        },
        "expected_ids": {"d1", "d3"},
    },
    {
        "operator": "date-text-range",
        "filter": {"day": {"$between": ["2026-01-01", "2026-01-02"]}},
        "expected_ids": {"d1", "d2"},
    },
    {
        "operator": "time-text-range",
        "kinds": ("time",),
        "filter": {"clock": {"$gte": "12:00:00"}},
        "expected_ids": {"d2", "d3"},
    },
    {
        "operator": "timestamp-text-range",
        "kinds": ("timestamp",),
        "filter": {"moment": {"$lt": "2026-01-03 00:00:00"}},
        "expected_ids": {"d1", "d2"},
    },
    {
        "operator": "object-contains",
        "kinds": ("object",),
        "filter": {"profile": {"$contains": {"team": "a"}}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "operator": "missing-key",
        "kinds": ("missing",),
        "filter": {"name": {"$exists": False}},
        "expected_ids": {"d4"},
    },
    {
        "operator": "empty-string-equality",
        "kinds": ("empty-string",),
        "filter": {"blank": {"$eq": ""}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "operator": "boolean-membership",
        "filter": {"active": {"$in": [True]}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "operator": "boolean-negative-membership",
        "filter": {"active": {"$nin": [True]}},
        "expected_ids": {"d2"},
    },
    {
        "operator": "array-negative",
        "filter": {"$not": {"tags": {"$contains": ["red"]}}},
        "expected_ids": {"d2", "d4", "d5"},
    },
    {
        "operator": "object-negative",
        "filter": {"$not": {"profile": {"$contains": {"team": "a"}}}},
        "expected_ids": {"d2", "d4", "d5"},
    },
    {
        "operator": "$nin-empty",
        "filter": {"name": {"$nin": []}},
        "expected_ids": {"d1", "d2", "d3", "d4", "d5"},
    },
)

LOGICAL_CASES = (
    {
        "shape": "implicit-and",
        "filter": {"active": True, "rank": {"$gte": 2}},
        "expected_ids": {"d3"},
    },
    {
        "shape": "$and",
        "filter": {
            "$and": [
                {"active": {"$eq": True}},
                {"rank": {"$gte": 2}},
            ]
        },
        "expected_ids": {"d3"},
    },
    {
        "shape": "$or",
        "filter": {
            "$or": [
                {"name": {"$eq": "Alpha"}},
                {"name": {"$eq": "Beta"}},
                {"name": {"$eq": "alphabet"}},
            ]
        },
        "expected_ids": {"d1", "d2", "d3"},
    },
    {
        "shape": "$not",
        "filter": {"$not": {"name": {"$eq": "Alpha"}}},
        "expected_ids": {"d2", "d3"},
    },
    {
        "shape": "double-$not",
        "filter": {"$not": {"$not": {"name": {"$eq": "Alpha"}}}},
        "expected_ids": {"d1"},
    },
    {
        "shape": "nested",
        "filter": {
            "$and": [
                {
                    "$or": [
                        {"name": {"$eq": "Alpha"}},
                        {"name": {"$eq": "Beta"}},
                    ]
                },
                {"rank": {"$gte": 2}},
            ]
        },
        "expected_ids": {"d2"},
    },
    {
        "shape": "empty",
        "filter": {},
        "expected_ids": {"d1", "d2", "d3", "d4", "d5"},
    },
)

INVALID_SHAPES = (
    {
        "shape": "unknown",
        "filter": {"name": {"$regex": "^A"}},
        "expected_error": "GaussDBFilterError",
    },
    {
        "shape": "mixed-sibling",
        "filter": {"$and": [{"name": "Alpha"}], "rank": 1},
        "expected_error": "GaussDBFilterError",
    },
    {
        "shape": "multi-operator",
        "filter": {"rank": {"$gt": 1, "$lt": 3}},
        "expected_error": "GaussDBFilterError",
    },
)

PRECEDENCE_CASES = (
    {
        "case": "wrong-value-type",
        "api": "similarity_search",
        "kwargs": {"filter": {"rank": {"$in": "not-a-list"}}},
        "expected_error": "GaussDBFilterError",
    },
    {
        "case": "invalid-k",
        "api": "similarity_search",
        "kwargs": {"k": -1},
        "expected_error": "ValueError",
    },
    {
        "case": "invalid-fetch-k",
        "api": "max_marginal_relevance_search",
        "kwargs": {"k": 1, "fetch_k": -1},
        "expected_error": "ValueError",
    },
    {
        "case": "invalid-bm25-query",
        "api": "similarity_search",
        "kwargs": {"bm25_query": 7},
        "retrieval_mode": "bm25",
        "expected_error": "ValueError",
    },
    {
        "case": "typed-rank-bool",
        "api": "similarity_search",
        "kwargs": {"filter": {"rank": {"$eq": True}}},
        "metadata_indexes": {"rank": "bigint"},
        "expected_error": "GaussDBFilterError",
    },
    {
        "case": "typed-boolean-range",
        "api": "similarity_search",
        "kwargs": {"filter": {"active": {"$gt": False}}},
        "metadata_indexes": {"active": "boolean"},
        "expected_error": "GaussDBFilterError",
    },
    {
        "case": "typed-date-like",
        "api": "similarity_search",
        "kwargs": {"filter": {"day": {"$like": "2026%"}}},
        "metadata_indexes": {"day": "date"},
        "expected_error": "GaussDBFilterError",
    },
)

FILTER_SHAPE_CASES = (
    {
        "family": "equality",
        "filter": {"name": {"$eq": "Alpha"}},
        "expected_ids": {"d1"},
    },
    {
        "family": "membership",
        "filter": {"name": {"$in": ["Alpha", "Beta"]}},
        "expected_ids": {"d1", "d2"},
    },
    {
        "family": "range",
        "filter": {"rank": {"$between": [2, 3]}},
        "expected_ids": {"d2", "d3"},
    },
    {
        "family": "text",
        "filter": {"name": {"$ilike": "al%"}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "family": "contains",
        "filter": {"tags": {"$contains": ["red"]}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "family": "negative-contains",
        "filter": {"$not": {"tags": {"$contains": ["red"]}}},
        "expected_ids": {"d2", "d4", "d5"},
    },
    {
        "family": "exists",
        "filter": {"name": {"$exists": True}},
        "expected_ids": {"d1", "d2", "d3", "d5"},
    },
    {
        "family": "logical",
        "filter": {
            "$and": [
                {"active": True},
                {"rank": {"$gte": 2}},
            ]
        },
        "expected_ids": {"d3"},
    },
)

TYPED_TEXT_CASES = (
    {"filter": {"label": {"$eq": ""}}, "expected_ids": {"empty"}},
    {"filter": {"label": {"$ne": "x"}}, "expected_ids": {"empty"}},
    {"filter": {"label": {"$in": ["", "x"]}}, "expected_ids": {"empty", "x"}},
    {"filter": {"label": {"$nin": ["x"]}}, "expected_ids": {"empty"}},
    {
        "filter": {"label": {"$nin": []}},
        "expected_ids": {"empty", "missing", "null", "x"},
    },
    {
        "filter": {"label": {"$exists": True}},
        "expected_ids": {"empty", "null", "x"},
    },
    {
        "filter": {"label": {"$exists": False}},
        "expected_ids": {"missing"},
    },
    {
        "filter": {"label": {"$gt": ""}},
        "expected_ids": {"x"},
    },
    {"filter": {"label": {"$like": "x%"}}, "expected_ids": {"x"}},
    {"filter": {"label": {"$ilike": "X"}}, "expected_ids": {"x"}},
)
TYPED_TEXT_IDS = (
    "eq-empty",
    "ne-json-selector",
    "in-empty-and-value",
    "nin-json-selector",
    "nin-empty-is-true",
    "exists-present",
    "exists-missing-or-null",
    "gt-empty",
    "like-prefix",
    "ilike-value",
)

TYPED_NUMERIC_BOOLEAN_CASES = (
    {"filter": {"rank": {"$eq": 2}}, "expected_ids": {"d2"}},
    {"filter": {"rank": {"$ne": 2}}, "expected_ids": {"d1", "d3"}},
    {"filter": {"rank": {"$gt": 1}}, "expected_ids": {"d2", "d3"}},
    {"filter": {"rank": {"$gte": 2}}, "expected_ids": {"d2", "d3"}},
    {"filter": {"rank": {"$lt": 3}}, "expected_ids": {"d1", "d2"}},
    {"filter": {"rank": {"$lte": 2}}, "expected_ids": {"d1", "d2"}},
    {"filter": {"rank": {"$in": [1, 3]}}, "expected_ids": {"d1", "d3"}},
    {"filter": {"rank": {"$nin": [1, 3]}}, "expected_ids": {"d2"}},
    {"filter": {"score": {"$between": [1.0, 2.0]}}, "expected_ids": {"d1", "d2"}},
    {"filter": {"active": {"$eq": True}}, "expected_ids": {"d1", "d3"}},
    {"filter": {"active": {"$ne": True}}, "expected_ids": {"d2"}},
    {"filter": {"active": {"$in": [True]}}, "expected_ids": {"d1", "d3"}},
    {"filter": {"active": {"$nin": [True]}}, "expected_ids": {"d2"}},
    {"filter": {"active": {"$exists": False}}, "expected_ids": {"d4"}},
)
TYPED_NUMERIC_BOOLEAN_IDS = (
    "integer-eq",
    "integer-ne",
    "integer-gt",
    "integer-gte",
    "integer-lt",
    "integer-lte",
    "integer-in",
    "integer-nin",
    "float-between",
    "boolean-eq",
    "boolean-ne",
    "boolean-in",
    "boolean-nin",
    "boolean-exists-false",
)

TYPED_TEMPORAL_CASES = (
    {
        "filter": {"day": {"$eq": dt.date(2026, 1, 2)}},
        "expected_ids": {"d2"},
    },
    {
        "filter": {
            "day": {
                "$between": [
                    dt.date(2026, 1, 1),
                    dt.date(2026, 1, 2),
                ]
            }
        },
        "expected_ids": {"d1", "d2"},
    },
    {
        "filter": {"day": {"$in": [dt.date(2026, 1, 1), dt.date(2026, 1, 3)]}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "filter": {"clock": {"$gte": dt.time(12, 0, 0)}},
        "expected_ids": {"d2", "d3"},
    },
    {
        "filter": {"clock": {"$in": [dt.time(8, 0, 0), dt.time(18, 0, 0)]}},
        "expected_ids": {"d1", "d3"},
    },
    {
        "filter": {"clock": {"$between": [dt.time(8, 0, 0), dt.time(12, 0, 0)]}},
        "expected_ids": {"d1", "d2"},
    },
    {
        "filter": {
            "moment": {
                "$lt": dt.datetime(2026, 1, 3, 0, 0, 0),
            }
        },
        "expected_ids": {"d1", "d2"},
    },
    {
        "filter": {
            "moment": {
                "$in": [
                    dt.datetime(2026, 1, 1, 8, 0, 0),
                    dt.datetime(2026, 1, 3, 18, 0, 0),
                ]
            }
        },
        "expected_ids": {"d1", "d3"},
    },
    {
        "filter": {
            "moment": {
                "$between": [
                    dt.datetime(2026, 1, 1, 8, 0, 0),
                    dt.datetime(2026, 1, 2, 12, 0, 0),
                ]
            }
        },
        "expected_ids": {"d1", "d2"},
    },
)
TYPED_TEMPORAL_IDS = (
    "date-eq",
    "date-between",
    "date-in",
    "time-gte",
    "time-in",
    "time-between",
    "timestamp-lt",
    "timestamp-in",
    "timestamp-between",
)


def _new_store(
    *,
    engine: GaussDBEngine,
    embeddings: Any,
    resource_registry: Any,
    e2e_namespace: Any,
    schema_name: str,
    role: str,
    retrieval_mode: str = "dense",
    metadata_indexes: dict[str, str | None] | None = None,
    bm25_config: BM25Config | None = None,
) -> GaussDBVectorStore:
    table = e2e_namespace.table(schema_name, role)
    return GaussDBVectorStore(
        embedding=embeddings,
        engine=engine,
        schema_name=table.schema,
        table_name=table.name,
        embedding_dimension=3,
        retrieval_mode=retrieval_mode,
        metadata_indexes=metadata_indexes,
        bm25_config=bm25_config,
    )


def _seed_json(store: GaussDBVectorStore) -> None:
    metadatas = [
        {
            "name": "Alpha",
            "rank": 1,
            "score": 1.5,
            "active": True,
            "day": "2026-01-01",
            "clock": "08:00:00",
            "moment": "2026-01-01 08:00:00",
            "tags": ["red", "blue"],
            "profile": {"team": "a"},
            "blank": "",
        },
        {
            "name": "Beta",
            "rank": 2,
            "score": 2.0,
            "active": False,
            "day": "2026-01-02",
            "clock": "12:00:00",
            "moment": "2026-01-02 12:00:00",
            "tags": ["blue"],
            "profile": {"team": "b"},
            "blank": "filled",
        },
        {
            "name": "alphabet",
            "rank": 3,
            "score": 3.5,
            "active": True,
            "day": "2026-01-03",
            "clock": "18:00:00",
            "moment": "2026-01-03 18:00:00",
            "tags": ["red"],
            "profile": {"team": "a"},
            "blank": "",
        },
        {"other": "missing"},
        {"name": None, "other": "json-null"},
    ]
    assert store.add_texts(
        [f"shared lexical document {index}" for index in range(1, 6)],
        metadatas=metadatas,
        ids=["d1", "d2", "d3", "d4", "d5"],
    ) == ["d1", "d2", "d3", "d4", "d5"]


def _seed_hybrid_leak_controls(
    store: GaussDBVectorStore,
    *,
    family: str,
) -> None:
    keep = {
        "name": "Alpha",
        "rank": 2,
        "active": True,
        "tags": ["red"],
    }
    leak = {
        "name": "Leak",
        "rank": 99,
        "active": False,
        "tags": ["black"],
    }
    if family == "exists":
        leak.pop("name")
    elif family == "negative-contains":
        keep["tags"] = ["black"]
        leak["tags"] = ["red"]
    assert store.add_texts(
        ["keep semantically remote", "hybrid target", "sparse remote"],
        metadatas=[keep, dict(leak), dict(leak)],
        ids=["keep", "dense-leak", "sparse-leak"],
        text_lemmatized_values=[
            "hybrid",
            "unrelated dense token",
            "hybrid target hybrid target",
        ],
    ) == ["keep", "dense-leak", "sparse-leak"]


def _result_ids(
    store: GaussDBVectorStore,
    filter_value: dict[str, Any],
    *,
    query: str = "shared lexical",
) -> set[str]:
    return {
        str(document.id)
        for document in store.similarity_search(
            query,
            k=20,
            filter=filter_value,
        )
    }


def _require_mode_capabilities(
    mode: str,
    capability_snapshot: dict[str, object],
) -> None:
    access_methods = capability_snapshot["access_methods"]
    if mode in {"dense", "hybrid", "mmr"} and "gsdiskann" not in access_methods:
        pytest.skip("GaussDB capability gsdiskann is explicitly unavailable")
    if mode in {"bm25", "hybrid"} and capability_snapshot["bm25"] is not True:
        pytest.skip("GaussDB capability bm25 is explicitly unavailable")


@pytest.mark.parametrize(
    "case",
    JSON_OPERATOR_CASES,
    ids=lambda case: str(case["operator"]),
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_dense_jsonb_operator_truth_table(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="json_operators",
    )
    _seed_json(store)
    assert _result_ids(store, case["filter"]) == case["expected_ids"]


@pytest.mark.parametrize(
    "case",
    (*LOGICAL_CASES, *INVALID_SHAPES),
    ids=lambda case: str(case["shape"]),
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_dense_logical_shape_truth_table(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="logical_shapes",
    )
    _seed_json(store)
    original_filter = copy.deepcopy(case["filter"])
    if "expected_ids" in case:
        assert _result_ids(store, case["filter"]) == case["expected_ids"]
    else:
        before = dict(deterministic_embeddings.calls)
        with pytest.raises(GaussDBFilterError):
            store.similarity_search(
                "must fail before embedding",
                k=20,
                filter=case["filter"],
            )
        assert deterministic_embeddings.calls == before
    assert case["filter"] == original_filter


@pytest.mark.parametrize(
    "case",
    TYPED_TEXT_CASES,
    ids=TYPED_TEXT_IDS,
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_typed_text_null_empty_truth_table(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="typed_text",
        metadata_indexes={"label": "text"},
    )
    assert store.add_texts(
        ["label x", "label empty", "label missing", "label null"],
        metadatas=[{"label": "x"}, {"label": ""}, {}, {"label": None}],
        ids=["x", "empty", "missing", "null"],
    ) == ["x", "empty", "missing", "null"]
    assert _result_ids(store, case["filter"]) == case["expected_ids"]


@pytest.mark.parametrize(
    "case",
    TYPED_NUMERIC_BOOLEAN_CASES,
    ids=TYPED_NUMERIC_BOOLEAN_IDS,
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_typed_numeric_boolean_operator_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="typed_numeric_boolean",
        metadata_indexes={
            "rank": "bigint",
            "score": "float",
            "active": "boolean",
        },
    )
    assert store.add_texts(
        ["one", "two", "three", "missing", "null"],
        metadatas=[
            {"rank": 1, "score": 1.0, "active": True},
            {"rank": 2, "score": 2.0, "active": False},
            {"rank": 3, "score": 3.5, "active": True},
            {},
            {"rank": None, "score": None, "active": None},
        ],
        ids=["d1", "d2", "d3", "d4", "d5"],
    ) == ["d1", "d2", "d3", "d4", "d5"]
    assert _result_ids(store, case["filter"]) == case["expected_ids"]


@pytest.mark.parametrize("case", TYPED_TEMPORAL_CASES, ids=TYPED_TEMPORAL_IDS)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_typed_temporal_operator_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="typed_temporal",
        metadata_indexes={"day": "date"},
    )
    assert store.add_texts(
        ["one", "two", "three"],
        metadatas=[
            {
                "day": "2026-01-01",
                "clock": "08:00:00",
                "moment": "2026-01-01 08:00:00",
            },
            {
                "day": "2026-01-02",
                "clock": "12:00:00",
                "moment": "2026-01-02 12:00:00",
            },
            {
                "day": "2026-01-03",
                "clock": "18:00:00",
                "moment": "2026-01-03 18:00:00",
            },
        ],
        ids=["d1", "d2", "d3"],
    ) == ["d1", "d2", "d3"]
    assert _result_ids(store, case["filter"]) == case["expected_ids"]


@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_invalid_stored_date_preserves_sqlstate_and_recovers(
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    requires_gsdiskann: bool,
) -> None:
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="invalid_date",
    )
    assert store.add_texts(
        ["valid", "invalid"],
        metadatas=[
            {"published": "2026-01-01"},
            {"published": "2026-02-31"},
        ],
        ids=["valid", "invalid"],
    ) == ["valid", "invalid"]

    with pytest.raises(GaussDBFilterError) as caught:
        store.similarity_search(
            "date",
            k=10,
            filter={"published": {"$gte": dt.date(2026, 1, 1)}},
        )
    assert caught.value.sqlstate is not None
    assert caught.value.sqlstate.startswith("22")
    evidence = sanitize_exception(caught.value)
    assert evidence["exception_class"] == "GaussDBFilterError"
    assert evidence["sqlstate"] == caught.value.sqlstate
    fragment = str(evidence["fragment"]).lower()
    for forbidden in (
        "2026-02-31",
        "://",
        "host=",
        "hostname=",
        "user=",
        "username=",
        "password=",
        "passwd=",
    ):
        assert forbidden not in fragment
    assert _result_ids(store, {"published": {"$eq": "2026-01-01"}}) == {"valid"}


@pytest.mark.parametrize(
    "case",
    FILTER_SHAPE_CASES,
    ids=lambda case: str(case["family"]),
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_bm25_filter_shape_family_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    capability_snapshot: dict[str, object],
) -> None:
    _require_mode_capabilities("bm25", capability_snapshot)
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="bm25_filter",
        retrieval_mode="bm25",
    )
    _seed_json(store)
    assert _result_ids(store, case["filter"]) == case["expected_ids"]


@pytest.mark.parametrize(
    "case",
    FILTER_SHAPE_CASES,
    ids=lambda case: str(case["family"]),
)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
def test_hybrid_filter_shape_family_matrix(
    case: dict[str, Any],
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    capability_snapshot: dict[str, object],
) -> None:
    _require_mode_capabilities("hybrid", capability_snapshot)
    bm25_config = BM25Config(column="lexical")
    hybrid = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role="hybrid_filter",
        retrieval_mode="hybrid",
        bm25_config=bm25_config,
    )
    _seed_hybrid_leak_controls(hybrid, family=str(case["family"]))

    dense_only = GaussDBVectorStore(
        embedding=deterministic_embeddings,
        engine=writable_engine,
        schema_name=temporary_schema,
        table_name=e2e_namespace.name("hybrid_filter"),
        embedding_dimension=3,
        retrieval_mode="dense",
        bm25_config=bm25_config,
    )
    sparse_only = GaussDBVectorStore(
        embedding=deterministic_embeddings,
        engine=writable_engine,
        schema_name=temporary_schema,
        table_name=e2e_namespace.name("hybrid_filter"),
        embedding_dimension=3,
        retrieval_mode="bm25",
        bm25_config=bm25_config,
    )
    controls = {
        "dense-only": (dense_only, "dense-leak"),
        "sparse-only": (sparse_only, "sparse-leak"),
    }
    for control in HYBRID_CONTROLS:
        control_store, expected_leak = controls[control]
        assert [
            str(document.id)
            for document in control_store.similarity_search("hybrid target", k=1)
        ] == [expected_leak]

    expected_ids = {"keep"}
    assert (
        _result_ids(dense_only, case["filter"], query="hybrid target") == expected_ids
    )
    assert (
        _result_ids(sparse_only, case["filter"], query="hybrid target") == expected_ids
    )
    assert _result_ids(hybrid, case["filter"], query="hybrid target") == expected_ids
    assert {"dense-leak", "sparse-leak"}.isdisjoint(expected_ids)


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
@pytest.mark.gaussdb_e2e
@pytest.mark.gaussdb_e2e_full
@pytest.mark.asyncio
async def test_async_filter_shape_family_matrix(
    mode: str,
    writable_engine: GaussDBEngine,
    deterministic_embeddings: Any,
    temporary_schema: str,
    resource_registry: Any,
    e2e_namespace: Any,
    capability_snapshot: dict[str, object],
) -> None:
    _require_mode_capabilities(mode, capability_snapshot)
    retrieval_mode = "dense" if mode == "mmr" else mode
    store = _new_store(
        engine=writable_engine,
        embeddings=deterministic_embeddings,
        resource_registry=resource_registry,
        e2e_namespace=e2e_namespace,
        schema_name=temporary_schema,
        role=f"async_{mode}",
        retrieval_mode=retrieval_mode,
    )
    before_add = dict(deterministic_embeddings.calls)
    assert await store.aadd_texts(
        [f"shared lexical document {index}" for index in range(1, 4)],
        metadatas=[
            {"tenant": "keep", "rank": 1},
            {"tenant": "drop", "rank": 2},
            {"tenant": "keep", "rank": 3},
        ],
        ids=["d1", "d2", "d3"],
    ) == ["d1", "d2", "d3"]
    after_add = deterministic_embeddings.calls
    assert after_add["embed_documents"] == before_add["embed_documents"] + 1
    assert after_add["aembed_documents"] == before_add["aembed_documents"]
    filter_value = {
        "$and": [
            {"tenant": {"$eq": "keep"}},
            {"rank": {"$gte": 1}},
        ]
    }
    expected_ids = {"d1", "d3"}
    sync_ids = _result_ids(store, filter_value)
    before_async = dict(deterministic_embeddings.calls)
    if mode == "mmr":
        documents = await store.amax_marginal_relevance_search(
            "shared lexical",
            k=3,
            fetch_k=3,
            lambda_mult=0.7,
            filter=filter_value,
        )
    else:
        documents = await store.asimilarity_search(
            "shared lexical",
            k=3,
            filter=filter_value,
        )
    assert {str(document.id) for document in documents} == expected_ids
    assert sync_ids == expected_ids
    after_async = deterministic_embeddings.calls
    assert after_async["embed_documents"] == before_async["embed_documents"]
    assert after_async["aembed_documents"] == before_async["aembed_documents"]
    if mode == "bm25":
        assert after_async["embed_query"] == before_async["embed_query"]
    else:
        assert after_async["embed_query"] == before_async["embed_query"] + 1
    assert after_async["aembed_query"] == before_async["aembed_query"]


class _FailIfEmbedded(Embeddings):
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise AssertionError("invalid filter reached embedding")

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        raise AssertionError("invalid filter reached embedding")


class _FailIfCatalogued:
    def __init__(self) -> None:
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        self.calls += 1
        raise AssertionError(f"invalid filter reached DDL/catalog through {name}")


@pytest.mark.parametrize("case", PRECEDENCE_CASES, ids=lambda case: str(case["case"]))
def test_invalid_filter_precedes_embedding_lifecycle_capability(
    case: dict[str, Any],
) -> None:
    embeddings = _FailIfEmbedded()
    engine = _FailIfCatalogued()
    store = GaussDBVectorStore(
        embedding=embeddings,
        engine=engine,
        table_name="synthetic_filter_precedence",
        embedding_dimension=3,
        retrieval_mode=str(case.get("retrieval_mode", "dense")),
        metadata_indexes=case.get("metadata_indexes"),
    )
    expected_error = {
        "GaussDBFilterError": GaussDBFilterError,
        "ValueError": ValueError,
    }[str(case["expected_error"])]
    with pytest.raises(expected_error):
        if case["api"] == "max_marginal_relevance_search":
            store.max_marginal_relevance_search(
                "must-not-embed",
                lambda_mult=0.7,
                **case["kwargs"],
            )
        else:
            store.similarity_search(
                "must-not-embed",
                **case["kwargs"],
            )
    assert embeddings.calls == 0
    assert engine.calls == 0
