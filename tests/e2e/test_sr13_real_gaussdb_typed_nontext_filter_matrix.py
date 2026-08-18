"""SR-13: typed JSONB expression-index filter matrices.

Covers ``bigint`` / ``float`` / ``boolean`` JSONB expression indexes
(declared via ``metadata_indexes={"col": "bigint"|"float"|"boolean"}``) that
are under-tested relative to the untyped-JSONB numeric path exercised in
``sr04`` (``shape`` / ``scientific`` there are NOT declared as typed columns;
they ride the JSONB scalar selector).

Focus areas vs sr04:
- bigint / float full operator matrix ($eq/$ne/$gt/$lt/$gte/$lte/$in/$nin/$between)
  including rows where the typed column is ABSENT (materialized column NULL).
- $ne / $nin NULL-handling on typed columns (the sr10 bm25 gap, re-checked in
  dense typed mode).
- float NaN rejection (``_infer_typed_scalar_type`` rejects non-finite floats).
- boolean $eq/$ne/$in/$exists semantics.
- type-mismatch errors (bigint col + string value, boolean col + int value,
  float col + bool value).

Each test follows the try/finally + ``completed`` flag pattern and cleans its
own table + engine.
"""

import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBFilterError, GaussDBVectorStore
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL, qualified_name

pytestmark = pytest.mark.gaussdb_e2e


class _E(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def _dsn():
    v = os.getenv("GAUSSDB_TEST_DSN")
    if not v:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return v


def _engine():
    return GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)


def _enable_writes(engine):
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="rw",
    )
    engine.execute(
        CompiledSQL(sql.SQL("SET maintenance_work_mem = '128MB'")),
        operation="set sr13 index maintenance memory",
    )


def _table():
    return "sr13_" + uuid.uuid4().hex[:10]


def _drop(engine, table):
    try:
        _enable_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
            ),
            operation="drop",
        )
    except Exception:
        pass


def _ids(docs):
    return sorted(d.id for d in docs)


def _indexed_store(engine, table, indexes):
    return GaussDBVectorStore(
        embedding=_E(),
        table_name=table,
        embedding_dimension=3,
        engine=engine,
        metadata_indexes=indexes,
    )


def _close(engine):
    try:
        engine.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# BIGINT matrix
# ---------------------------------------------------------------------------


def test_sr13_bigint_full_operator_matrix_with_present_values():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"rank": "bigint"})
        store.setup()
        # rows: 10, 20, 30 (all present). No absent row here.
        store.add_texts(
            ["a", "b", "c"],
            metadatas=[{"rank": 10}, {"rank": 20}, {"rank": 30}],
            ids=["d10", "d20", "d30"],
        )

        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$eq": 20}})
        ) == ["d20"]
        assert _ids(store.similarity_search("q", k=10, filter={"rank": 20})) == ["d20"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$ne": 20}})
        ) == ["d10", "d30"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$gt": 15}})
        ) == ["d20", "d30"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$gte": 20}})
        ) == ["d20", "d30"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$lt": 20}})
        ) == ["d10"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$lte": 20}})
        ) == ["d10", "d20"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$in": [10, 30]}})
        ) == ["d10", "d30"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$nin": [10, 30]}})
        ) == ["d20"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$between": [15, 25]}})
        ) == ["d20"]
        # empty $in -> FALSE, empty $nin -> TRUE
        assert (
            _ids(store.similarity_search("q", k=10, filter={"rank": {"$in": []}})) == []
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$nin": []}})
        ) == ["d10", "d20", "d30"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_bigint_ne_nin_use_sql_null_semantics():
    """Typed ``NULL`` rows do not satisfy ``<>`` or ``<> ALL`` predicates."""
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"rank": "bigint"})
        store.setup()
        store.add_texts(
            ["a", "b", "c"],
            metadatas=[{"rank": 20}, {"rank": 30}, {"other": "x"}],
            ids=["d20", "d30", "d_abs"],
        )

        # $eq and $in do not match absent rows (NULL is not equal).
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$eq": 20}})
        ) == ["d20"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$in": [20, 30]}})
        ) == ["d20", "d30"]

        # SQL NULL produces UNKNOWN and is therefore excluded.
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$ne": 20}})
        ) == ["d30"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$nin": [20]}})
        ) == ["d30"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_bigint_gt_lt_on_absent_rows():
    """Range operators on a typed column must not match absent (NULL) rows.

    SQL ``NULL > 15`` is NULL (not TRUE), so absent rows are naturally
    excluded from $gt/$gte/$lt/$lte even though the typed selector reads
    the materialized NULL column directly (no key-existence guard here).
    """
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"rank": "bigint"})
        store.setup()
        store.add_texts(
            ["a", "b"],
            metadatas=[{"rank": 20}, {"other": "x"}],
            ids=["d20", "d_abs"],
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$gt": 15}})
        ) == ["d20"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$lt": 100}})
        ) == ["d20"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$between": [0, 100]}})
        ) == ["d20"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_bigint_type_mismatch_string_value_raises():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"rank": "bigint"})
        store.setup()
        store.add_texts(["a"], metadatas=[{"rank": 10}], ids=["d10"])

        with pytest.raises(GaussDBFilterError) as exc:
            store.similarity_search("q", k=10, filter={"rank": {"$eq": "10"}})
        assert "bigint" in str(exc.value).lower() or "int" in str(exc.value).lower()

        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"rank": {"$in": ["10", "20"]}})

        with pytest.raises(GaussDBFilterError):
            store.similarity_search(
                "q", k=10, filter={"rank": {"$between": ["a", "z"]}}
            )
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_bigint_bool_value_rejected():
    """Python bool is an int subclass; typed bigint path must reject bool."""
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"rank": "bigint"})
        store.setup()
        store.add_texts(["a"], metadatas=[{"rank": 10}], ids=["d10"])
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"rank": {"$eq": True}})
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"rank": {"$in": [10, False]}})
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


# ---------------------------------------------------------------------------
# FLOAT matrix
# ---------------------------------------------------------------------------


def test_sr13_float_full_operator_matrix_with_present_values():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"score": "float"})
        store.setup()
        store.add_texts(
            ["a", "b", "c"],
            metadatas=[{"score": 0.5}, {"score": 1.5}, {"score": 2.5}],
            ids=["f05", "f15", "f25"],
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$eq": 1.5}})
        ) == ["f15"]
        # int filter value against float column is allowed (numeric).
        assert (
            _ids(store.similarity_search("q", k=10, filter={"score": {"$eq": 2}})) == []
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$ne": 1.5}})
        ) == ["f05", "f25"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$gt": 1.0}})
        ) == ["f15", "f25"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$gte": 1.5}})
        ) == ["f15", "f25"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$lt": 1.5}})
        ) == ["f05"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$lte": 1.5}})
        ) == ["f05", "f15"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$in": [0.5, 2.5]}})
        ) == ["f05", "f25"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$nin": [0.5, 2.5]}})
        ) == ["f15"]
        assert _ids(
            store.similarity_search(
                "q", k=10, filter={"score": {"$between": [1.0, 2.0]}}
            )
        ) == ["f15"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_float_int_filter_value_matches_compatible_float():
    """A float column storing 2.0 must be matched by an int filter value 2,
    and a stored int-coerced value 3 should match float 3.0."""
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"score": "float"})
        store.setup()
        store.add_texts(
            ["a", "b"],
            metadatas=[{"score": 2.0}, {"score": 3}],
            ids=["f2", "f3"],
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$eq": 2}})
        ) == ["f2"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$eq": 3.0}})
        ) == ["f3"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$in": [2, 3]}})
        ) == ["f2", "f3"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_float_nan_and_inf_filter_value_rejected():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"score": "float"})
        store.setup()
        store.add_texts(["a"], metadatas=[{"score": 1.5}], ids=["f15"])

        with pytest.raises(GaussDBFilterError) as exc:
            store.similarity_search("q", k=10, filter={"score": {"$eq": float("nan")}})
        assert "finite" in str(exc.value).lower()

        with pytest.raises(GaussDBFilterError):
            store.similarity_search(
                "q", k=10, filter={"score": {"$in": [float("inf"), 1.5]}}
            )

        with pytest.raises(GaussDBFilterError):
            store.similarity_search(
                "q", k=10, filter={"score": {"$between": [float("-inf"), 1.5]}}
            )
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_float_bool_value_rejected():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"score": "float"})
        store.setup()
        store.add_texts(["a"], metadatas=[{"score": 1.5}], ids=["f15"])
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"score": {"$eq": True}})
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"score": {"$in": [1.5, False]}})
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_float_ne_nin_use_sql_null_semantics():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"score": "float"})
        store.setup()
        store.add_texts(
            ["a", "b", "c"],
            metadatas=[{"score": 1.5}, {"score": 2.5}, {"other": "x"}],
            ids=["f15", "f25", "f_abs"],
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$ne": 1.5}})
        ) == ["f25"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$nin": [1.5]}})
        ) == ["f25"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$gt": 1.0}})
        ) == ["f15", "f25"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


# ---------------------------------------------------------------------------
# BOOLEAN matrix
# ---------------------------------------------------------------------------


def test_sr13_boolean_full_operator_matrix():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"flag": "boolean"})
        store.setup()
        store.add_texts(
            ["a", "b", "c"],
            metadatas=[{"flag": True}, {"flag": False}, {"other": "x"}],
            ids=["b_t", "b_f", "b_abs"],
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$eq": True}})
        ) == ["b_t"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$eq": False}})
        ) == ["b_f"]
        assert _ids(store.similarity_search("q", k=10, filter={"flag": True})) == [
            "b_t"
        ]
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$ne": True}})
        ) == ["b_f"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$ne": False}})
        ) == ["b_t"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$in": [True, False]}})
        ) == ["b_f", "b_t"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$nin": [True]}})
        ) == ["b_f"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_boolean_exists_semantics():
    """$exists reads the JSON scalar and applies PG v2 SQL NULL semantics."""
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"flag": "boolean"})
        store.setup()
        store.add_texts(
            ["a", "b", "c"],
            metadatas=[{"flag": True}, {"flag": False}, {"other": "x"}],
            ids=["b_t", "b_f", "b_abs"],
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$exists": True}})
        ) == [
            "b_f",
            "b_t",
        ]
        assert _ids(
            store.similarity_search("q", k=10, filter={"flag": {"$exists": False}})
        ) == ["b_abs"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_boolean_int_value_rejected():
    """Typed boolean column must reject int 0/1 filter values (bool required).

    This confirms bool is NOT conflated with int 1 on the typed boolean path:
    even though Python ``True == 1``, the typed kind check rejects non-bool.
    """
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"flag": "boolean"})
        store.setup()
        store.add_texts(["a"], metadatas=[{"flag": True}], ids=["b_t"])
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"flag": {"$eq": 1}})
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"flag": {"$eq": 0}})
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"flag": {"$in": [1, 0]}})
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


def test_sr13_boolean_range_operators_rejected():
    """boolean is not in _RANGE_KINDS, so $gt/$lt/$between must raise."""
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _indexed_store(engine, table, {"flag": "boolean"})
        store.setup()
        store.add_texts(["a"], metadatas=[{"flag": True}], ids=["b_t"])
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("q", k=10, filter={"flag": {"$gt": False}})
        with pytest.raises(GaussDBFilterError):
            store.similarity_search(
                "q", k=10, filter={"flag": {"$between": [False, True]}}
            )
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed


# ---------------------------------------------------------------------------
# Cross-type: verify the type strings accepted by the constructor normalize.
# ---------------------------------------------------------------------------


def test_sr13_integer_and_double_precision_type_aliases_normalize():
    """Keep ``integer`` distinct and normalize ``double precision`` to float."""
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = GaussDBVectorStore(
            embedding=_E(),
            table_name=table,
            embedding_dimension=3,
            engine=engine,
            metadata_indexes={"rank": "integer", "score": "double precision"},
        )
        assert store.metadata_indexes == {"rank": "bigint", "score": "float"}
        store.setup()
        store.add_texts(
            ["a", "b"],
            metadatas=[{"rank": 5, "score": 1.25}, {"rank": 6, "score": 2.0}],
            ids=["a1", "a2"],
        )
        assert _ids(
            store.similarity_search("q", k=10, filter={"rank": {"$eq": 5}})
        ) == ["a1"]
        assert _ids(
            store.similarity_search("q", k=10, filter={"score": {"$gte": 1.5}})
        ) == ["a2"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine)
    assert completed
