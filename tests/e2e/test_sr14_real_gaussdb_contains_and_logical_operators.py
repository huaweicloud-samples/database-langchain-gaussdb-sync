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
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")), operation="rw"
    )


def _table():
    return "sr14_" + uuid.uuid4().hex[:10]


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


def _store(engine, table):
    return GaussDBVectorStore(
        embedding=_E(), table_name=table, embedding_dimension=3, engine=engine
    )


def _seed(store):
    """Seed a store with rich nested metadata used across $contains / logical tests."""
    store.setup()
    store.add_texts(
        ["d0", "d1", "d2", "d3"],
        metadatas=[
            # doc-a: tags list with 3 elements; meta nested object; scalar fields
            {
                "tags": ["a", "b", "c"],
                "meta": {"k": "v", "x": 1},
                "shape": "circle",
                "rank": 10,
                "tenant": "t1",
            },
            # doc-b: tags list superset; different meta; different scalars
            {
                "tags": ["a", "b", "c", "d"],
                "meta": {"k": "v", "x": 2},
                "shape": "square",
                "rank": 3,
                "tenant": "t1",
            },
            # doc-c: tags list partial overlap; meta missing k; scalar
            {
                "tags": ["a", "z"],
                "meta": {"k": "other"},
                "shape": "circle",
                "rank": 8,
                "tenant": "t2",
            },
            # doc-d: no tags; meta object only
            {
                "meta": {"k": "v"},
                "shape": "triangle",
                "rank": 1,
                "tenant": "t2",
            },
        ],
        ids=["doc-a", "doc-b", "doc-c", "doc-d"],
    )


# ---------------------------------------------------------------------------
# 1. $contains with a LIST value (jsonb @> containment on arrays)
# ---------------------------------------------------------------------------


def test_contains_list_partial_and_exact_and_nonmatching():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # partial-list containment: ["a","b"] is contained in doc-a and doc-b
        partial = store.similarity_search(
            "n", k=4, filter={"tags": {"$contains": ["a", "b"]}}
        )
        assert _ids(partial) == ["doc-a", "doc-b"]

        # exact-list containment: ["a","b","c"] contained in doc-a (exact) and doc-b (superset)
        exact = store.similarity_search(
            "n", k=4, filter={"tags": {"$contains": ["a", "b", "c"]}}
        )
        assert _ids(exact) == ["doc-a", "doc-b"]

        # superset query list contained nowhere
        none_match = store.similarity_search(
            "n", k=4, filter={"tags": {"$contains": ["a", "b", "c", "d", "e"]}}
        )
        assert _ids(none_match) == []

        # single-element list containment (superset behavior)
        single = store.similarity_search(
            "n", k=4, filter={"tags": {"$contains": ["z"]}}
        )
        assert _ids(single) == ["doc-c"]

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed


# ---------------------------------------------------------------------------
# 2. $contains with a NESTED OBJECT (jsonb @> containment on objects)
# ---------------------------------------------------------------------------


def test_contains_nested_object():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # meta contains {"k":"v"}: doc-a, doc-b (meta.k=="v"), doc-d (meta.k=="v")
        obj = store.similarity_search(
            "n", k=4, filter={"meta": {"$contains": {"k": "v"}}}
        )
        assert _ids(obj) == ["doc-a", "doc-b", "doc-d"]

        # narrower: {"k":"v","x":1} only matches doc-a
        narrow = store.similarity_search(
            "n", k=4, filter={"meta": {"$contains": {"k": "v", "x": 1}}}
        )
        assert _ids(narrow) == ["doc-a"]

        # {"k":"other"} matches doc-c
        other = store.similarity_search(
            "n", k=4, filter={"meta": {"$contains": {"k": "other"}}}
        )
        assert _ids(other) == ["doc-c"]

        # nested key not present in any doc
        missing = store.similarity_search(
            "n", k=4, filter={"meta": {"$contains": {"k": "v", "x": 99}}}
        )
        assert _ids(missing) == []

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed


# ---------------------------------------------------------------------------
# 3. $contains with a SCALAR — behaves like $eq (jsonb @> scalar)
# ---------------------------------------------------------------------------


def test_contains_scalar_matches_eq():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # scalar $contains on a string field
        contains_circle = store.similarity_search(
            "n", k=4, filter={"shape": {"$contains": "circle"}}
        )
        eq_circle = store.similarity_search(
            "n", k=4, filter={"shape": {"$eq": "circle"}}
        )
        assert _ids(contains_circle) == _ids(eq_circle)
        assert _ids(contains_circle) == ["doc-a", "doc-c"]

        # scalar $contains on a numeric field
        contains_rank = store.similarity_search(
            "n", k=4, filter={"rank": {"$contains": 10}}
        )
        eq_rank = store.similarity_search("n", k=4, filter={"rank": {"$eq": 10}})
        assert _ids(contains_rank) == _ids(eq_rank)
        assert _ids(contains_rank) == ["doc-a"]

        # scalar $contains that matches nothing
        none_match = store.similarity_search(
            "n", k=4, filter={"shape": {"$contains": "hexagon"}}
        )
        assert _ids(none_match) == []

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed


# ---------------------------------------------------------------------------
# 4. Nested $and wrapping $or (2-3 levels deep)
# ---------------------------------------------------------------------------


def test_nested_and_wrapping_or():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # ($and: [ shape==circle OR shape==square ]) AND (rank>=3)
        nested = store.similarity_search(
            "n",
            k=4,
            filter={
                "$and": [
                    {
                        "$or": [
                            {"shape": {"$eq": "circle"}},
                            {"shape": {"$eq": "square"}},
                        ]
                    },
                    {"rank": {"$gte": 3}},
                ]
            },
        )
        # shape circle|square and rank>=3 -> doc-a(10), doc-b(3), doc-c(8)
        assert _ids(nested) == ["doc-a", "doc-b", "doc-c"]

        # 3 levels: $and([ $or([ shape==circle, $not rank==3 ]), tenant==t1, rank>=3 ])
        three_deep = store.similarity_search(
            "n",
            k=4,
            filter={
                "$and": [
                    {
                        "$or": [
                            {"shape": {"$eq": "circle"}},
                            {"$not": {"rank": {"$eq": 3}}},
                        ]
                    },
                    {"tenant": {"$eq": "t1"}},
                    {"rank": {"$gte": 3}},
                ]
            },
        )
        # tenant t1: doc-a(shape circle,rank10), doc-b(shape square,rank3)
        # first clause: shape==circle OR NOT(rank==3)
        #   doc-a: circle True -> True
        #   doc-b: square False, NOT(rank==3) -> NOT(True)->False -> False
        assert _ids(three_deep) == ["doc-a"]

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed


# ---------------------------------------------------------------------------
# 5. $not negation and $not wrapping a complex $and
# ---------------------------------------------------------------------------


def test_not_negation_and_not_wrapping_and():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # $not {shape==circle}: not circle -> doc-b(square), doc-d(triangle)
        not_circle = store.similarity_search(
            "n", k=4, filter={"$not": {"shape": {"$eq": "circle"}}}
        )
        assert _ids(not_circle) == ["doc-b", "doc-d"]

        # $not wrapping a complex $and: NOT( shape==square AND rank==3 )
        not_complex = store.similarity_search(
            "n",
            k=4,
            filter={
                "$not": {
                    "$and": [
                        {"shape": {"$eq": "square"}},
                        {"rank": {"$eq": 3}},
                    ]
                }
            },
        )
        # only doc-b matches the inner AND; negation -> everything except doc-b
        assert _ids(not_complex) == ["doc-a", "doc-c", "doc-d"]

        # $not on scalar equality via $contains-equivalent: NOT tenant==t1
        not_t1 = store.similarity_search(
            "n", k=4, filter={"$not": {"tenant": {"$eq": "t1"}}}
        )
        assert _ids(not_t1) == ["doc-c", "doc-d"]

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed


# ---------------------------------------------------------------------------
# 6. $or with 3 branches
# ---------------------------------------------------------------------------


def test_or_three_branches():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        three_or = store.similarity_search(
            "n",
            k=4,
            filter={
                "$or": [
                    {"shape": {"$eq": "triangle"}},
                    {"rank": {"$eq": 10}},
                    {"tenant": {"$eq": "t2"}},
                ]
            },
        )
        # triangle: doc-d ; rank 10: doc-a ; tenant t2: doc-c, doc-d
        assert _ids(three_or) == ["doc-a", "doc-c", "doc-d"]

        # $or where all branches fail -> empty
        all_fail = store.similarity_search(
            "n",
            k=4,
            filter={
                "$or": [
                    {"shape": {"$eq": "pentagon"}},
                    {"rank": {"$eq": 999}},
                    {"tenant": {"$eq": "t9"}},
                ]
            },
        )
        assert _ids(all_fail) == []

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed


# ---------------------------------------------------------------------------
# 7. Error cases (raise GaussDBFilterError)
# ---------------------------------------------------------------------------


def test_error_cases_invalid_logical_and_field_operator():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # $and with non-list value
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"$and": {"shape": "circle"}})

        # $and with empty list
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"$and": []})

        # $or with non-list value
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"$or": {"shape": "circle"}})

        # $or with empty list
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"$or": []})

        # $not with non-dict value
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"$not": [1, 2, 3]})

        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"$not": "circle"})

        # mixing $ operator and field in the same dict
        with pytest.raises(GaussDBFilterError):
            store.similarity_search(
                "n", k=1, filter={"$and": [{"shape": "circle"}], "rank": 1}
            )

        # unsupported operator ($regex not supported)
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"shape": {"$regex": "cir.*"}})

        # unsupported top-level logical operator
        with pytest.raises(GaussDBFilterError):
            store.similarity_search("n", k=1, filter={"$nand": [{"shape": "circle"}]})

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed


# ---------------------------------------------------------------------------
# 8. Implicit AND (multiple fields, no $ operator) equals explicit $and
# ---------------------------------------------------------------------------


def test_implicit_and_equals_explicit_and():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        _seed(store)

        # implicit AND: shape==circle AND tenant==t1 -> doc-a
        implicit = store.similarity_search(
            "n",
            k=4,
            filter={"shape": "circle", "tenant": "t1"},
        )
        # explicit $and
        explicit = store.similarity_search(
            "n",
            k=4,
            filter={
                "$and": [
                    {"shape": {"$eq": "circle"}},
                    {"tenant": {"$eq": "t1"}},
                ]
            },
        )
        assert _ids(implicit) == _ids(explicit)
        assert _ids(implicit) == ["doc-a"]

        # implicit AND across 3 fields
        implicit3 = store.similarity_search(
            "n",
            k=4,
            filter={"tenant": "t1", "rank": {"$gte": 3}, "shape": "square"},
        )
        assert _ids(implicit3) == ["doc-b"]

        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            engine.close()
            assert completed
