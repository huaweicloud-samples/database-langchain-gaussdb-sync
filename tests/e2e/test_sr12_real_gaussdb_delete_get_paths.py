import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from psycopg2 import sql

from langchain_gaussdb import GaussDBVectorStore
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


def _table():
    return "sr12_" + uuid.uuid4().hex[:10]


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
        embedding=_E(),
        table_name=table,
        embedding_dimension=3,
        engine=engine,
    )


def _close(engine, *, strict):
    try:
        engine.close()
    except Exception:
        if strict:
            raise


# ---------------------------------------------------------------------------
# 1. delete(ids=[...]) where SOME ids exist and SOME don't -- only existing
#    ones are deleted, no error on the missing ids.
# ---------------------------------------------------------------------------
def test_sr12_delete_partial_existing_and_missing_ids():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(["alpha", "beta", "gamma"], ids=["d1", "d2", "d3"])

        # mix of existing + non-existing ids
        assert store.delete(ids=["d1", "missing-1", "d3", "missing-2"]) is True

        remaining = _ids(store.get_by_ids(["d1", "d2", "d3", "missing-1", "missing-2"]))
        assert remaining == ["d2"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)


# ---------------------------------------------------------------------------
# 2. delete(ids=[]) -- empty list. Should be a no-op returning True.
# ---------------------------------------------------------------------------
def test_sr12_delete_empty_id_list_is_noop():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(["alpha", "beta"], ids=["d1", "d2"])

        assert store.delete(ids=[]) is True

        # nothing removed
        assert sorted(_ids(store.get_by_ids(["d1", "d2"]))) == ["d1", "d2"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)


# ---------------------------------------------------------------------------
# 3. delete by a single NON-existent id -- no error, returns True.
# ---------------------------------------------------------------------------
def test_sr12_delete_single_nonexistent_id_is_noop():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(["alpha", "beta"], ids=["d1", "d2"])

        assert store.delete(ids=["totally-missing"]) is True

        # nothing removed
        assert sorted(_ids(store.get_by_ids(["d1", "d2"]))) == ["d1", "d2"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)


# ---------------------------------------------------------------------------
# 4. get_by_ids with a mix of existing + non-existing ids -- returns only
#    existing, in arbitrary order.
# ---------------------------------------------------------------------------
def test_sr12_get_by_ids_mixed_existing_and_missing():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(
            ["alpha", "beta", "gamma"],
            metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
            ids=["d1", "d2", "d3"],
        )

        docs = store.get_by_ids(["d3", "missing-x", "d1", "missing-y", "d2"])
        by_id = {d.id: d for d in docs}
        assert sorted(by_id) == ["d1", "d2", "d3"]
        assert by_id["d1"].page_content == "alpha"
        assert by_id["d1"].metadata == {"k": "1"}
        assert by_id["d3"].page_content == "gamma"
        assert by_id["d3"].metadata == {"k": "3"}
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)


# ---------------------------------------------------------------------------
# 5. get_by_ids with an empty list -- returns [].
# ---------------------------------------------------------------------------
def test_sr12_get_by_ids_empty_list_returns_empty():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(["alpha", "beta"], ids=["d1", "d2"])

        assert store.get_by_ids([]) == []
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)


# ---------------------------------------------------------------------------
# 6. delete(ids=[all ids]) then get_by_ids returns [] and similarity_search
#    returns nothing.
# ---------------------------------------------------------------------------
def test_sr12_delete_all_ids_then_get_and_search_are_empty():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(["alpha", "beta", "gamma"], ids=["d1", "d2", "d3"])

        # delete every id that exists
        assert store.delete(ids=["d1", "d2", "d3"]) is True

        assert store.get_by_ids(["d1", "d2", "d3"]) == []
        assert store.similarity_search("alpha", k=4) == []
        assert store.similarity_search("anything", k=10) == []
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)


# ---------------------------------------------------------------------------
# 7. delete() supports ids + delete_all only (no metadata filter in the
#    signature). Document the contract guards:
#    - delete() with no args raises ValueError (delete_all required when ids None)
#    - delete(ids=[...], delete_all=True) raises ValueError (mutually exclusive)
# ---------------------------------------------------------------------------
def test_sr12_delete_contract_guards():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(["alpha"], ids=["d1"])

        # No args -> ValueError (delete_all=True required when ids is None)
        with pytest.raises(ValueError, match="delete_all"):
            store.delete()

        # ids + delete_all=True -> ValueError (mutually exclusive)
        with pytest.raises(ValueError, match="delete_all"):
            store.delete(ids=["d1"], delete_all=True)

        # row should still exist (guards raised before any DELETE ran)
        assert _ids(store.get_by_ids(["d1"])) == ["d1"]
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)


# ---------------------------------------------------------------------------
# 8. add_texts then delete then re-add the SAME id (upsert-after-delete) --
#    verify a clean re-add (content/metadata overwritten, single row).
# ---------------------------------------------------------------------------
def test_sr12_delete_then_re_add_same_id_upserts_cleanly():
    engine = _engine()
    table = _table()
    completed = False
    try:
        _enable_writes(engine)
        store = _store(engine, table)
        store.setup()
        store.add_texts(
            ["original"],
            metadatas=[{"v": "1"}],
            ids=["d-upsert"],
        )
        assert _ids(store.get_by_ids(["d-upsert"])) == ["d-upsert"]

        # delete it
        assert store.delete(ids=["d-upsert"]) is True
        assert store.get_by_ids(["d-upsert"]) == []

        # re-add the same id with new content/metadata
        store.add_texts(
            ["reborn"],
            metadatas=[{"v": "2", "extra": True}],
            ids=["d-upsert"],
        )

        docs = store.get_by_ids(["d-upsert"])
        assert len(docs) == 1, "re-add should yield exactly one row, not a duplicate"
        assert docs[0].id == "d-upsert"
        assert docs[0].page_content == "reborn"
        assert docs[0].metadata == {"v": "2", "extra": True}

        # similarity_search should find the reborn content
        results = store.similarity_search("reborn", k=4)
        assert any(r.id == "d-upsert" for r in results)
        completed = True
    finally:
        try:
            _drop(engine, table)
        finally:
            _close(engine, strict=completed)
