from __future__ import annotations

import pytest

from langchain_gaussdb import BM25Config, GaussDBVectorStore
from langchain_gaussdb.errors import (
    GaussDBCapabilityError,
    GaussDBSQLBuildError,
    GaussDBSQLError,
)
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings, RecordingEngine

REQUIRED_COLUMNS = [("id",), ("content",), ("metadata",), ("embedding",)]


def _store(engine: RecordingEngine, **kwargs) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=engine,
        **kwargs,
    )


@pytest.mark.parametrize("table_name", ["bad\0table", "界" * 22])
def test_constructor_rejects_invalid_table_identifier_before_sql(table_name) -> None:
    engine = RecordingEngine()

    with pytest.raises(GaussDBSQLBuildError):
        GaussDBVectorStore(
            embedding=DeterministicEmbeddings(),
            table_name=table_name,
            embedding_dimension=3,
            engine=engine,
        )

    assert engine.executed == []
    assert engine.fetched == []


def test_constructor_does_not_touch_database() -> None:
    engine = RecordingEngine()

    store = _store(engine)

    assert isinstance(store, GaussDBVectorStore)
    assert engine.calls == []


def test_missing_table_is_created_then_checked_for_required_columns() -> None:
    engine = RecordingEngine(fetch_results=[[], REQUIRED_COLUMNS])
    store = _store(engine)

    store._prepare_table_if_needed()

    assert [operation for _compiled, operation in engine.executed] == [
        "create vectorstore table"
    ]
    ddl = repr(engine.executed[0][0].statement)
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert "JSONB NOT NULL DEFAULT" in ddl
    assert "floatvector" in ddl
    assert "storage_type=ustore" in ddl
    assert [operation for _compiled, operation in engine.fetched] == [
        "check vectorstore required columns",
        "check vectorstore required columns",
    ]
    assert engine.fetched[0][0].params == ("documents", "public")


def test_existing_public_table_is_not_recreated() -> None:
    engine = RecordingEngine(fetch_results=[REQUIRED_COLUMNS])
    store = _store(engine)

    store._prepare_table_if_needed()

    assert engine.executed == []
    required_sql, operation = engine.fetched[-1]
    assert operation == "check vectorstore required columns"
    assert required_sql.params == ("documents", "public")
    assert "current_schema" not in repr(required_sql.statement)


def test_existing_table_only_checks_column_names() -> None:
    engine = RecordingEngine(fetch_results=[REQUIRED_COLUMNS])
    store = _store(engine)

    store._prepare_table_if_needed()

    statement = repr(engine.fetched[-1][0].statement)
    assert "column_name" in statement
    assert "data_type" not in statement
    assert "udt_name" not in statement
    assert "pg_index" not in statement
    assert len(engine.fetched) == 1


def test_missing_required_column_is_rejected() -> None:
    engine = RecordingEngine(fetch_results=[[("id",), ("content",), ("embedding",)]])
    store = _store(engine)

    with pytest.raises(GaussDBCapabilityError, match="metadata"):
        store._prepare_table_if_needed()


def test_metadata_index_keys_are_not_columns_but_lexical_projection_is_required() -> (
    None
):
    engine = RecordingEngine(
        fetch_results=[
            [*REQUIRED_COLUMNS, ("content_lexical",)],
        ]
    )
    store = _store(
        engine,
        retrieval_mode="bm25",
        metadata_indexes={"tenant_id": "text"},
        bm25_config=BM25Config(column="content_lexical"),
    )

    store._prepare_table_if_needed()

    assert engine.executed == []


def test_required_column_check_ignores_extra_columns() -> None:
    engine = RecordingEngine(fetch_results=[[*REQUIRED_COLUMNS, ("legacy_payload",)]])
    store = _store(engine)

    store._prepare_table_if_needed()

    assert engine.executed == []


def test_configured_schema_is_used_for_table_and_column_check() -> None:
    engine = RecordingEngine(fetch_results=[REQUIRED_COLUMNS])
    store = _store(engine, schema_name="application")

    store._prepare_table_if_needed()

    required_sql, _operation = engine.fetched[0]
    assert required_sql.params == ("documents", "application")
    assert "pg_table_is_visible" not in repr(required_sql.statement)
    assert engine.executed == []


@pytest.mark.parametrize("sqlstate", ["42P07", "42710", "23505"])
def test_concurrent_table_creator_is_treated_as_success(sqlstate: str) -> None:
    engine = RecordingEngine(fetch_results=[[], REQUIRED_COLUMNS])
    duplicate = GaussDBSQLError("duplicate table", sqlstate=sqlstate)

    def lose_create_race(compiled, *, operation="execute"):
        engine.executed.append((compiled, operation))
        raise duplicate

    engine.execute = lose_create_race
    store = _store(engine)

    store._prepare_table_if_needed()

    assert engine.fetched[-1][1] == "check vectorstore required columns"


def test_nonduplicate_create_error_is_preserved() -> None:
    engine = RecordingEngine(fetch_results=[[]])
    failure = GaussDBSQLError("read only transaction", sqlstate="25006")

    def fail_create(_compiled, *, operation="execute"):
        raise failure

    engine.execute = fail_create
    store = _store(engine)

    with pytest.raises(GaussDBSQLError) as exc_info:
        store._prepare_table_if_needed()

    assert exc_info.value is failure


def test_table_must_be_visible_after_create() -> None:
    engine = RecordingEngine(fetch_results=[[], []])
    store = _store(engine)

    with pytest.raises(GaussDBCapabilityError, match="missing required columns"):
        store._prepare_table_if_needed()
