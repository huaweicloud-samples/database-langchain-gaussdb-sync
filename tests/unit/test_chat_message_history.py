from __future__ import annotations

import asyncio
import json
import math
import threading

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

import langchain_gaussdb.chat_message_history as chat_history_module
from langchain_gaussdb.chat_message_history import GaussDBChatMessageHistory
from langchain_gaussdb.errors import (
    GaussDBCapabilityError,
    GaussDBConnectionError,
    GaussDBSQLBuildError,
    GaussDBSQLError,
)


class RecordingEngine:
    def __init__(self) -> None:
        self.executed = []
        self.fetches = []
        self.rows = []
        self.closed = False
        self.fail_operations = set()
        self.failures = {}
        self.rows_by_operation = {}
        self.execute_threads = []
        self.fetch_threads = []

    def execute(self, compiled, *, operation: str = "execute") -> None:
        self.execute_threads.append(threading.get_ident())
        if operation in self.failures:
            raise self.failures[operation]
        if operation in self.fail_operations:
            raise RuntimeError(f"{operation} failed")
        self.executed.append((compiled, operation))

    def fetch_all(self, compiled, *, operation: str = "fetch_all"):
        self.fetch_threads.append(threading.get_ident())
        if operation in self.failures:
            raise self.failures[operation]
        if operation in self.fail_operations:
            raise RuntimeError(f"{operation} failed")
        self.fetches.append((compiled, operation))
        return list(self.rows_by_operation.get(operation, self.rows))

    def close(self) -> None:
        self.closed = True


def _statement_repr(compiled) -> str:
    return repr(compiled.statement)


def _valid_chat_index_row(
    *,
    schema: str = "public",
    table: str = "chat_messages",
    method: str = "ubtree",
    key_count: int = 2,
    total_count: int = 2,
    first_key: str = "session_id",
    second_key: str = "id",
    valid: bool = True,
    ready: bool = True,
    usable: bool = True,
    predicate=None,
    expressions=None,
):
    return (
        schema,
        table,
        method,
        key_count,
        total_count,
        first_key,
        second_key,
        valid,
        ready,
        usable,
        predicate,
        expressions,
    )


def _valid_chat_table_rows(
    *,
    schema: str = "public",
    table: str = "chat_messages",
):
    primary_key = (1, 1, 1, True, True, True, None, None)
    return [
        (
            schema,
            table,
            1,
            "id",
            "bigint",
            True,
            True,
            "nextval('chat_messages_id_seq'::regclass)",
            *primary_key,
        ),
        (
            schema,
            table,
            2,
            "session_id",
            "text",
            True,
            False,
            None,
            *primary_key,
        ),
        (
            schema,
            table,
            3,
            "message",
            "jsonb",
            True,
            False,
            None,
            *primary_key,
        ),
        (
            schema,
            table,
            4,
            "created_at",
            "timestamp without time zone",
            False,
            True,
            "CURRENT_TIMESTAMP",
            *primary_key,
        ),
    ]


def _changed_chat_table_rows(
    row_index: int,
    field_index: int,
    value,
):
    rows = _valid_chat_table_rows()
    changed = list(rows[row_index])
    changed[field_index] = value
    rows[row_index] = tuple(changed)
    return rows


def _history(
    engine: RecordingEngine | None = None,
    *,
    session_id: str = "s1",
    table_name: str = "chat_messages",
) -> GaussDBChatMessageHistory:
    return GaussDBChatMessageHistory(
        session_id=session_id,
        table_name=table_name,
        engine=engine or RecordingEngine(),
    )


def _executed_json_messages(engine: RecordingEngine) -> list[dict]:
    return [
        json.loads(param)
        for compiled, _ in engine.executed
        for param in compiled.params[1::2]
    ]


def test_constructor_accepts_external_engine_without_sql():
    engine = RecordingEngine()

    history = GaussDBChatMessageHistory(
        session_id="s1",
        table_name="chat_messages",
        engine=engine,
    )

    assert history.session_id == "s1"
    assert history.table_name == "chat_messages"
    assert history.schema_name is None
    assert engine.executed == []


def test_constructor_builds_owned_engine_from_dsn(monkeypatch):
    created_engines = []

    class FakeGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            created_engines.append(self)

    monkeypatch.setattr(chat_history_module, "GaussDBEngine", FakeGaussDBEngine)

    dsn = "host=127.0.0.1 dbname=unit"
    history = GaussDBChatMessageHistory(
        session_id="s1",
        table_name="chat_messages",
        dsn=dsn,
    )

    assert history.session_id == "s1"
    assert len(created_engines) == 1
    assert created_engines[0].kwargs == {
        "dsn": dsn,
        "connection_kwargs": None,
    }
    assert created_engines[0].executed == []

    history.close()

    assert created_engines[0].closed is True


@pytest.mark.parametrize(
    "source_kwargs",
    [
        {"dsn": "host=127.0.0.1 dbname=unit"},
        {"connection_kwargs": {"host": "127.0.0.1", "dbname": "unit"}},
    ],
)
def test_constructor_rejects_engine_with_other_connection_sources(source_kwargs):
    with pytest.raises(GaussDBConnectionError, match="engine"):
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            engine=RecordingEngine(),
            **source_kwargs,
        )


def test_constructor_rejects_missing_connection_source():
    with pytest.raises(GaussDBConnectionError, match="connection source"):
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
        )


def test_constructor_rejects_multiple_connection_sources_without_engine():
    with pytest.raises(GaussDBConnectionError, match="connection source"):
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            dsn="host=127.0.0.1 dbname=unit",
            connection_kwargs={"host": "127.0.0.1", "dbname": "unit"},
        )


@pytest.mark.parametrize("removed_argument", ["pool", "connection"])
def test_constructor_rejects_removed_connection_injection(removed_argument):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            **{removed_argument: object()},
        )


@pytest.mark.parametrize("session_id", ["", "   "])
def test_constructor_rejects_empty_session_id(session_id):
    with pytest.raises(ValueError, match="session_id"):
        GaussDBChatMessageHistory(
            session_id=session_id,
            table_name="chat_messages",
            engine=RecordingEngine(),
        )


@pytest.mark.parametrize("session_id", [None, 123])
def test_constructor_rejects_non_string_session_id(session_id):
    with pytest.raises(ValueError, match="session_id"):
        GaussDBChatMessageHistory(
            session_id=session_id,
            table_name="chat_messages",
            engine=RecordingEngine(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"table_name": "public.chat"},
        {"schema_name": "bad.schema"},
    ],
)
def test_constructor_rejects_dotted_table_or_schema(kwargs):
    constructor_kwargs = {
        "session_id": "s1",
        "table_name": "chat_messages",
        "engine": RecordingEngine(),
    }
    constructor_kwargs.update(kwargs)

    with pytest.raises(GaussDBSQLBuildError, match="single identifier"):
        GaussDBChatMessageHistory(**constructor_kwargs)


def test_create_table_if_not_exists_executes_table_and_index_ddl():
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row()],
    }
    history = GaussDBChatMessageHistory(
        session_id="s1",
        table_name="chat_messages",
        engine=engine,
    )

    history.create_table_if_not_exists()

    assert len(engine.executed) == 2

    create_table, create_table_operation = engine.executed[0]
    create_table_sql = _statement_repr(create_table)
    assert create_table.params == ()
    assert create_table_operation == "create chat message table"
    assert "CREATE TABLE IF NOT EXISTS" in create_table_sql
    assert "BIGSERIAL PRIMARY KEY" in create_table_sql
    assert "JSONB NOT NULL" in create_table_sql
    assert "WITH (storage_type=ustore)" in create_table_sql
    assert "Identifier('chat_messages')" in create_table_sql

    create_index, create_index_operation = engine.executed[1]
    create_index_sql = _statement_repr(create_index)
    assert create_index.params == ()
    assert create_index_operation == "create chat message index"
    assert "CREATE INDEX IF NOT EXISTS" in create_index_sql
    assert "Identifier('chat_messages_session_id_id_idx')" in create_index_sql
    assert "session_id, id" in create_index_sql


def test_successful_create_table_validates_table_before_creating_index():
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row()],
    }
    history = _history(engine)

    history.create_table_if_not_exists()

    assert [operation for _compiled, operation in engine.fetches] == [
        "validate chat message table",
        "validate chat message index",
    ]


def test_successful_create_table_rejects_wrong_table_before_creating_index():
    engine = RecordingEngine()
    engine.rows_by_operation["validate chat message table"] = _changed_chat_table_rows(
        2, 4, "text"
    )
    history = _history(engine)

    with pytest.raises(GaussDBCapabilityError, match="different definition"):
        history.create_table_if_not_exists()

    assert [operation for _compiled, operation in engine.executed] == [
        "create chat message table"
    ]
    assert [operation for _compiled, operation in engine.fetches] == [
        "validate chat message table"
    ]


def test_create_table_if_not_exists_shortens_long_generated_index_name():
    engine = RecordingEngine()
    long_table_name = "x" * 63
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(table=long_table_name),
        "validate chat message index": [_valid_chat_index_row(table=long_table_name)],
    }
    history = GaussDBChatMessageHistory(
        session_id="s1",
        table_name=long_table_name,
        engine=engine,
    )

    history.create_table_if_not_exists()

    create_index, _ = engine.executed[1]
    create_index_sql = _statement_repr(create_index)
    expected_prefix = f"{long_table_name[:36]}_"
    assert expected_prefix in create_index_sql
    assert "session_id_id_idx" in create_index_sql
    assert f"Identifier('{long_table_name}_session_id_id_idx')" not in create_index_sql


def test_create_table_if_not_exists_uses_schema_qualified_table_name():
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row()],
    }
    history = GaussDBChatMessageHistory(
        session_id="s1",
        schema_name="public",
        table_name="chat_messages",
        engine=engine,
    )

    history.create_table_if_not_exists()

    statements = [_statement_repr(compiled) for compiled, _ in engine.executed]
    assert all("Identifier('public')" in statement for statement in statements)
    assert all("Identifier('chat_messages')" in statement for statement in statements)


def test_constructor_create_table_option_runs_lifecycle_once():
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row()],
    }

    GaussDBChatMessageHistory(
        session_id="s1",
        table_name="chat_messages",
        engine=engine,
        create_table=True,
    )

    assert len(engine.executed) == 2
    statements = [_statement_repr(compiled) for compiled, _ in engine.executed]
    assert all("INSERT" not in statement for statement in statements)


@pytest.mark.parametrize("method", ["btree", "ubtree"])
def test_create_table_if_not_exists_accepts_matching_btree_family_index(method):
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row(method=method)],
    }
    history = _history(engine)

    history.create_table_if_not_exists()
    history.create_table_if_not_exists()

    assert len(engine.executed) == 4
    assert [operation for _compiled, operation in engine.fetches] == [
        "validate chat message table",
        "validate chat message index",
        "validate chat message table",
        "validate chat message index",
    ]


def test_create_table_if_not_exists_validates_index_in_explicit_schema():
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(schema="chat_schema"),
        "validate chat message index": [_valid_chat_index_row(schema="chat_schema")],
    }
    history = GaussDBChatMessageHistory(
        session_id="s1",
        schema_name="chat_schema",
        table_name="chat_messages",
        engine=engine,
    )

    history.create_table_if_not_exists()

    compiled, operation = engine.fetches[1]
    statement = _statement_repr(compiled)
    assert operation == "validate chat message index"
    assert compiled.params == (
        "chat_messages_session_id_id_idx",
        "chat_messages",
        "chat_schema",
        "chat_schema",
    )
    assert "pg_table_is_visible" in statement
    assert "pg_index" in statement
    assert "pg_am" in statement
    assert "indnkeyatts" in statement
    assert "indnatts" in statement
    assert "indkey[0]" in statement
    assert "indkey[1]" in statement
    assert "indisvalid" in statement
    assert "indisready" in statement
    assert "indisusable" in statement
    assert "indpred" in statement
    assert "indexprs" in statement


@pytest.mark.parametrize(
    "row",
    [
        _valid_chat_index_row(schema="other_schema"),
        _valid_chat_index_row(table="other_table"),
        _valid_chat_index_row(method="gin"),
        _valid_chat_index_row(key_count=1),
        _valid_chat_index_row(total_count=3),
        _valid_chat_index_row(first_key="id", second_key="session_id"),
        _valid_chat_index_row(valid=False),
        _valid_chat_index_row(ready=False),
        _valid_chat_index_row(usable=False),
        _valid_chat_index_row(predicate="partial predicate"),
        _valid_chat_index_row(expressions="index expression"),
    ],
    ids=[
        "wrong-schema",
        "wrong-table",
        "wrong-method",
        "wrong-key-count",
        "extra-included-column",
        "wrong-key-order",
        "invalid",
        "unready",
        "unusable",
        "partial",
        "expression",
    ],
)
def test_create_table_if_not_exists_rejects_wrong_named_index_definition(row):
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [row],
    }
    history = GaussDBChatMessageHistory(
        session_id="s1",
        schema_name="public",
        table_name="chat_messages",
        engine=engine,
    )

    with pytest.raises(GaussDBCapabilityError, match="different definition"):
        history.create_table_if_not_exists()

    assert len(engine.executed) == 2
    assert all(
        "DROP" not in _statement_repr(compiled) for compiled, _ in engine.executed
    )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("public", "chat_messages")],
        [_valid_chat_index_row(), _valid_chat_index_row()],
    ],
    ids=["missing", "malformed", "multiple"],
)
def test_create_table_if_not_exists_rejects_invalid_index_catalog_result(rows):
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": rows,
    }
    history = _history(engine)

    with pytest.raises(GaussDBCapabilityError, match="different definition"):
        history.create_table_if_not_exists()


@pytest.mark.parametrize("sqlstate", ["42P07", "42710", "23505"])
def test_create_table_if_not_exists_recovers_duplicate_index_race_when_matching(
    sqlstate,
):
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row()],
    }
    engine.failures["create chat message index"] = GaussDBSQLError(
        "duplicate index",
        sqlstate=sqlstate,
    )
    history = _history(engine)

    history.create_table_if_not_exists()

    assert [operation for _compiled, operation in engine.fetches] == [
        "validate chat message table",
        "validate chat message index",
    ]


def test_create_table_if_not_exists_duplicate_race_still_rejects_wrong_definition():
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row(table="other_table")],
    }
    engine.failures["create chat message index"] = GaussDBSQLError(
        "duplicate index",
        sqlstate="42P07",
    )
    history = _history(engine)

    with pytest.raises(GaussDBCapabilityError, match="different definition"):
        history.create_table_if_not_exists()


def test_create_table_if_not_exists_preserves_non_duplicate_index_error():
    engine = RecordingEngine()
    engine.rows_by_operation["validate chat message table"] = _valid_chat_table_rows()
    error = GaussDBSQLError("read only", sqlstate="25006")
    engine.failures["create chat message index"] = error
    history = _history(engine)

    with pytest.raises(GaussDBSQLError) as exc_info:
        history.create_table_if_not_exists()

    assert exc_info.value is error
    assert [operation for _compiled, operation in engine.fetches] == [
        "validate chat message table"
    ]


@pytest.mark.parametrize("sqlstate", ["42P07", "42710", "23505"])
def test_create_table_if_not_exists_recovers_duplicate_table_race_when_matching(
    sqlstate,
):
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(),
        "validate chat message index": [_valid_chat_index_row()],
    }
    engine.failures["create chat message table"] = GaussDBSQLError(
        "duplicate table",
        sqlstate=sqlstate,
    )
    history = _history(engine)

    history.create_table_if_not_exists()

    assert [operation for _compiled, operation in engine.executed] == [
        "create chat message index"
    ]
    assert [operation for _compiled, operation in engine.fetches] == [
        "validate chat message table",
        "validate chat message index",
    ]


@pytest.mark.parametrize(
    ("rows", "case"),
    [
        ([], "missing"),
        (_valid_chat_table_rows() * 2, "multiple-table"),
        (
            _valid_chat_table_rows()
            + [
                (
                    "public",
                    "chat_messages",
                    5,
                    "extra",
                    "text",
                    False,
                    False,
                    None,
                    1,
                    1,
                    1,
                    True,
                    True,
                    True,
                    None,
                    None,
                )
            ],
            "extra-column",
        ),
        (
            [
                _valid_chat_table_rows()[1],
                _valid_chat_table_rows()[0],
                *_valid_chat_table_rows()[2:],
            ],
            "wrong-order",
        ),
        (_changed_chat_table_rows(0, 4, "integer"), "wrong-id-type"),
        (_changed_chat_table_rows(0, 5, False), "nullable-id"),
        (_changed_chat_table_rows(0, 7, "0"), "wrong-id-default"),
        (_changed_chat_table_rows(0, 8, 2), "wrong-primary-key"),
        (_changed_chat_table_rows(1, 4, "varchar"), "wrong-session-type"),
        (_changed_chat_table_rows(1, 5, False), "nullable-session"),
        (_changed_chat_table_rows(2, 4, "text"), "wrong-message-type"),
        (_changed_chat_table_rows(2, 5, False), "nullable-message"),
        (
            _changed_chat_table_rows(3, 4, "timestamp with time zone"),
            "wrong-created-type",
        ),
        (_changed_chat_table_rows(3, 6, False), "missing-created-default"),
        (_changed_chat_table_rows(0, 0, "other_schema"), "wrong-schema"),
        (_changed_chat_table_rows(0, 1, "other_table"), "wrong-table"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_duplicate_table_race_rejects_wrong_table_contract(rows, case):
    del case
    engine = RecordingEngine()
    engine.rows_by_operation["validate chat message table"] = rows
    engine.failures["create chat message table"] = GaussDBSQLError(
        "duplicate table",
        sqlstate="42P07",
    )
    history = GaussDBChatMessageHistory(
        session_id="s1",
        schema_name="public",
        table_name="chat_messages",
        engine=engine,
    )

    with pytest.raises(GaussDBCapabilityError, match="table|definition"):
        history.create_table_if_not_exists()

    assert engine.executed == []
    assert [operation for _compiled, operation in engine.fetches] == [
        "validate chat message table"
    ]


def test_duplicate_table_race_validation_targets_explicit_schema_and_catalogs():
    engine = RecordingEngine()
    engine.rows_by_operation = {
        "validate chat message table": _valid_chat_table_rows(schema="chat_schema"),
        "validate chat message index": [_valid_chat_index_row(schema="chat_schema")],
    }
    engine.failures["create chat message table"] = GaussDBSQLError(
        "duplicate table",
        sqlstate="42710",
    )
    history = GaussDBChatMessageHistory(
        session_id="s1",
        schema_name="chat_schema",
        table_name="chat_messages",
        engine=engine,
    )

    history.create_table_if_not_exists()

    compiled, operation = engine.fetches[0]
    statement = _statement_repr(compiled)
    assert operation == "validate chat message table"
    assert compiled.params == ("chat_messages", "chat_schema", "chat_schema")
    assert "pg_class" in statement
    assert "pg_namespace" in statement
    assert "pg_attribute" in statement
    assert "pg_attrdef" in statement
    assert "pg_index" in statement
    assert "pg_table_is_visible" in statement


def test_create_table_if_not_exists_preserves_non_duplicate_table_error():
    engine = RecordingEngine()
    error = GaussDBSQLError("read only", sqlstate="25006")
    engine.failures["create chat message table"] = error
    history = _history(engine)

    with pytest.raises(GaussDBSQLError) as exc_info:
        history.create_table_if_not_exists()

    assert exc_info.value is error
    assert engine.executed == []
    assert engine.fetches == []


def test_constructor_create_table_failure_closes_owned_engine(monkeypatch):
    created_engines = []

    class FailingGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.fail_operations.add("create chat message table")
            created_engines.append(self)

    monkeypatch.setattr(chat_history_module, "GaussDBEngine", FailingGaussDBEngine)

    with pytest.raises(RuntimeError, match="create chat message table"):
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            dsn="host=127.0.0.1 dbname=unit",
            create_table=True,
        )

    assert created_engines[0].closed is True


def test_constructor_create_table_failure_does_not_close_external_engine():
    engine = RecordingEngine()
    engine.fail_operations.add("create chat message table")

    with pytest.raises(RuntimeError, match="create chat message table"):
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            engine=engine,
            create_table=True,
        )

    assert engine.closed is False


@pytest.mark.parametrize(
    "failure",
    [KeyboardInterrupt("stop table creation"), SystemExit("exit table creation")],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_constructor_base_exception_closes_owned_engine_and_preserves_primary(
    monkeypatch,
    failure,
):
    created_engines = []

    class InterruptingGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.failures["create chat message table"] = failure
            created_engines.append(self)

    monkeypatch.setattr(
        chat_history_module,
        "GaussDBEngine",
        InterruptingGaussDBEngine,
    )

    with pytest.raises(type(failure)) as exc_info:
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            dsn="host=127.0.0.1 dbname=unit",
            create_table=True,
        )

    assert exc_info.value is failure
    assert created_engines[0].closed is True


def test_constructor_cleanup_base_exception_does_not_replace_primary(monkeypatch):
    primary = SystemExit("exit table creation")
    created_engines = []

    class CleanupFailingGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.close_called = False
            self.failures["create chat message table"] = primary
            created_engines.append(self)

        def close(self) -> None:
            self.close_called = True
            raise KeyboardInterrupt("cleanup interrupted")

    monkeypatch.setattr(
        chat_history_module,
        "GaussDBEngine",
        CleanupFailingGaussDBEngine,
    )

    with pytest.raises(SystemExit) as exc_info:
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            dsn="host=127.0.0.1 dbname=unit",
            create_table=True,
        )

    assert exc_info.value is primary
    assert created_engines[0].close_called is True


def test_constructor_base_exception_does_not_close_external_engine():
    engine = RecordingEngine()
    failure = KeyboardInterrupt("stop table creation")
    engine.failures["create chat message table"] = failure

    with pytest.raises(KeyboardInterrupt) as exc_info:
        GaussDBChatMessageHistory(
            session_id="s1",
            table_name="chat_messages",
            engine=engine,
            create_table=True,
        )

    assert exc_info.value is failure
    assert engine.closed is False


def test_messages_selects_current_session_ordered_by_id():
    engine = RecordingEngine()
    history = _history(engine, session_id="current-session")

    assert history.messages == []

    assert len(engine.fetches) == 1
    compiled, operation = engine.fetches[0]
    statement = _statement_repr(compiled)
    assert operation == "get chat messages"
    assert compiled.params == ("current-session",)
    assert "SELECT message FROM" in statement
    assert "Identifier('chat_messages')" in statement
    assert "WHERE session_id=%s" in statement
    assert "ORDER BY id ASC" in statement


def test_messages_deserializes_dict_rows_to_langchain_messages():
    first = HumanMessage(content="hello")
    second = AIMessage(content="hi")
    engine = RecordingEngine()
    engine.rows = [(message_to_dict(first),), (message_to_dict(second),)]
    history = _history(engine)

    messages = history.messages

    assert [type(message) for message in messages] == [HumanMessage, AIMessage]
    assert [message.content for message in messages] == ["hello", "hi"]


def test_messages_deserializes_json_string_rows():
    first = HumanMessage(content="hello")
    second = AIMessage(content="hi")
    engine = RecordingEngine()
    engine.rows = [
        (json.dumps(message_to_dict(first)),),
        (json.dumps(message_to_dict(second)),),
    ]
    history = _history(engine)

    messages = history.messages

    assert [type(message) for message in messages] == [HumanMessage, AIMessage]
    assert [message.content for message in messages] == ["hello", "hi"]


def test_messages_rejects_invalid_json_row():
    engine = RecordingEngine()
    engine.rows = [("{not-json",)]
    history = _history(engine)

    with pytest.raises(ValueError, match="Invalid chat message JSON row"):
        history.messages


def test_add_messages_empty_sequence_is_noop():
    engine = RecordingEngine()
    history = _history(engine)

    history.add_messages([])

    assert engine.executed == []


def test_add_messages_uses_single_multi_row_insert():
    engine = RecordingEngine()
    history = _history(engine, session_id="session-1")
    first = HumanMessage(content="hello")
    second = AIMessage(content="hi")

    history.add_messages([first, second])

    assert len(engine.executed) == 1
    compiled, operation = engine.executed[0]
    statement = _statement_repr(compiled)
    assert operation == "add chat messages"
    assert "INSERT INTO" in statement
    assert "Identifier('chat_messages')" in statement
    assert "(session_id, message)" in statement
    assert statement.count("%s::jsonb") == 2
    assert compiled.params == (
        "session-1",
        json.dumps(message_to_dict(first)),
        "session-1",
        json.dumps(message_to_dict(second)),
    )


def test_add_messages_splits_large_insert_statements(monkeypatch):
    monkeypatch.setattr(chat_history_module, "_WRITE_BATCH_SIZE", 2)
    engine = RecordingEngine()
    history = _history(engine)

    history.add_messages(
        [
            HumanMessage(content="first"),
            AIMessage(content="second"),
            HumanMessage(content="third"),
        ]
    )

    assert [len(compiled.params) for compiled, _operation in engine.executed] == [4, 2]
    assert [item["type"] for item in _executed_json_messages(engine)] == [
        "human",
        "ai",
        "human",
    ]


def test_add_messages_preserves_duplicate_messages():
    engine = RecordingEngine()
    history = _history(engine)
    message = HumanMessage(content="repeat")

    history.add_messages([message, message])

    assert len(engine.executed) == 1
    compiled, _ = engine.executed[0]
    assert len(compiled.params) == 4
    assert compiled.params[1] == compiled.params[3]


def test_add_messages_rejects_non_base_message():
    history = _history()

    with pytest.raises(ValueError, match="BaseMessage"):
        history.add_messages([HumanMessage(content="ok"), object()])


def test_add_messages_rejects_non_json_serializable_message_with_index():
    history = _history()
    message = HumanMessage(
        content="hello",
        additional_kwargs={"not_json": object()},
    )

    with pytest.raises(ValueError) as exc_info:
        history.add_messages([message])

    error = str(exc_info.value)
    assert "message 0" in error
    assert "human" in error
    assert "JSON" in error


def test_add_messages_rejects_non_finite_json_number_without_sql():
    engine = RecordingEngine()
    history = _history(engine)
    message = HumanMessage(
        content="hello",
        additional_kwargs={"score": math.nan},
    )

    with pytest.raises(ValueError) as exc_info:
        history.add_messages([message])

    error = str(exc_info.value)
    assert "message 0" in error
    assert "human" in error
    assert "JSON" in error
    assert engine.executed == []


@pytest.mark.parametrize("messages", [None, "not-a-message-list"])
def test_add_messages_rejects_none_and_string_iterable(messages):
    history = _history()

    with pytest.raises(ValueError, match="messages"):
        history.add_messages(messages)


def test_inherited_add_message_delegates_to_add_messages():
    engine = RecordingEngine()
    history = _history(engine)

    history.add_message(HumanMessage(content="hello"))

    assert len(engine.executed) == 1
    compiled, _ = engine.executed[0]
    assert "INSERT INTO" in _statement_repr(compiled)
    assert len(compiled.params) == 2


def test_inherited_add_user_and_ai_message_are_persisted():
    engine = RecordingEngine()
    history = _history(engine)

    history.add_user_message("u")
    history.add_ai_message("a")

    assert len(engine.executed) == 2
    stored_messages = _executed_json_messages(engine)
    assert [message["type"] for message in stored_messages] == ["human", "ai"]
    assert [message["data"]["content"] for message in stored_messages] == ["u", "a"]


def test_add_messages_preserves_rich_message_fields():
    engine = RecordingEngine()
    history = _history(engine)
    ai_message = AIMessage(
        content="answer",
        id="ai-id",
        name="assistant-name",
        response_metadata={"finish_reason": "tool_calls"},
        additional_kwargs={"vendor": {"trace": "abc"}},
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "GaussDB"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    tool_message = ToolMessage(
        content="tool result",
        id="tool-id",
        name="tool-name",
        tool_call_id="call-1",
    )

    history.add_messages([ai_message, tool_message])

    compiled, _ = engine.executed[0]
    restored = messages_from_dict(
        [json.loads(compiled.params[1]), json.loads(compiled.params[3])]
    )
    restored_ai, restored_tool = restored
    assert isinstance(restored_ai, AIMessage)
    assert restored_ai.id == "ai-id"
    assert restored_ai.name == "assistant-name"
    assert restored_ai.response_metadata == {"finish_reason": "tool_calls"}
    assert restored_ai.additional_kwargs == {"vendor": {"trace": "abc"}}
    assert restored_ai.tool_calls == [
        {
            "name": "lookup",
            "args": {"query": "GaussDB"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]
    assert isinstance(restored_tool, ToolMessage)
    assert restored_tool.id == "tool-id"
    assert restored_tool.name == "tool-name"
    assert restored_tool.tool_call_id == "call-1"


def test_clear_deletes_only_current_session():
    engine = RecordingEngine()
    history = _history(engine, session_id="current-session")

    history.clear()

    assert len(engine.executed) == 1
    compiled, operation = engine.executed[0]
    statement = _statement_repr(compiled)
    assert operation == "clear chat messages"
    assert compiled.params == ("current-session",)
    assert "DELETE FROM" in statement
    assert "WHERE session_id = %s" in statement
    assert "TRUNCATE" not in statement.upper()


@pytest.mark.asyncio
async def test_async_methods_inherit_langchain_executor_fallback():
    engine = RecordingEngine()
    engine.rows = [(message_to_dict(HumanMessage(content="stored")),)]
    history = _history(engine)
    event_loop_thread = threading.get_ident()

    assert "aget_messages" not in GaussDBChatMessageHistory.__dict__
    assert "aadd_messages" not in GaussDBChatMessageHistory.__dict__
    assert "aclear" not in GaussDBChatMessageHistory.__dict__

    messages = await history.aget_messages()
    await history.aadd_messages([HumanMessage(content="new")])
    await history.aclear()

    assert [message.content for message in messages] == ["stored"]
    assert all(thread_id != event_loop_thread for thread_id in engine.fetch_threads)
    assert all(thread_id != event_loop_thread for thread_id in engine.execute_threads)
    assert [operation for _, operation in engine.executed] == [
        "add chat messages",
        "clear chat messages",
    ]
    assert [operation for _, operation in engine.fetches] == ["get chat messages"]


@pytest.mark.asyncio
async def test_concurrent_async_adds_use_sync_entrypoint_via_executor():
    engine = RecordingEngine()
    history = _history(engine)
    event_loop_thread = threading.get_ident()

    await asyncio.gather(
        history.aadd_messages([HumanMessage(content="first")]),
        history.aadd_messages([HumanMessage(content="second")]),
    )

    assert len(engine.executed) == 2
    assert all(thread_id != event_loop_thread for thread_id in engine.execute_threads)
    assert engine.fetches == []
    assert [operation for _, operation in engine.executed] == [
        "add chat messages",
        "add chat messages",
    ]


def test_str_uses_langchain_buffer_string():
    engine = RecordingEngine()
    engine.rows = [
        (message_to_dict(HumanMessage(content="hello")),),
        (message_to_dict(AIMessage(content="hi")),),
    ]
    history = _history(engine)

    rendered = str(history)

    assert "Human: hello" in rendered
    assert "AI: hi" in rendered


def test_database_write_error_preserves_sqlstate_and_redacts_context():
    engine = RecordingEngine()
    error = GaussDBSQLError(
        "add chat messages failed; context: password=***; cause: duplicate key",
        sqlstate="23505",
    )
    engine.failures["add chat messages"] = error
    history = _history(engine)

    with pytest.raises(GaussDBSQLError) as exc_info:
        history.add_messages([HumanMessage(content="hello")])

    assert exc_info.value is error
    assert exc_info.value.sqlstate == "23505"
    assert "super-secret" not in str(exc_info.value)
    assert "password=***" in str(exc_info.value)


def test_database_read_error_preserves_sqlstate():
    engine = RecordingEngine()
    error = GaussDBSQLError(
        "get chat messages failed; cause: database unavailable",
        sqlstate="08006",
    )
    engine.failures["get chat messages"] = error
    history = _history(engine)

    with pytest.raises(GaussDBSQLError) as exc_info:
        history.messages

    assert exc_info.value is error
    assert exc_info.value.sqlstate == "08006"


def test_messages_rejects_malformed_message_dict_row_with_index():
    engine = RecordingEngine()
    engine.rows = [({},)]
    history = _history(engine)

    with pytest.raises(ValueError) as exc_info:
        history.messages

    error = str(exc_info.value)
    assert "row 0" in error
    assert "type" in error
    assert "data" in error


def test_messages_wraps_langchain_deserialization_error_without_row_payload():
    engine = RecordingEngine()
    engine.rows = [({"type": "human", "data": {"content": {"secret": "dont-leak"}}},)]
    history = _history(engine)

    with pytest.raises(ValueError) as exc_info:
        history.messages

    error = str(exc_info.value)
    assert "row 0" in error
    assert "human" in error
    assert "dont-leak" not in error
    assert "content" not in error


def test_close_closes_only_owned_engine(monkeypatch):
    external_engine = RecordingEngine()
    external_history = GaussDBChatMessageHistory(
        session_id="s1",
        table_name="chat_messages",
        engine=external_engine,
    )

    external_history.close()

    assert external_engine.closed is False

    created_engines = []

    class FakeGaussDBEngine(RecordingEngine):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            created_engines.append(self)

    monkeypatch.setattr(chat_history_module, "GaussDBEngine", FakeGaussDBEngine)
    owned_history = GaussDBChatMessageHistory(
        session_id="s1",
        table_name="chat_messages",
        dsn="host=127.0.0.1 dbname=unit",
    )

    owned_history.close()

    assert created_engines[0].closed is True
