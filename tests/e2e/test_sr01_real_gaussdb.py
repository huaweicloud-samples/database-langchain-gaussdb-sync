import os
import uuid

import pytest
from psycopg2 import sql

from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.sql import (
    CompiledSQL,
    build_odku_insert,
    identifier,
    qualified_name,
)

pytestmark = pytest.mark.gaussdb_e2e


def _dsn():
    value = os.getenv("GAUSSDB_TEST_DSN")
    if not value:
        pytest.skip("GAUSSDB_TEST_DSN is not set")
    return value


def _table_name():
    return "sr01_" + uuid.uuid4().hex[:12]


def _drop_table(engine, table, *, strict):
    try:
        engine.execute(
            CompiledSQL(
                sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
            ),
            operation="drop sr01 e2e table",
        )
    except Exception:
        if strict:
            raise


def _enable_session_writes(engine):
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable sr01 e2e writes",
    )


def _close_engine(engine, *, strict):
    try:
        engine.close()
    except Exception:
        if strict:
            raise


def test_real_gaussdb_connects_and_selects_one():
    engine = GaussDBEngine(dsn=_dsn())
    completed = False
    try:
        rows = engine.fetch_all(CompiledSQL(sql.SQL("SELECT %s"), [1]))
        assert rows[0][0] == 1
        completed = True
    finally:
        _close_engine(engine, strict=completed)


@pytest.mark.gaussdb_e2e_fast
def test_real_gaussdb_odku_updates_existing_row():
    engine = GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("CREATE TABLE {} (id text PRIMARY KEY, content text)").format(
                    qualified_name(None, table)
                )
            ),
            operation="create sr01 odku e2e table",
        )
        insert_old = build_odku_insert(
            schema=None,
            table=table,
            insert_columns=["id", "content"],
            update_columns=["content"],
            rows=[("doc-1", "old")],
        )
        insert_new = build_odku_insert(
            schema=None,
            table=table,
            insert_columns=["id", "content"],
            update_columns=["content"],
            rows=[("doc-1", "new")],
        )

        engine.execute(insert_old, operation="insert old sr01 odku row")
        engine.execute(insert_new, operation="update sr01 odku row")
        rows = engine.fetch_all(
            CompiledSQL(
                sql.SQL("SELECT content FROM {} WHERE id = %s").format(
                    qualified_name(None, table)
                ),
                ["doc-1"],
            )
        )

        assert rows == [("new",)]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_transaction_rollback_reuses_connection():
    engine = GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)
    table = _table_name()
    completed = False
    try:
        _enable_session_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("CREATE TABLE {} (id text PRIMARY KEY)").format(
                    qualified_name(None, table)
                )
            ),
            operation="create sr01 rollback e2e table",
        )

        class RollbackSentinel(Exception):
            pass

        def insert_then_fail(cursor):
            cursor.execute(
                sql.SQL("INSERT INTO {} (id) VALUES (%s)").format(
                    qualified_name(None, table)
                ),
                ["doc-1"],
            )
            raise RollbackSentinel("force rollback after successful insert")

        with pytest.raises(RollbackSentinel, match="force rollback"):
            engine.transaction(
                insert_then_fail,
                operation="rollback sr01 inserted row",
            )

        rows = engine.fetch_all(
            CompiledSQL(
                sql.SQL("SELECT count(*) FROM {}").format(qualified_name(None, table))
            )
        )

        assert rows[0][0] == 0
        assert engine.check_connection() is True
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)


def test_real_gaussdb_identifier_and_params_roundtrip():
    engine = GaussDBEngine(dsn=_dsn(), minconn=1, maxconn=1)
    table = _table_name()
    column = "safe_column"
    completed = False
    try:
        _enable_session_writes(engine)
        engine.execute(
            CompiledSQL(
                sql.SQL("CREATE TABLE {} ({} text)").format(
                    qualified_name(None, table),
                    identifier(column),
                )
            ),
            operation="create sr01 identifier e2e table",
        )
        engine.execute(
            CompiledSQL(
                sql.SQL("INSERT INTO {} ({}) VALUES (%s)").format(
                    qualified_name(None, table),
                    identifier(column),
                ),
                ["hello"],
            ),
            operation="insert sr01 identifier e2e row",
        )
        rows = engine.fetch_all(
            CompiledSQL(
                sql.SQL("SELECT {} FROM {}").format(
                    identifier(column),
                    qualified_name(None, table),
                )
            )
        )

        assert rows == [("hello",)]
        completed = True
    finally:
        try:
            _drop_table(engine, table, strict=completed)
        finally:
            _close_engine(engine, strict=completed)
