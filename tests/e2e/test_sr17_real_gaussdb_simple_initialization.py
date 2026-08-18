from __future__ import annotations

import os
import uuid

import pytest
from psycopg2 import sql

from langchain_gaussdb import GaussDBCapabilityError, GaussDBEngine, GaussDBVectorStore
from langchain_gaussdb.sql import CompiledSQL, qualified_name
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings

pytestmark = [pytest.mark.gaussdb_e2e, pytest.mark.gaussdb_e2e_fast]


def _engine() -> GaussDBEngine:
    dsn = os.getenv("GAUSSDB_TEST_DSN")
    if not dsn:
        pytest.skip("GAUSSDB_TEST_DSN is not configured")
    engine = GaussDBEngine(dsn=dsn)
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only = off")),
        operation="enable sr17 test writes",
    )
    return engine


def _drop(engine: GaussDBEngine, table: str) -> None:
    engine.execute(
        CompiledSQL(
            sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_name(None, table))
        ),
        operation="drop sr17 table",
    )


def _store(engine: GaussDBEngine, table: str) -> GaussDBVectorStore:
    return GaussDBVectorStore(
        engine=engine,
        embedding=DeterministicEmbeddings(dimension=3),
        embedding_dimension=3,
        table_name=table,
    )


def test_existing_table_accepts_any_types_when_required_names_exist() -> None:
    engine = _engine()
    table = f"sr17_weak_{uuid.uuid4().hex[:10]}"
    try:
        engine.execute(
            CompiledSQL(
                sql.SQL(
                    "CREATE TABLE {} (id integer, content bigint, "
                    "metadata text, embedding text)"
                ).format(qualified_name(None, table))
            ),
            operation="create sr17 weak-shape table",
        )

        _store(engine, table)._prepare_table_if_needed()
    finally:
        _drop(engine, table)
        engine.close()


def test_existing_table_still_rejects_missing_required_name() -> None:
    engine = _engine()
    table = f"sr17_missing_{uuid.uuid4().hex[:10]}"
    try:
        engine.execute(
            CompiledSQL(
                sql.SQL(
                    "CREATE TABLE {} (id text, content text, embedding text)"
                ).format(qualified_name(None, table))
            ),
            operation="create sr17 missing-column table",
        )

        with pytest.raises(GaussDBCapabilityError, match="metadata"):
            _store(engine, table)._prepare_table_if_needed()
    finally:
        _drop(engine, table)
        engine.close()
