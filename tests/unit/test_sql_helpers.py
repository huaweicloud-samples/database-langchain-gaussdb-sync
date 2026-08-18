import pytest
from psycopg2 import sql

from langchain_gaussdb.errors import GaussDBSQLBuildError
from langchain_gaussdb.sql import (
    CompiledSQL,
    identifier,
    qualified_name,
)


def test_identifier_rejects_empty_name():
    with pytest.raises(GaussDBSQLBuildError, match="identifier"):
        identifier("")


def test_identifier_rejects_whitespace_only_name():
    with pytest.raises(GaussDBSQLBuildError, match="identifier"):
        identifier("   ")


def test_identifier_rejects_dotted_name():
    with pytest.raises(GaussDBSQLBuildError, match="single identifier"):
        identifier("public.documents")


def test_identifier_rejects_nul_character():
    with pytest.raises(GaussDBSQLBuildError, match="NUL"):
        identifier("documents\x00archive")


@pytest.mark.parametrize(
    "name",
    [
        "a" * 63,
        "界" * 21,
    ],
    ids=["ascii-63-bytes", "unicode-63-bytes"],
)
def test_identifier_accepts_exactly_63_utf8_bytes(name):
    assert isinstance(identifier(name), sql.Identifier)


@pytest.mark.parametrize(
    "name",
    [
        "a" * 64,
        "界" * 21 + "a",
    ],
    ids=["ascii-64-bytes", "unicode-64-bytes"],
)
def test_identifier_rejects_more_than_63_utf8_bytes(name):
    with pytest.raises(GaussDBSQLBuildError, match="63 UTF-8 bytes"):
        identifier(name)


def test_identifier_returns_psycopg_identifier():
    value = identifier("documents")
    assert isinstance(value, sql.Identifier)


@pytest.mark.parametrize(
    "name",
    ["My Table", "表", 'Ix"Name'],
    ids=["space-and-case", "unicode", "embedded-quote"],
)
def test_identifier_accepts_names_that_require_quoting(name):
    value = identifier(name)

    assert value == sql.Identifier(name)
    assert name in repr(value)


def test_qualified_name_uses_two_identifiers():
    value = qualified_name("public", "documents")
    assert isinstance(value, sql.Composed)
    rendered = repr(value)
    assert "Identifier('public')" in rendered
    assert "Identifier('documents')" in rendered


def test_qualified_name_rejects_whitespace_schema():
    with pytest.raises(GaussDBSQLBuildError, match="schema"):
        qualified_name("   ", "documents")


def test_compiled_sql_keeps_statement_and_params_separate():
    compiled = CompiledSQL(sql.SQL("SELECT %s"), ["value"])
    assert compiled.statement == sql.SQL("SELECT %s")
    assert compiled.params == ("value",)
