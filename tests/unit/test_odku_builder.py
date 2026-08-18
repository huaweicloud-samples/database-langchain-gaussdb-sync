from psycopg2 import sql

from langchain_gaussdb.sql import build_odku_insert


def _walk_sql_nodes(node):
    yield node
    if isinstance(node, sql.Composed):
        for child in node._wrapped:
            yield from _walk_sql_nodes(child)


def _identifier_parts(node):
    return tuple(getattr(node, "_wrapped", ()))


def _statement_repr(compiled):
    return repr(compiled.statement)


def test_build_odku_uses_insert_on_duplicate_key_update():
    compiled = build_odku_insert(
        schema="public",
        table="documents",
        insert_columns=["id", "content"],
        update_columns=["content"],
        rows=[("1", "hello")],
    )

    statement = _statement_repr(compiled)
    assert "ON DUPLICATE KEY UPDATE" in statement
    assert "ON CONFLICT" not in statement
    assert "MERGE" not in statement
    assert compiled.params == ("1", "hello")


def test_build_odku_preserves_identifier_and_placeholder_nodes():
    compiled = build_odku_insert(
        schema="public",
        table="documents",
        insert_columns=["id", "content", "metadata"],
        update_columns=["content", "metadata"],
        rows=[("1", "hello", "{}"), ("2", "bye", '{"source":"test"}')],
    )

    nodes = tuple(_walk_sql_nodes(compiled.statement))
    identifiers = [
        _identifier_parts(node) for node in nodes if isinstance(node, sql.Identifier)
    ]
    sql_chunks = [node._wrapped for node in nodes if isinstance(node, sql.SQL)]

    assert identifiers == [
        ("public",),
        ("documents",),
        ("id",),
        ("content",),
        ("metadata",),
        ("content",),
        ("content",),
        ("metadata",),
        ("metadata",),
    ]
    assert sum(isinstance(node, sql.Placeholder) for node in nodes) == 6
    assert " ON DUPLICATE KEY UPDATE " in sql_chunks
    assert " = VALUES(" in sql_chunks


def test_build_odku_accepts_internal_value_expressions():
    compiled = build_odku_insert(
        schema=None,
        table="documents",
        insert_columns=["id", "event_date"],
        update_columns=["event_date"],
        rows=[("doc-1", "2026-07-01")],
        value_expressions={
            "event_date": sql.SQL("to_date({}, 'YYYY-MM-DD')").format(sql.Placeholder())
        },
    )

    statement = _statement_repr(compiled)
    assert "to_date" in statement
    assert "YYYY-MM-DD" in statement
    assert compiled.params == ("doc-1", "2026-07-01")


def test_build_odku_keeps_malicious_payload_out_of_statement_repr():
    malicious_content = "x'); DROP TABLE documents; --"

    compiled = build_odku_insert(
        schema=None,
        table="documents",
        insert_columns=["id", "content"],
        update_columns=["content"],
        rows=[("doc-1", malicious_content)],
    )

    statement = _statement_repr(compiled)
    assert malicious_content not in statement
    assert "DROP TABLE" not in statement
    assert compiled.params == ("doc-1", malicious_content)


def test_build_odku_flattens_row_params_in_order():
    compiled = build_odku_insert(
        schema=None,
        table="documents",
        insert_columns=["id", "content"],
        update_columns=["content"],
        rows=[("1", "a"), ("2", "b")],
    )

    assert compiled.params == ("1", "a", "2", "b")


def test_build_odku_does_not_inspect_business_ids():
    compiled = build_odku_insert(
        schema=None,
        table="documents",
        insert_columns=["id", "content"],
        update_columns=["content"],
        rows=[("same-id", "a"), ("same-id", "b")],
    )

    assert compiled.params == ("same-id", "a", "same-id", "b")
