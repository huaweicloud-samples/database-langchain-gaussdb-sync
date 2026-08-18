from __future__ import annotations

import re

import pytest
from psycopg2 import sql

from langchain_gaussdb.errors import GaussDBSQLBuildError
from langchain_gaussdb.indexes import (
    build_create_bm25_index,
    build_create_metadata_index,
    build_create_vector_index,
)


def _statement(compiled) -> str:
    assert compiled.params == ()
    return repr(compiled.statement)


def test_build_vector_index_uses_gsdiskann() -> None:
    index_name, compiled = build_create_vector_index(
        "public", "docs", "embedding", 1536
    )

    assert index_name == "docs_embedding_gsdiskann_cosine_idx"
    statement = _statement(compiled)
    assert "CREATE INDEX" in statement
    assert "IF NOT EXISTS" in statement
    assert "USING gsdiskann" in statement
    assert "COSINE" in statement


def test_build_vector_index_supports_gsdiskann_at_4096_dimensions() -> None:
    index_name, compiled = build_create_vector_index(
        "public", "docs", "embedding", 4096
    )

    assert index_name == "docs_embedding_gsdiskann_cosine_idx"
    statement = _statement(compiled)
    assert "USING gsdiskann" in statement


def test_build_vector_index_rejects_dimensions_above_4096() -> None:
    with pytest.raises(GaussDBSQLBuildError, match="4096"):
        build_create_vector_index("public", "docs", "embedding", 4097)


def test_build_vector_index_rejects_invalid_params() -> None:
    with pytest.raises(GaussDBSQLBuildError, match="relation option"):
        build_create_vector_index("public", "docs", "embedding", 1536, raw="x")

    with pytest.raises(GaussDBSQLBuildError, match="enable_pq"):
        build_create_vector_index("public", "docs", "embedding", 1536, enable_pq="yes")

    with pytest.raises(GaussDBSQLBuildError, match="enable_pq"):
        build_create_vector_index("public", "docs", "embedding", 1536, enable_pq=1)

    with pytest.raises(GaussDBSQLBuildError, match="pq_nclus"):
        build_create_vector_index("public", "docs", "embedding", 1536, pq_nclus=0)


@pytest.mark.parametrize("legacy_kind", ["auto", "gsdiskann", "none"])
def test_build_vector_index_rejects_removed_kind_option(legacy_kind: str) -> None:
    with pytest.raises(GaussDBSQLBuildError, match="relation option kind"):
        build_create_vector_index(
            "public",
            "docs",
            "embedding",
            1536,
            kind=legacy_kind,
        )


def test_build_vector_index_rejects_legacy_positional_kind() -> None:
    with pytest.raises(TypeError):
        build_create_vector_index("public", "docs", "embedding", 1536, "auto")


def test_build_vector_index_allows_pq_nseg_equal_to_dimension() -> None:
    _index_name, compiled = build_create_vector_index(
        "public", "docs", "embedding", 3, pq_nseg=3
    )

    statement = _statement(compiled)
    assert "pq_nseg" in statement
    assert "SQL('3')" in statement


def test_build_vector_index_rejects_pq_nseg_greater_than_dimension() -> None:
    with pytest.raises(GaussDBSQLBuildError, match="pq_nseg"):
        build_create_vector_index("public", "docs", "embedding", 3, pq_nseg=4)


@pytest.mark.parametrize(
    ("distance_strategy", "metric"),
    [
        ("cosine", "COSINE"),
        ("l2", "L2"),
    ],
)
def test_build_vector_indexes_render_distance_metric(
    distance_strategy: str,
    metric: str,
) -> None:
    _index_name, compiled = build_create_vector_index(
        "public",
        "docs",
        "embedding",
        3,
        distance_strategy=distance_strategy,
    )

    statement = _statement(compiled)
    assert f"SQL('{metric}')" in statement
    other_metric = "L2" if metric == "COSINE" else "COSINE"
    assert f"SQL('{other_metric}')" not in statement


@pytest.mark.parametrize(
    ("distance_strategy", "expected_name"),
    [
        ("cosine", "docs_embedding_gsdiskann_cosine_idx"),
        ("l2", "docs_embedding_gsdiskann_l2_idx"),
    ],
)
def test_build_vector_index_default_name_includes_metric(
    distance_strategy: str,
    expected_name: str,
) -> None:
    index_name, _compiled = build_create_vector_index(
        "public",
        "docs",
        "embedding",
        3,
        distance_strategy=distance_strategy,
    )

    assert index_name == expected_name


@pytest.mark.parametrize(
    "distance_strategy",
    ["", "COSINE", "euclidean", "inner_product", None, 1],
)
def test_build_vector_index_rejects_invalid_distance_strategy(
    distance_strategy,
) -> None:
    with pytest.raises(GaussDBSQLBuildError) as exc_info:
        build_create_vector_index(
            "public",
            "docs",
            "embedding",
            3,
            distance_strategy=distance_strategy,
        )

    assert str(exc_info.value) == "distance_strategy must be one of: cosine, l2"


def test_build_vector_index_uses_idempotent_ddl() -> None:
    _index_name, compiled = build_create_vector_index(
        "public",
        "docs",
        "embedding",
        3,
    )

    statement = _statement(compiled)
    assert "CREATE INDEX" in statement
    assert "IF NOT EXISTS" in statement


def test_build_vector_index_shortens_overlong_automatic_name_stably() -> None:
    table = "table_" + "x" * 34
    column = "embedding_column_" + "y" * 4

    first_name, first = build_create_vector_index("public", table, column, 3)
    second_name, second = build_create_vector_index("public", table, column, 3)

    assert first_name == second_name
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first_name)
    assert len(first_name.encode("utf-8")) <= 63
    assert first_name.endswith("_idx")
    assert _statement(first) == _statement(second)


def test_build_vector_index_shortens_multibyte_automatic_name_by_utf8_bytes() -> None:
    table = "表" * 21

    first_name, first = build_create_vector_index("public", table, "embedding", 3)
    second_name, second = build_create_vector_index("public", table, "embedding", 3)

    assert first_name == second_name
    assert len(first_name.encode("utf-8")) <= 63
    assert first_name.endswith("_idx")
    assert _statement(first) == _statement(second)


def test_build_vector_index_rejects_overlong_explicit_name() -> None:
    with pytest.raises(GaussDBSQLBuildError) as exc_info:
        build_create_vector_index(
            "public",
            "docs",
            "embedding",
            3,
            index_name="i" * 64,
        )

    assert str(exc_info.value) == "index_name must not exceed 63 UTF-8 bytes"


def test_build_metadata_index_jsonb_exact_expression() -> None:
    index_name, compiled = build_create_metadata_index(
        "public", "docs", "metadata", "tenant_id"
    )

    assert index_name == "docs_metadata_tenant_id_json_text_idx"
    statement = _statement(compiled)
    assert "CREATE INDEX IF NOT EXISTS" in statement
    assert "metadata" in statement
    assert "SQL(' ((')" in statement
    assert "SQL('->')" in statement
    assert "SQL('->>')" not in statement
    assert "::text" in statement
    assert "Literal('tenant_id')" in statement
    assert "SQL('))')" in statement
    assert "tenant_id" in statement


def test_build_metadata_index_cast_expression() -> None:
    index_name, compiled = build_create_metadata_index(
        "public", "docs", "metadata", "score", cast="float"
    )

    assert index_name == "docs_metadata_score_float_idx"
    statement = _statement(compiled)
    assert "::" in statement
    assert "SQL('->>')" in statement
    assert "Literal('score')" in statement
    assert "SQL('float')" in statement
    assert "float" in statement

    with pytest.raises(GaussDBSQLBuildError, match="cast"):
        build_create_metadata_index(
            "public", "docs", "metadata", "score", cast="numeric(10,2)"
        )


def test_build_metadata_index_date_expression_uses_immutable_iso_text() -> None:
    index_name, compiled = build_create_metadata_index(
        "public", "docs", "metadata", "event_date", cast="date"
    )

    assert index_name == "docs_metadata_event_date_date_idx"
    statement = _statement(compiled)
    assert "to_date" not in statement
    assert "SQL('->>')" in statement
    assert "SQL('date')" not in statement


@pytest.mark.parametrize(
    "key",
    [
        "source.name",
        "source-name",
        "1source",
        "来源 名称",
        "source name!",
    ],
)
def test_build_metadata_index_allows_one_level_literal_keys(key: str) -> None:
    first_name, first = build_create_metadata_index(
        "public",
        "docs",
        "metadata",
        key,
    )
    second_name, second = build_create_metadata_index(
        "public",
        "docs",
        "metadata",
        key,
    )

    assert first_name == second_name
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first_name)
    assert len(first_name.encode("utf-8")) <= 63
    assert re.fullmatch(
        r"docs_metadata_json_text_[0-9a-f]{12}_idx",
        first_name,
    )
    assert repr(sql.Literal(key)) in _statement(first)
    assert _statement(first) == _statement(second)


def test_build_metadata_index_hashes_long_automatic_names_deterministically() -> None:
    key = "来源。" * 80

    first_name, _first = build_create_metadata_index(
        "public",
        "docs",
        "metadata",
        key,
        cast="date",
    )
    second_name, _second = build_create_metadata_index(
        "public",
        "docs",
        "metadata",
        key,
        cast="date",
    )

    assert first_name == second_name
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first_name)
    assert len(first_name.encode("utf-8")) <= 63


@pytest.mark.parametrize("key", ["", "$tenant", "bad\0key", 1, None])
def test_build_metadata_index_rejects_only_invalid_one_level_keys(key) -> None:
    with pytest.raises(GaussDBSQLBuildError):
        build_create_metadata_index("public", "docs", "metadata", key)


@pytest.mark.parametrize(
    "index_name",
    ["source-name", "1source", "索引", "My Index", 'Ix"Name'],
)
def test_build_metadata_index_accepts_explicit_names_that_are_safely_quoted(
    index_name: str,
) -> None:
    returned_name, compiled = build_create_metadata_index(
        "public",
        "docs",
        "metadata",
        "source.name",
        index_name=index_name,
    )

    assert returned_name == index_name
    assert repr(sql.Identifier(index_name)) in _statement(compiled)


def test_build_vector_index_accepts_quoted_unicode_public_identifiers() -> None:
    index_name, compiled = build_create_vector_index(
        "My Schema",
        "向量 Table",
        'Embedding"Vector',
        3,
        index_name='Ix"Dense',
    )

    statement = _statement(compiled)
    assert index_name == 'Ix"Dense'
    for name in ("My Schema", "向量 Table", 'Embedding"Vector', 'Ix"Dense'):
        assert repr(sql.Identifier(name)) in statement


@pytest.mark.parametrize("index_name", ["bad.index", "bad\0index", "i" * 64])
def test_build_metadata_index_rejects_unsafe_explicit_names(
    index_name: str,
) -> None:
    with pytest.raises(GaussDBSQLBuildError, match="index_name"):
        build_create_metadata_index(
            "public",
            "docs",
            "metadata",
            "source.name",
            index_name=index_name,
        )


def test_build_bm25_index_quotes_field() -> None:
    bm25_name, bm25 = build_create_bm25_index("public", "docs", "content")

    assert bm25_name == "docs_content_bm25_idx"
    bm25_statement = _statement(bm25)
    assert "CREATE INDEX" in bm25_statement
    assert "IF NOT EXISTS" in bm25_statement
    assert "USING bm25" in bm25_statement

    with pytest.raises(GaussDBSQLBuildError):
        build_create_bm25_index("public", "docs", "content.bad")
