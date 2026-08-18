from __future__ import annotations

import pytest
from langchain_core.documents import Document
from psycopg2 import sql

import langchain_gaussdb.hybrid_search as hybrid_search
from langchain_gaussdb.filters import CompiledFilter
from langchain_gaussdb.hybrid_search import (
    BM25Config,
    HybridFusionConfig,
    build_bm25_search_sql,
    fuse_hybrid_results,
)


def test_bm25_config_defaults_to_content_column() -> None:
    assert BM25Config().column == "content"


def test_build_bm25_search_sql_uses_score_desc() -> None:
    compiled = build_bm25_search_sql(
        schema=None,
        table="documents",
        id_column="id",
        content_column="content",
        metadata_column="metadata",
        bm25_column="content",
        query="GaussDB BM25",
        k=3,
        compiled_filter=None,
    )

    statement = repr(compiled.statement)
    assert "###" in statement
    assert "AS bm25_score" in statement
    assert "ORDER BY bm25_score DESC" in statement
    assert "WHERE" not in statement
    assert compiled.params == ("GaussDB BM25", 3)


def test_build_bm25_search_sql_uses_query_filter_limit_param_order() -> None:
    compiled_filter = CompiledFilter(
        sql.SQL("metadata->>%s = %s"),
        ("tenant", "t1"),
        ("tenant",),
    )

    compiled = build_bm25_search_sql(
        schema=None,
        table="documents",
        id_column="id",
        content_column="content",
        metadata_column="metadata",
        bm25_column="content",
        query="GaussDB",
        k=2,
        compiled_filter=compiled_filter,
    )

    assert "WHERE" in repr(compiled.statement)
    assert compiled.params == ("GaussDB", "tenant", "t1", 2)


def test_bm25_catalog_probe_builders_are_not_exposed() -> None:
    assert not hasattr(hybrid_search, "build_bm25_index_probe_sql")
    assert not hasattr(hybrid_search, "build_bm25_operator_probe_sql")


def test_rrf_fusion_merges_dense_and_bm25_hits() -> None:
    dense_doc = Document(id="same", page_content="dense", metadata={})
    sparse_doc = Document(id="same", page_content="sparse", metadata={})

    fused = fuse_hybrid_results(
        [(dense_doc, 0.01)],
        [(sparse_doc, 9.0)],
        k=1,
        config=HybridFusionConfig(),
    )

    assert fused[0][0].id == "same"
    assert fused[0][1] > 0


def test_rrf_fusion_keeps_dense_only_and_bm25_only_candidates() -> None:
    config = HybridFusionConfig()
    fused = fuse_hybrid_results(
        [(Document(id="dense", page_content="dense", metadata={}), 0.01)],
        [(Document(id="bm25", page_content="bm25", metadata={}), 5.0)],
        k=2,
        config=config,
    )

    assert {doc.id for doc, _score in fused} == {"dense", "bm25"}
    scores = {str(doc.id): score for doc, score in fused}
    assert scores["dense"] == pytest.approx(config.dense_weight / (config.rrf_k + 1))
    assert scores["bm25"] == pytest.approx(config.sparse_weight / (config.rrf_k + 1))


def test_rrf_fusion_does_not_reward_a_missing_branch_past_rank_1000() -> None:
    config = HybridFusionConfig()
    dense_results = [
        (Document(id=f"dense-{index}", page_content="dense", metadata={}), 0.0)
        for index in range(1, 1002)
    ]
    bm25_results = [(Document(id="bm25-only", page_content="bm25", metadata={}), 0.0)]

    fused = fuse_hybrid_results(
        dense_results,
        bm25_results,
        k=1002,
        config=config,
    )

    scores = {str(doc.id): score for doc, score in fused}
    assert scores["dense-1001"] == pytest.approx(
        config.dense_weight / (config.rrf_k + 1001)
    )
    assert scores["bm25-only"] == pytest.approx(
        config.sparse_weight / (config.rrf_k + 1)
    )


def test_rrf_fusion_orders_by_hybrid_score_desc_and_limits_k() -> None:
    fused = fuse_hybrid_results(
        [
            (Document(id="a", page_content="a", metadata={}), 0.01),
            (Document(id="b", page_content="b", metadata={}), 0.02),
        ],
        [(Document(id="b", page_content="b", metadata={}), 9.0)],
        k=1,
        config=HybridFusionConfig(),
    )

    assert [(doc.id, score) for doc, score in fused] == [(fused[0][0].id, fused[0][1])]
    assert fused[0][0].id == "b"


@pytest.mark.parametrize(
    "kwargs",
    [{"dense_weight": -0.1}, {"sparse_weight": float("inf")}, {"rrf_k": 0}],
)
def test_rrf_fusion_rejects_invalid_weights(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        HybridFusionConfig(**kwargs)
