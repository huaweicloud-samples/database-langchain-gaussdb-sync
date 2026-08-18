from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from psycopg2 import sql

from langchain_gaussdb.filters import CompiledFilter
from langchain_gaussdb.sql import CompiledSQL, identifier, qualified_name


@dataclass(frozen=True)
class BM25Config:
    column: str = "content"


@dataclass(frozen=True)
class HybridFusionConfig:
    fetch_k: int = 20
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: float = 60.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fetch_k, int)
            or isinstance(self.fetch_k, bool)
            or self.fetch_k < 0
        ):
            raise ValueError("fetch_k must be a non-negative integer")
        for name in ("dense_weight", "sparse_weight", "rrf_k"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number")


def build_bm25_search_sql(
    *,
    schema: str | None,
    table: str,
    id_column: str,
    content_column: str,
    metadata_column: str,
    bm25_column: str,
    query: str,
    k: int,
    compiled_filter: CompiledFilter | None,
) -> CompiledSQL:
    where_clause = sql.SQL("")
    filter_params = ()
    if compiled_filter is not None:
        where_clause = sql.SQL("WHERE {} ").format(compiled_filter.sql)
        filter_params = compiled_filter.params

    statement = sql.SQL(
        "SELECT {}, {}, {}, {} ### %s AS bm25_score "
        "FROM {} {}ORDER BY bm25_score DESC LIMIT %s"
    ).format(
        identifier(id_column, label="id_column"),
        identifier(content_column, label="content_column"),
        identifier(metadata_column, label="metadata_column"),
        identifier(bm25_column, label="bm25_column"),
        qualified_name(schema, table),
        where_clause,
    )
    return CompiledSQL(statement, (query,) + filter_params + (k,))


def fuse_hybrid_results(
    dense_results: Sequence[tuple[Document, float]],
    bm25_results: Sequence[tuple[Document, float]],
    *,
    k: int,
    config: HybridFusionConfig,
) -> list[tuple[Document, float]]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if k == 0:
        return []

    documents: dict[str, Document] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    for rank, (document, _score) in enumerate(dense_results, start=1):
        doc_id = _document_id(document)
        documents.setdefault(doc_id, document)
        dense_ranks.setdefault(doc_id, rank)

    for rank, (document, _score) in enumerate(bm25_results, start=1):
        doc_id = _document_id(document)
        documents.setdefault(doc_id, document)
        sparse_ranks.setdefault(doc_id, rank)

    fused: list[tuple[Document, float]] = []
    for doc_id, document in documents.items():
        score = 0.0
        dense_rank = dense_ranks.get(doc_id)
        if dense_rank is not None:
            score += float(config.dense_weight) / (float(config.rrf_k) + dense_rank)
        sparse_rank = sparse_ranks.get(doc_id)
        if sparse_rank is not None:
            score += float(config.sparse_weight) / (float(config.rrf_k) + sparse_rank)
        fused.append((document, score))

    fused.sort(key=lambda item: (-item[1], _document_id(item[0])))
    return fused[:k]


def _document_id(document: Document) -> str:
    if document.id is None:
        raise ValueError("hybrid fusion requires Document.id")
    return str(document.id)
