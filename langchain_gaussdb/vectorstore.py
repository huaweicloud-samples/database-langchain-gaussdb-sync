from __future__ import annotations

import json
import math
import uuid
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from psycopg2 import sql

from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.errors import (
    GaussDBCapabilityError,
    GaussDBConnectionError,
    GaussDBFilterError,
    GaussDBSQLBuildError,
    GaussDBSQLError,
)
from langchain_gaussdb.filters import CompiledFilter, compile_metadata_filter
from langchain_gaussdb.hybrid_search import (
    BM25Config,
    HybridFusionConfig,
    build_bm25_search_sql,
    fuse_hybrid_results,
)
from langchain_gaussdb.indexes import (
    MAX_DISTRIBUTED_EMBEDDING_DIMENSION,
    MAX_EMBEDDING_DIMENSION,
    build_create_bm25_index,
    build_create_metadata_index,
    build_create_vector_index,
)
from langchain_gaussdb.metadata_json import dumps_metadata_json
from langchain_gaussdb.metadata_sql import normalize_metadata_index_cast
from langchain_gaussdb.sql import (
    CompiledSQL,
    build_odku_insert,
    identifier,
    qualified_name,
)

_STORAGE_LATER_SR_KWARGS = frozenset({"drop_old", "filter"})
_CONSTRUCTOR_ONLY_KWARGS = frozenset({"retrieval_mode", "bm25_config"})
_REMOVED_FACTORY_KWARGS = frozenset({"create_index", "create_table"})
_DISTANCE_OPERATORS = {
    "cosine": "<+>",
    "l2": "<->",
}
_RETRIEVAL_MODES = {"dense", "bm25", "hybrid"}
_DUPLICATE_OBJECT_SQLSTATES = frozenset({"42P07", "42710", "23505"})
_WRITE_BATCH_SIZE = 1000


class GaussDBVectorStore(VectorStore):
    def __init__(
        self,
        *,
        embedding: Embeddings,
        table_name: str,
        embedding_dimension: int,
        schema_name: str | None = "public",
        engine: GaussDBEngine | None = None,
        dsn: str | None = None,
        connection_kwargs: dict[str, Any] | None = None,
        id_column: str = "id",
        content_column: str = "content",
        metadata_column: str = "metadata",
        embedding_column: str = "embedding",
        distance_strategy: str = "cosine",
        retrieval_mode: str = "dense",
        bm25_config: BM25Config | None = None,
        metadata_indexes: Mapping[str, str | None] | Sequence[str] | None = None,
    ) -> None:
        if (
            not isinstance(embedding_dimension, int)
            or isinstance(embedding_dimension, bool)
            or embedding_dimension <= 0
        ):
            raise ValueError("embedding_dimension must be a positive integer")
        if embedding_dimension > MAX_EMBEDDING_DIMENSION:
            raise ValueError(
                "embedding_dimension must be less than or equal to "
                f"{MAX_EMBEDDING_DIMENSION}"
            )
        if (
            not isinstance(distance_strategy, str)
            or distance_strategy not in _DISTANCE_OPERATORS
        ):
            allowed = ", ".join(sorted(_DISTANCE_OPERATORS))
            raise ValueError(f"distance_strategy must be one of: {allowed}")
        if not isinstance(retrieval_mode, str):
            raise ValueError("retrieval_mode must be one of: bm25, dense, hybrid")
        normalized_retrieval_mode = retrieval_mode.lower()
        if normalized_retrieval_mode not in _RETRIEVAL_MODES:
            allowed = ", ".join(sorted(_RETRIEVAL_MODES))
            raise ValueError(f"retrieval_mode must be one of: {allowed}")

        identifier(table_name, label="table_name")
        if schema_name is None:
            schema_name = "public"
        identifier(schema_name, label="schema_name")
        identifier(id_column, label="id_column")
        identifier(content_column, label="content_column")
        identifier(metadata_column, label="metadata_column")
        identifier(embedding_column, label="embedding_column")
        core_columns = (
            id_column,
            content_column,
            metadata_column,
            embedding_column,
        )
        if len(set(core_columns)) != len(core_columns):
            raise ValueError(
                "id, content, metadata, and embedding core columns must be distinct"
            )
        resolved_bm25_config = bm25_config or BM25Config(column=content_column)
        identifier(
            resolved_bm25_config.column,
            label="bm25_config.column",
        )
        if resolved_bm25_config.column in {metadata_column, embedding_column}:
            raise GaussDBSQLBuildError(
                "bm25_config.column must reference a text column"
            )
        resolved_metadata_indexes = _normalize_metadata_indexes(metadata_indexes)

        connection_sources = [
            dsn is not None,
            connection_kwargs is not None,
        ]
        if engine is not None and any(connection_sources):
            raise GaussDBConnectionError(
                "engine is mutually exclusive with other connection sources"
            )
        if engine is None and sum(connection_sources) != 1:
            raise GaussDBConnectionError(
                "GaussDBVectorStore requires exactly one connection source"
            )

        self._embedding = embedding
        self._table_name = table_name
        self._schema_name = schema_name
        self._embedding_dimension = embedding_dimension
        self._id_column = id_column
        self._content_column = content_column
        self._metadata_column = metadata_column
        self._embedding_column = embedding_column
        self._distance_strategy = distance_strategy
        self._distance_operator = _DISTANCE_OPERATORS[distance_strategy]
        self._retrieval_mode = normalized_retrieval_mode
        self._bm25_config = resolved_bm25_config
        self._metadata_indexes = resolved_metadata_indexes
        self._hybrid_fusion_config = HybridFusionConfig()
        self._initialization_lock = Lock()
        self._initialized = False
        self._distributed_deployment: bool | None = None
        self._owns_engine = engine is None
        self._engine = engine or GaussDBEngine(
            dsn=dsn,
            connection_kwargs=connection_kwargs,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def retrieval_mode(self) -> str:
        return self._retrieval_mode

    @property
    def bm25_config(self) -> BM25Config:
        return self._bm25_config

    @property
    def metadata_indexes(self) -> dict[str, str | None]:
        return dict(self._metadata_indexes)

    def close(self) -> None:
        if self._owns_engine:
            self._engine.close()

    def _prepare_table_if_needed(self) -> None:
        existing_columns = self._fetch_table_columns()
        if not existing_columns:
            try:
                self._engine.execute(
                    self._build_create_table_sql(),
                    operation="create vectorstore table",
                )
            except GaussDBSQLError as exc:
                if exc.sqlstate not in _DUPLICATE_OBJECT_SQLSTATES:
                    raise
            existing_columns = self._fetch_table_columns()
        self._check_required_columns(existing_columns)

    def setup(self) -> None:
        self._ensure_initialized()

    def _create_retrieval_indexes(self) -> None:
        if self._retrieval_mode in {"dense", "hybrid"}:
            self._create_vector_index()
        if self._retrieval_mode in {"bm25", "hybrid"}:
            self._create_bm25_index()

    def _is_distributed_deployment(self) -> bool:
        if self._distributed_deployment is not None:
            return self._distributed_deployment
        rows = self._engine.fetch_all(
            CompiledSQL(
                sql.SQL(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_catalog.pgxc_node WHERE node_type = 'D'"
                    ")"
                )
            ),
            operation="detect GaussDB deployment topology",
        )
        if (
            len(rows) != 1
            or not isinstance(rows[0], Sequence)
            or len(rows[0]) != 1
            or not isinstance(rows[0][0], bool)
        ):
            raise GaussDBCapabilityError(
                "Could not determine whether GaussDB is centralized or distributed"
            )
        self._distributed_deployment = rows[0][0]
        return self._distributed_deployment

    def _validate_deployment_contract(self) -> None:
        requires_topology = (
            self._retrieval_mode != "dense"
            or self._embedding_dimension > MAX_DISTRIBUTED_EMBEDDING_DIMENSION
        )
        if not requires_topology or not self._is_distributed_deployment():
            return
        if self._embedding_dimension > MAX_DISTRIBUTED_EMBEDDING_DIMENSION:
            raise GaussDBCapabilityError(
                "Distributed GaussDB supports embedding_dimension up to "
                f"{MAX_DISTRIBUTED_EMBEDDING_DIMENSION}"
            )
        raise GaussDBCapabilityError(
            f"retrieval_mode '{self._retrieval_mode}' is not supported on "
            "distributed GaussDB; use retrieval_mode='dense'"
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            self._validate_deployment_contract()
            self._prepare_table_if_needed()
            for key, cast in list(self._metadata_indexes.items()):
                self._create_metadata_index(key, cast=cast)
            self._create_retrieval_indexes()
            self._initialized = True

    def _build_create_table_sql(self) -> CompiledSQL:
        optional_columns = self._build_optional_table_columns_sql()
        statement = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "{} text PRIMARY KEY, "
            "{} text NOT NULL, "
            "{} JSONB NOT NULL DEFAULT {}, "
            "{}"
            "{} floatvector({}) NOT NULL"
            ") WITH (storage_type=ustore)"
        ).format(
            qualified_name(self._schema_name, self._table_name),
            identifier(self._id_column, label="id_column"),
            identifier(self._content_column, label="content_column"),
            identifier(self._metadata_column, label="metadata_column"),
            sql.SQL("'{}'::jsonb"),
            optional_columns,
            identifier(self._embedding_column, label="embedding_column"),
            sql.SQL(str(self._embedding_dimension)),
        )
        return CompiledSQL(statement)

    def _build_optional_table_columns_sql(self) -> sql.Composable:
        columns: list[sql.Composable] = []
        if self._uses_lexical_projection:
            columns.append(
                sql.SQL("{} text NULL, ").format(
                    identifier(self._bm25_config.column, label="bm25_config.column")
                )
            )
        return sql.SQL("").join(columns)

    def _fetch_table_columns(self) -> set[str]:
        rows = self._engine.fetch_all(
            self._build_required_columns_sql(),
            operation="check vectorstore required columns",
        )
        return {
            row[0]
            for row in rows
            if isinstance(row, Sequence)
            and len(row) == 1
            and isinstance(row[0], str)
            and row[0]
        }

    def _check_required_columns(self, existing_columns: set[str]) -> None:
        required_columns = {
            self._id_column,
            self._content_column,
            self._metadata_column,
            self._embedding_column,
        }
        if self._uses_lexical_projection:
            required_columns.add(self._bm25_config.column)
        missing_columns = sorted(required_columns.difference(existing_columns))
        if missing_columns:
            raise GaussDBCapabilityError(
                "VectorStore table is missing required columns: "
                + ", ".join(missing_columns)
            )

    def _build_required_columns_sql(self) -> CompiledSQL:
        return CompiledSQL(
            sql.SQL(
                "SELECT cols.column_name "
                "FROM information_schema.columns AS cols "
                "WHERE cols.table_name = %s "
                "AND cols.table_schema = %s"
            ),
            (self._table_name, self._schema_name),
        )

    def _create_vector_index(self) -> None:
        _index_name, compiled = build_create_vector_index(
            self._schema_name,
            self._table_name,
            self._embedding_column,
            self._embedding_dimension,
            distance_strategy=self._distance_strategy,
        )
        self._engine.execute(
            compiled,
            operation="create gsdiskann vector index",
        )

    def _create_metadata_index(
        self,
        key: str,
        cast: str | None = None,
    ) -> None:
        _index_name, compiled = build_create_metadata_index(
            self._schema_name,
            self._table_name,
            self._metadata_column,
            key,
            cast=cast,
        )
        self._engine.execute(compiled, operation="create metadata index")

    def _create_bm25_index(self) -> None:
        _index_name, compiled = build_create_bm25_index(
            self._schema_name,
            self._table_name,
            self._bm25_config.column,
        )
        self._engine.execute(compiled, operation="create bm25 index")

    @property
    def _uses_lexical_projection(self) -> bool:
        return self._bm25_config.column not in {
            self._content_column,
            self._id_column,
            self._metadata_column,
            self._embedding_column,
        }

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        return self._add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    def _add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        (
            text_values,
            resolved_ids,
            metadata_json_values,
            lexical_values,
        ) = self._prepare_add_texts_inputs(
            texts,
            metadatas=metadatas,
            ids=ids,
            kwargs=kwargs,
        )
        if not text_values:
            return []

        embeddings = self._embedding.embed_documents(text_values)
        if len(embeddings) != len(text_values):
            raise ValueError("embedding count must match texts length")
        vector_strings = [
            _embedding_to_vector_string(
                embedding,
                expected_dimension=self._embedding_dimension,
                index=index,
            )
            for index, embedding in enumerate(embeddings)
        ]
        self._ensure_initialized()
        for start in range(0, len(text_values), _WRITE_BATCH_SIZE):
            end = start + _WRITE_BATCH_SIZE
            compiled = self._build_add_texts_sql(
                text_values[start:end],
                resolved_ids[start:end],
                metadata_json_values[start:end],
                lexical_values[start:end],
                vector_strings[start:end],
            )
            self._engine.execute(compiled, operation="add vectorstore texts")
        return resolved_ids

    def _prepare_add_texts_inputs(
        self,
        texts: Iterable[str],
        *,
        metadatas: list[dict] | None,
        ids: list[str] | None,
        kwargs: dict[str, Any],
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        list[str],
    ]:
        text_lemmatized_values = kwargs.pop("text_lemmatized_values", None)
        _reject_constructor_only_kwargs(kwargs, context="write")
        _reject_later_sr_kwargs(kwargs)
        _reject_unknown_kwargs(kwargs)
        _reject_string_like_sequence(texts, "texts")
        text_values = list(texts)
        resolved_ids = _resolve_ids(ids, len(text_values))
        metadata_values = _coerce_metadatas(metadatas, len(text_values))
        lexical_values = self._coerce_lexical_values(
            text_lemmatized_values,
            len(text_values),
        )
        metadata_json_values = [
            _metadata_to_json(metadata, index)
            for index, metadata in enumerate(metadata_values)
        ]
        return (
            text_values,
            resolved_ids,
            metadata_json_values,
            lexical_values,
        )

    def _build_add_texts_sql(
        self,
        text_values: list[str],
        resolved_ids: list[str],
        metadata_json_values: list[str],
        lexical_values: list[str],
        vector_strings: list[str],
    ) -> CompiledSQL:
        if len(vector_strings) != len(text_values):
            raise ValueError("vector string count must match texts length")
        insert_columns = [
            self._id_column,
            self._content_column,
            self._metadata_column,
            *([self._bm25_config.column] if self._uses_lexical_projection else []),
            self._embedding_column,
        ]
        update_columns = [
            self._content_column,
            self._metadata_column,
            *([self._bm25_config.column] if self._uses_lexical_projection else []),
            self._embedding_column,
        ]
        rows = [
            (
                resolved_ids[index],
                text_values[index],
                metadata_json_values[index],
                *([lexical_values[index]] if self._uses_lexical_projection else []),
                vector_strings[index],
            )
            for index in range(len(text_values))
        ]
        return build_odku_insert(
            schema=self._schema_name,
            table=self._table_name,
            insert_columns=insert_columns,
            update_columns=update_columns,
            rows=rows,
        )

    def _coerce_lexical_values(
        self,
        values: Sequence[str] | None,
        text_count: int,
    ) -> list[str]:
        if not self._uses_lexical_projection:
            if values is not None:
                raise ValueError(
                    "text_lemmatized_values requires BM25Config(column=...) to "
                    "target a non-content projection column"
                )
            return []
        if text_count == 0 and values is None:
            return []
        if values is None:
            raise ValueError(
                "text_lemmatized_values is required when bm25_config.column targets "
                f"{self._bm25_config.column}"
            )
        _reject_string_like_sequence(values, "text_lemmatized_values")
        result = list(values)
        if len(result) != text_count:
            raise ValueError("text_lemmatized_values length must match texts length")
        for index, value in enumerate(result):
            if not isinstance(value, str):
                raise ValueError(f"text_lemmatized_values[{index}] must be a string")
        return result

    def add_documents(
        self,
        documents: Iterable[Document],
        **kwargs: Any,
    ) -> list[str]:
        texts, metadatas, ids, lexical_values = self._prepare_add_documents_inputs(
            documents,
            kwargs,
        )
        return self._add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            text_lemmatized_values=lexical_values,
        )

    def _prepare_add_documents_inputs(
        self,
        documents: Iterable[Document],
        kwargs: dict[str, Any],
    ) -> tuple[list[str], list[dict], list[str] | None, Sequence[str] | None]:
        lexical_values = kwargs.pop("text_lemmatized_values", None)
        _reject_constructor_only_kwargs(kwargs, context="write")
        _reject_later_sr_kwargs(kwargs)
        explicit_ids = kwargs.pop("ids", None)
        _reject_unknown_kwargs(kwargs)
        texts, metadatas, ids = _documents_to_texts_metadatas_ids(
            documents,
            explicit_ids,
        )
        return texts, metadatas, ids, lexical_values

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            document
            for document, _score in self.similarity_search_with_score(
                query,
                k=k,
                **kwargs,
            )
        ]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            document
            for document, _score in self.similarity_search_with_score_by_vector(
                embedding,
                k=k,
                **kwargs,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return self._similarity_search_with_score_for_mode(
            self._retrieval_mode,
            query,
            k=k,
            **kwargs,
        )

    def _similarity_search_with_score_for_mode(
        self,
        mode: str,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        if mode not in _RETRIEVAL_MODES:
            raise ValueError("mode must be one of: bm25, dense, hybrid")

        bm25_query = query
        if mode == "dense":
            if "bm25_query" in kwargs:
                raise ValueError("bm25_query is not supported for dense search")
        else:
            candidate = kwargs.pop("bm25_query", query)
            if not isinstance(candidate, str):
                raise ValueError("bm25_query must be a string")
            bm25_query = candidate

        result_count, compiled_filter = self._prepare_search(k, kwargs)
        if result_count == 0:
            return []

        if mode == "bm25":
            return self._bm25_search_with_score(
                bm25_query,
                result_count,
                compiled_filter,
            )

        query_embedding = self._embedding.embed_query(query)
        if mode == "hybrid":
            return self._hybrid_search_with_score(
                query_embedding,
                result_count,
                compiled_filter,
                bm25_query=bm25_query,
            )

        return self._similarity_search_with_score_by_vector(
            query_embedding,
            result_count,
            compiled_filter,
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        result_count, compiled_filter = self._prepare_search(k, kwargs)
        if result_count == 0:
            return []
        return self._similarity_search_with_score_by_vector(
            embedding,
            result_count,
            compiled_filter,
        )

    def _similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int,
        compiled_filter: CompiledFilter | None,
        *,
        operation: str = "similarity search vectorstore documents with score",
    ) -> list[tuple[Document, float]]:
        vector = _embedding_to_vector_string(
            embedding,
            expected_dimension=self._embedding_dimension,
            index=0,
        )
        rows = self._fetch_all_with_filter_context(
            self._build_vector_distance_select(vector, k, compiled_filter),
            operation=operation,
            compiled_filter=compiled_filter,
        )
        return [(_document_from_row(row), float(row[3])) for row in rows]

    def _bm25_search_with_score(
        self,
        query: str,
        k: int,
        compiled_filter: CompiledFilter | None,
        *,
        operation: str = "bm25 search vectorstore documents",
    ) -> list[tuple[Document, float]]:
        rows = self._fetch_all_with_filter_context(
            build_bm25_search_sql(
                schema=self._schema_name,
                table=self._table_name,
                id_column=self._id_column,
                content_column=self._content_column,
                metadata_column=self._metadata_column,
                bm25_column=self._bm25_config.column,
                query=query,
                k=k,
                compiled_filter=compiled_filter,
            ),
            operation=operation,
            compiled_filter=compiled_filter,
        )
        return [(_document_from_row(row), float(row[3])) for row in rows]

    def _hybrid_search_with_score(
        self,
        query_embedding: list[float],
        k: int,
        compiled_filter: CompiledFilter | None,
        *,
        bm25_query: str,
    ) -> list[tuple[Document, float]]:
        fetch_k = max(k, self._hybrid_fusion_config.fetch_k)
        dense_results = self._similarity_search_with_score_by_vector(
            query_embedding,
            fetch_k,
            compiled_filter,
            operation=("hybrid dense search vectorstore documents with score"),
        )
        bm25_results = self._bm25_search_with_score(
            bm25_query,
            fetch_k,
            compiled_filter,
            operation="hybrid bm25 search vectorstore documents with score",
        )
        return fuse_hybrid_results(
            dense_results,
            bm25_results,
            k=k,
            config=self._hybrid_fusion_config,
        )

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        if self._retrieval_mode != "dense":
            raise ValueError(
                "similarity_score_threshold relevance is only supported for "
                "dense retrieval"
            )
        return super().similarity_search_with_relevance_scores(query, k=k, **kwargs)

    def _compile_metadata_filter(self, filter_value: Any) -> CompiledFilter | None:
        return compile_metadata_filter(
            filter_value,
            metadata_column=self._metadata_column,
            metadata_indexes=self._metadata_indexes,
        )

    def _prepare_search(
        self,
        k: Any,
        kwargs: dict[str, Any],
    ) -> tuple[int, CompiledFilter | None]:
        filter_value = kwargs.pop("filter", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported search kwargs: {names}")
        result_count = _validate_non_negative_int(k, "k")
        return result_count, self._compile_metadata_filter(filter_value)

    def _fetch_all_with_filter_context(
        self,
        compiled: CompiledSQL,
        *,
        operation: str,
        compiled_filter: CompiledFilter | None,
    ) -> list[tuple[Any, ...]]:
        try:
            return self._engine.fetch_all(
                compiled,
                operation=operation,
            )
        except GaussDBSQLError as exc:
            if compiled_filter is not None and _is_filter_cast_error(exc):
                fields = ", ".join(compiled_filter.fields) or "metadata"
                raise GaussDBFilterError(
                    "metadata filter failed because stored metadata values "
                    f"do not match the requested filter type; fields={fields}",
                    sqlstate=exc.sqlstate,
                ) from None
            raise

    def _build_vector_distance_select(
        self,
        vector: str,
        limit: int,
        compiled_filter: CompiledFilter | None = None,
        *,
        include_embedding_text: bool = False,
    ) -> CompiledSQL:
        projections = [
            identifier(self._id_column, label="id_column"),
            identifier(self._content_column, label="content_column"),
            identifier(self._metadata_column, label="metadata_column"),
        ]
        if include_embedding_text:
            projections.append(
                sql.SQL("{}::text").format(
                    identifier(self._embedding_column, label="embedding_column")
                )
            )
        projections.append(
            sql.SQL("{} {} %s::floatvector({}) AS distance").format(
                identifier(self._embedding_column, label="embedding_column"),
                sql.SQL(self._distance_operator),
                sql.SQL(str(self._embedding_dimension)),
            )
        )
        statement = sql.SQL(
            "SELECT {} FROM {} {}ORDER BY distance ASC LIMIT %s"
        ).format(
            sql.SQL(", ").join(projections),
            qualified_name(self._schema_name, self._table_name),
            _where_clause(compiled_filter),
        )
        return CompiledSQL(
            statement, (vector,) + _filter_params(compiled_filter) + (limit,)
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        _ensure_dense_retrieval_mode(
            self._retrieval_mode,
            "similarity_score_threshold relevance",
        )
        if self._distance_strategy == "cosine":
            return lambda distance: _clamp_score(1.0 - float(distance) / 2.0)
        return lambda distance: _clamp_score(1.0 / (1.0 + float(distance)))

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        _ensure_dense_retrieval_mode(self._retrieval_mode, "MMR")
        result_count, compiled_filter = self._prepare_search(k, kwargs)
        candidate_count = _validate_non_negative_int(fetch_k, "fetch_k")
        diversity = _validate_lambda_mult(lambda_mult)
        if result_count == 0 or candidate_count == 0:
            return []
        return self._max_marginal_relevance_search_by_vector_compiled(
            self._embedding.embed_query(query),
            result_count,
            candidate_count,
            diversity,
            compiled_filter,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        _ensure_dense_retrieval_mode(self._retrieval_mode, "MMR")
        result_count, compiled_filter = self._prepare_search(k, kwargs)
        candidate_count = _validate_non_negative_int(fetch_k, "fetch_k")
        diversity = _validate_lambda_mult(lambda_mult)
        if result_count == 0 or candidate_count == 0:
            return []
        return self._max_marginal_relevance_search_by_vector_compiled(
            embedding,
            result_count,
            candidate_count,
            diversity,
            compiled_filter,
        )

    def _max_marginal_relevance_search_by_vector_compiled(
        self,
        embedding: list[float],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        compiled_filter: CompiledFilter | None,
    ) -> list[Document]:
        vector = _embedding_to_vector_string(
            embedding,
            expected_dimension=self._embedding_dimension,
            index=0,
        )
        rows = self._fetch_all_with_filter_context(
            self._build_vector_distance_select(
                vector,
                fetch_k,
                compiled_filter,
                include_embedding_text=True,
            ),
            operation="max marginal relevance search vectorstore documents",
            compiled_filter=compiled_filter,
        )
        if not rows:
            return []
        candidate_embeddings = [
            _vector_string_to_floats(
                row[3],
                expected_dimension=self._embedding_dimension,
            )
            for row in rows
        ]
        selected_indexes = maximal_marginal_relevance(
            _to_mmr_query_array(embedding),
            candidate_embeddings,
            lambda_mult=lambda_mult,
            k=min(k, len(rows)),
        )
        return [_document_from_row(rows[index]) for index in selected_indexes]

    def delete(
        self,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> bool:
        prepared = self._prepare_delete(ids, kwargs)
        if prepared is None:
            return True
        compiled, operation = prepared
        self._engine.execute(compiled, operation=operation)
        return True

    def _prepare_delete(
        self,
        ids: list[str] | None,
        kwargs: dict[str, Any],
    ) -> tuple[CompiledSQL, str] | None:
        delete_all = kwargs.pop("delete_all", False)
        _reject_later_sr_kwargs(kwargs)
        _reject_unknown_kwargs(kwargs)

        if ids is None:
            if delete_all is not True:
                raise ValueError("delete_all=True is required when ids is None")
            compiled = CompiledSQL(
                sql.SQL("DELETE FROM {}").format(
                    qualified_name(self._schema_name, self._table_name)
                )
            )
            return compiled, "delete all vectorstore documents"

        if delete_all is True:
            raise ValueError("delete_all=True cannot be used when ids are provided")

        id_values = _coerce_id_sequence(ids, "ids")
        if not id_values:
            return None

        compiled = CompiledSQL(
            sql.SQL("DELETE FROM {} WHERE {} = ANY(%s)").format(
                qualified_name(self._schema_name, self._table_name),
                identifier(self._id_column, label="id_column"),
            ),
            (id_values,),
        )
        return compiled, "delete vectorstore documents by ids"

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        compiled = self._build_get_by_ids_sql(ids)
        if compiled is None:
            return []
        rows = self._engine.fetch_all(
            compiled,
            operation="get vectorstore documents by ids",
        )
        return [_document_from_row(row) for row in rows]

    def _build_get_by_ids_sql(
        self,
        ids: Sequence[str],
    ) -> CompiledSQL | None:
        id_values = _coerce_id_sequence(ids, "ids")
        if not id_values:
            return None
        return CompiledSQL(
            sql.SQL("SELECT {}, {}, {} FROM {} WHERE {} = ANY(%s)").format(
                identifier(self._id_column, label="id_column"),
                identifier(self._content_column, label="content_column"),
                identifier(self._metadata_column, label="metadata_column"),
                qualified_name(self._schema_name, self._table_name),
                identifier(self._id_column, label="id_column"),
            ),
            (id_values,),
        )

    @classmethod
    def _from_factory(
        cls,
        embedding: Embeddings,
        writer: Callable[["GaussDBVectorStore", Any], object],
        kwargs: dict[str, Any],
    ) -> "GaussDBVectorStore":
        text_lemmatized_values = kwargs.pop("text_lemmatized_values", None)
        _reject_removed_factory_kwargs(kwargs)
        _reject_later_sr_kwargs(kwargs)
        store = cls(embedding=embedding, **kwargs)
        try:
            writer(store, text_lemmatized_values)
        except BaseException as exc:
            _close_owned_store_preserving_failure(store, exc)
            raise
        return store

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "GaussDBVectorStore":
        return cls._from_factory(
            embedding,
            lambda store, lexical_values: store.add_texts(
                texts,
                metadatas=metadatas,
                ids=ids,
                text_lemmatized_values=lexical_values,
            ),
            kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "GaussDBVectorStore":
        explicit_ids = kwargs.pop("ids", None)
        return cls._from_factory(
            embedding,
            lambda store, lexical_values: store.add_documents(
                documents,
                ids=explicit_ids,
                text_lemmatized_values=lexical_values,
            ),
            kwargs,
        )


def _close_owned_store_preserving_failure(
    store: GaussDBVectorStore,
    primary_error: BaseException,
) -> None:
    if not store._owns_engine:
        return
    try:
        store.close()
    except BaseException:
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note("Owned GaussDBEngine cleanup also failed")


def _reject_later_sr_kwargs(kwargs: dict[str, Any]) -> None:
    unsupported = sorted(_STORAGE_LATER_SR_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"{names} is not supported by this storage API")


def _reject_constructor_only_kwargs(kwargs: dict[str, Any], *, context: str) -> None:
    unsupported = sorted(_CONSTRUCTOR_ONLY_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"{names} must be configured on the constructor, not {context}"
        )


def _reject_removed_factory_kwargs(kwargs: dict[str, Any]) -> None:
    unsupported = sorted(_REMOVED_FACTORY_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"{names} is no longer supported; factories initialize automatically"
        )


def _reject_unknown_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise ValueError(f"Unsupported storage kwargs: {names}")


def _ensure_dense_retrieval_mode(retrieval_mode: str, feature: str) -> None:
    if retrieval_mode != "dense":
        raise ValueError(f"{feature} is only supported for dense retrieval")


def _where_clause(compiled_filter: CompiledFilter | None) -> sql.Composable:
    if compiled_filter is None:
        return sql.SQL("")
    return sql.SQL("WHERE {} ").format(compiled_filter.sql)


def _filter_params(compiled_filter: CompiledFilter | None) -> tuple[Any, ...]:
    if compiled_filter is None:
        return ()
    return compiled_filter.params


def _is_filter_cast_error(exc: GaussDBSQLError) -> bool:
    return bool(exc.sqlstate and exc.sqlstate.startswith("22"))


def _validate_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_lambda_mult(value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("lambda_mult must be a finite number between 0 and 1")
    return float(value)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _reject_string_like_sequence(value: Any, name: str) -> None:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a string")


def _coerce_id_sequence(ids: Sequence[str], name: str) -> list[str]:
    _reject_string_like_sequence(ids, name)
    id_values = list(ids)
    for index, value in enumerate(id_values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")
    return id_values


def _normalize_metadata_indexes(
    metadata_indexes: Mapping[str, str | None] | Sequence[str] | None,
) -> dict[str, str | None]:
    if metadata_indexes is None:
        return {}
    if isinstance(metadata_indexes, Mapping):
        items = list(metadata_indexes.items())
    elif isinstance(metadata_indexes, (list, tuple)):
        items = [(key, None) for key in metadata_indexes]
    else:
        raise ValueError("metadata_indexes must be a mapping or a sequence")

    normalized: dict[str, str | None] = {}
    for key, cast in items:
        if not isinstance(key, str) or not key:
            raise ValueError("metadata index key must be a non-empty string")
        if key.startswith("$"):
            raise ValueError("metadata index key must not start with '$'")
        if "\0" in key:
            raise ValueError("metadata index key must not contain NUL")
        normalized[key] = normalize_metadata_index_cast(cast)
    return normalized


def _resolve_ids(ids: list[str] | None, text_count: int) -> list[str]:
    if ids is None:
        return [str(uuid.uuid4()) for _ in range(text_count)]

    _reject_string_like_sequence(ids, "ids")
    resolved_ids = list(ids)
    if len(resolved_ids) != text_count:
        raise ValueError("ids length must match texts length")

    for index, value in enumerate(resolved_ids):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ids[{index}] must not be empty")
    return resolved_ids


def _coerce_metadatas(
    metadatas: list[dict] | None,
    text_count: int,
) -> list[dict]:
    if metadatas is None:
        return [{} for _ in range(text_count)]

    metadata_values = list(metadatas)
    if len(metadata_values) != text_count:
        raise ValueError("metadatas length must match texts length")
    for index, metadata in enumerate(metadata_values):
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata at index {index} must be a dict")
    return metadata_values


def _metadata_to_json(metadata: dict, index: int) -> str:
    try:
        return dumps_metadata_json(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata at index {index} is not JSON serializable "
            f"(type={type(metadata).__name__})"
        ) from exc


def _metadata_from_db(metadata: Any) -> dict:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str):
        try:
            value = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "metadata from database must be a valid JSON object"
            ) from exc
        if isinstance(value, dict):
            return value
    raise ValueError("metadata from database must be a JSON object")


def _documents_to_texts_metadatas_ids(
    documents: Iterable[Document],
    explicit_ids: list[str] | None,
) -> tuple[list[str], list[dict], list[str]]:
    document_values = list(documents)
    return (
        [document.page_content for document in document_values],
        [document.metadata or {} for document in document_values],
        explicit_ids
        if explicit_ids is not None
        else [
            document.id if document.id is not None else str(uuid.uuid4())
            for document in document_values
        ],
    )


def _document_from_row(row: Sequence[Any]) -> Document:
    return Document(
        id=str(row[0]),
        page_content=str(row[1]),
        metadata=_metadata_from_db(row[2]),
    )


def _vector_string_to_floats(value: Any, *, expected_dimension: int) -> list[float]:
    if not isinstance(value, str):
        raise ValueError("embedding from database must be a vector string")

    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("embedding from database must be a vector string")

    raw_values = [] if text == "[]" else text[1:-1].split(",")
    values: list[float] = []
    for raw_value in raw_values:
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "embedding from database must contain finite numeric values"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                "embedding from database must contain finite numeric values"
            )
        values.append(number)

    if len(values) != expected_dimension:
        raise ValueError("embedding from database has incompatible dimension")
    return values


def _to_mmr_query_array(query_embedding: list[float]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "maximal_marginal_relevance requires numpy to be installed"
        ) from exc
    return np.array(query_embedding, dtype=float)


def _embedding_to_vector_string(
    embedding: Iterable[float],
    *,
    expected_dimension: int,
    index: int,
) -> str:
    values = list(embedding)
    if len(values) != expected_dimension:
        raise ValueError(
            f"embedding dimension mismatch at index {index}: "
            f"expected {expected_dimension}, got {len(values)}"
        )

    serialized_values = []
    for value_index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"embedding value at index {index}.{value_index} must be numeric"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                f"embedding value at index {index}.{value_index} must be finite"
            )
        serialized_values.append(str(number))
    return "[" + ",".join(serialized_values) + "]"
