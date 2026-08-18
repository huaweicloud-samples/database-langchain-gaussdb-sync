from __future__ import annotations

from psycopg2 import sql

DATE_FORMAT = "YYYY-MM-DD"
METADATA_INDEX_CASTS = frozenset({"text", "float", "bigint", "boolean", "date"})
_METADATA_INDEX_CAST_ALIASES = {
    "integer": "bigint",
    "double precision": "float",
}


def normalize_metadata_index_cast(cast: object) -> str | None:
    """Normalize the metadata index cast used by DDL and filter selectors."""
    if cast is None:
        return None
    if not isinstance(cast, str):
        raise ValueError("metadata index cast must be a string or None")
    normalized = cast.strip().lower()
    normalized = _METADATA_INDEX_CAST_ALIASES.get(normalized, normalized)
    if normalized not in METADATA_INDEX_CASTS:
        allowed = ", ".join(sorted(METADATA_INDEX_CASTS))
        raise ValueError(f"metadata index cast must be one of: {allowed}")
    return normalized


def metadata_literal_text_selector(
    metadata_column: sql.Composable,
    key: str,
) -> sql.Composable:
    return sql.SQL("{}->>{}").format(metadata_column, sql.Literal(key))


def metadata_literal_json_selector(
    metadata_column: sql.Composable,
    key: str,
) -> sql.Composable:
    return sql.SQL("{}->{}").format(metadata_column, sql.Literal(key))


def metadata_literal_json_text_selector(
    metadata_column: sql.Composable,
    key: str,
) -> sql.Composable:
    return sql.SQL("({}->{})::text").format(
        metadata_column,
        sql.Literal(key),
    )


def metadata_literal_index_selector(
    metadata_column: sql.Composable,
    key: str,
    cast: str | None,
) -> sql.Composable:
    """Return the canonical JSONB expression used by metadata indexes/filters."""
    if cast in {None, "text"}:
        return metadata_literal_json_text_selector(metadata_column, key)

    text_selector = metadata_literal_text_selector(metadata_column, key)
    if cast == "date":
        # ISO YYYY-MM-DD text sorts chronologically and stays IMMUTABLE/indexable.
        return text_selector
    return sql.SQL("({})::{}").format(text_selector, sql.SQL(cast))
