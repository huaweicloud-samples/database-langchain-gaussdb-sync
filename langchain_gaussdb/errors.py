from __future__ import annotations

import re
from typing import Any, Mapping, TypeVar, overload
from urllib.parse import unquote

_SENSITIVE_KEY_NAMES = (
    "password",
    "pass",
    "pwd",
    "sslpassword",
    "passfile",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "secret",
    "secret_key",
    "client_secret",
)
_SENSITIVE_KEYS = set(_SENSITIVE_KEY_NAMES)
_SENSITIVE_KEY_PATTERN = "|".join(re.escape(key) for key in _SENSITIVE_KEY_NAMES)
_KEYWORD_SECRET_START_RE = re.compile(
    rf"(?i)(?<![\w?&-])({_SENSITIVE_KEY_PATTERN})\s*=\s*"
)
_URL_QUERY_PARAMETER_START_RE = re.compile(r"([?&])([^=&#\s]+)=")
_URL_SECRET_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/\s]+:)([^\s]*)(@[^@\s]+)")


class GaussDBError(Exception):
    """Base error for database-langchain-gaussdb-sync failures."""

    def __init__(
        self,
        message: str,
        *,
        sqlstate: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate
        if cause is not None:
            self.__cause__ = cause


class GaussDBConnectionError(GaussDBError):
    """Raised when connecting to GaussDB fails."""


class GaussDBSQLError(GaussDBError):
    """Raised when GaussDB SQL execution fails."""


class GaussDBSQLBuildError(GaussDBError):
    """Raised when SQL generation fails."""


class GaussDBTransactionError(GaussDBError):
    """Raised when a transaction operation fails."""


class GaussDBCapabilityError(GaussDBError):
    """Raised when a requested GaussDB capability is unavailable."""


class GaussDBFilterError(GaussDBError):
    """Raised when a metadata filter is invalid or cannot be applied."""


ErrorT = TypeVar("ErrorT", bound=GaussDBError)


def sanitize_dsn(dsn: str) -> str:
    """Mask credentials in keyword and URL-style DSNs."""
    sanitized = _sanitize_url_query_secrets(dsn)
    sanitized = _sanitize_keyword_secrets(sanitized)
    return _URL_SECRET_RE.sub(
        lambda match: f"{match.group(1)}***{match.group(3)}", sanitized
    )


@overload
def wrap_psycopg_error(
    exc: BaseException,
    *,
    operation: str,
    context: Mapping[str, Any] | None = None,
    error_cls: type[ErrorT],
) -> ErrorT: ...


@overload
def wrap_psycopg_error(
    exc: BaseException,
    *,
    operation: str,
    context: Mapping[str, Any] | None = None,
    error_cls: None = None,
) -> GaussDBSQLError: ...


def wrap_psycopg_error(
    exc: BaseException,
    *,
    operation: str,
    context: Mapping[str, Any] | None = None,
    error_cls: type[GaussDBError] | None = None,
) -> GaussDBError:
    sqlstate = getattr(exc, "pgcode", None)
    sensitive_values = _sensitive_context_values(context) if context else set()
    sanitized_operation = sanitize_dsn(operation)
    if sensitive_values:
        sanitized_operation = _redact_values(sanitized_operation, sensitive_values)
    message_parts = [f"{sanitized_operation} failed"]

    if context:
        formatted_context = ", ".join(
            f"{key}={_sanitize_context_value(key, value, sensitive_values)}"
            for key, value in context.items()
        )
        message_parts.append(f"context: {formatted_context}")

    cause = sanitize_dsn(str(exc))
    if sensitive_values:
        cause = _redact_values(cause, sensitive_values)
    message_parts.append(f"cause: {cause}")
    resolved_error_cls = error_cls or GaussDBSQLError
    return resolved_error_cls("; ".join(message_parts), sqlstate=sqlstate)


def _sanitize_context_value(
    key: str,
    value: Any,
    sensitive_values: set[str] | None = None,
) -> str:
    if key.lower() in _SENSITIVE_KEYS:
        return "***"
    if isinstance(value, str):
        sanitized = sanitize_dsn(value)
    else:
        sanitized = sanitize_dsn(str(value))
    if sensitive_values:
        sanitized = _redact_values(sanitized, sensitive_values)
    return sanitized


def _redact_sensitive_context_values(text: str, context: Mapping[str, Any]) -> str:
    return _redact_values(text, _sensitive_context_values(context))


def _redact_values(text: str, values: set[str]) -> str:
    redacted = text
    for value in sorted(values, key=len, reverse=True):
        redacted = redacted.replace(value, "***")
    return redacted


def _sensitive_context_values(context: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key, value in context.items():
        normalized_key = key.lower()
        if normalized_key in _SENSITIVE_KEYS:
            sensitive_value = str(value)
            if sensitive_value:
                values.add(sensitive_value)
        elif normalized_key == "dsn" and isinstance(value, str):
            values.update(_sensitive_values_from_dsn(value))
    return values


def _sensitive_values_from_dsn(dsn: str) -> set[str]:
    values: set[str] = set()
    search_at = 0
    while match := _URL_QUERY_PARAMETER_START_RE.search(dsn, search_at):
        value_end, value = _scan_url_query_value(dsn, match.end())
        if unquote(match.group(2)).lower() in _SENSITIVE_KEYS:
            _add_raw_and_decoded_value(values, value)
        search_at = max(value_end, match.end())

    search_at = 0
    while match := _KEYWORD_SECRET_START_RE.search(dsn, search_at):
        value_end, value = _scan_keyword_value(dsn, match.end())
        if value:
            values.add(value)
        search_at = max(value_end, match.end())

    for match in _URL_SECRET_RE.finditer(dsn):
        _add_raw_and_decoded_value(values, match.group(2))
    return values


def _sanitize_url_query_secrets(text: str) -> str:
    parts: list[str] = []
    copy_from = 0
    search_at = 0
    while match := _URL_QUERY_PARAMETER_START_RE.search(text, search_at):
        value_end, _ = _scan_url_query_value(text, match.end())
        if unquote(match.group(2)).lower() in _SENSITIVE_KEYS:
            parts.append(text[copy_from : match.end()])
            parts.append("***")
            copy_from = value_end
        search_at = max(value_end, match.end())
    parts.append(text[copy_from:])
    return "".join(parts)


def _sanitize_keyword_secrets(text: str) -> str:
    parts: list[str] = []
    copy_from = 0
    search_at = 0
    while match := _KEYWORD_SECRET_START_RE.search(text, search_at):
        value_end, _ = _scan_keyword_value(text, match.end())
        parts.append(text[copy_from : match.start()])
        parts.append(f"{match.group(1)}=***")
        copy_from = value_end
        search_at = max(value_end, match.end())
    parts.append(text[copy_from:])
    return "".join(parts)


def _scan_keyword_value(text: str, start: int) -> tuple[int, str]:
    if start >= len(text):
        return start, ""

    quote = text[start] if text[start] in {"'", '"'} else None
    cursor = start + 1 if quote is not None else start
    unescaped: list[str] = []

    while cursor < len(text):
        character = text[cursor]
        if character == "\\":
            if cursor + 1 < len(text):
                unescaped.append(text[cursor + 1])
                cursor += 2
                continue
            unescaped.append(character)
            cursor += 1
            continue
        if quote is not None:
            if character == quote:
                return cursor + 1, "".join(unescaped)
        elif character.isspace():
            break
        unescaped.append(character)
        cursor += 1

    return cursor, "".join(unescaped)


def _scan_url_query_value(text: str, start: int) -> tuple[int, str]:
    cursor = start
    while cursor < len(text):
        character = text[cursor]
        if character.isspace() or character in {"&", "#"}:
            break
        cursor += 1
    return cursor, text[start:cursor]


def _add_raw_and_decoded_value(values: set[str], value: str) -> None:
    if not value:
        return
    values.add(value)
    decoded_value = unquote(value)
    if decoded_value:
        values.add(decoded_value)
