"""Small, explicitly sanitized evidence records for real-database tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

_CONNECTION_URI = re.compile(r"(?i)\b[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_KEY_VALUE_SECRET = re.compile(
    r"(?i)\b(?:dsn|password|passwd|pwd|host|hostname|user|username|port|"
    r"database|dbname|sslkey|sslcert|sslrootcert)"
    r"\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s,;]+)"
)
_USERINFO_HOST = re.compile(
    r"(?i)\b[^\s:/@]+(?::[^\s/@]*)?@[A-Za-z0-9_.-]+"
    r"(?::\d+)?(?:/[A-Za-z0-9_.-]+)?"
)
_LIBPQ_ENDPOINT = re.compile(
    r'(?i)\bconnection to server at\s+"[^"]+"\s*,\s*port\s+\d+'
)
_LIBPQ_ENDPOINT_LINE = re.compile(
    r"(?im)^.*(?:connection to server at|could not translate host name).*$"
)
_LIBPQ_IDENTITY = re.compile(r'(?i)\b(?:user|database)\s+"[^"]+"')
_SAFE_FRAGMENT = re.compile(r"[^A-Za-z0-9_.:() /-]+")


def sanitize_capabilities(values: Mapping[str, object]) -> dict[str, object]:
    """Retain only non-connection capability facts."""

    allowed = {
        "access_methods",
        "opclasses",
        "bm25",
        "ugin",
        "server_version_class",
        "default_transaction_read_only",
        "backend_termination_privilege",
    }
    return {key: values[key] for key in sorted(values.keys() & allowed)}


def sanitize_exception(error: BaseException) -> dict[str, str | None]:
    """Return exception class, SQLSTATE and a bounded secret-free fragment."""

    sqlstate = getattr(error, "sqlstate", None) or getattr(error, "pgcode", None)
    fragment = _LIBPQ_ENDPOINT_LINE.sub(
        "[connection-endpoint-redacted]",
        str(error),
    )
    fragment = _CONNECTION_URI.sub("[connection-uri-redacted]", fragment)
    fragment = _KEY_VALUE_SECRET.sub("[connection-field-redacted]", fragment)
    fragment = _USERINFO_HOST.sub("[connection-endpoint-redacted]", fragment)
    fragment = _LIBPQ_ENDPOINT.sub("[connection-endpoint-redacted]", fragment)
    fragment = _LIBPQ_IDENTITY.sub("[connection-identity-redacted]", fragment)
    fragment = _SAFE_FRAGMENT.sub("?", fragment).strip()[:160]
    return {
        "exception_class": type(error).__name__,
        "sqlstate": str(sqlstate) if sqlstate is not None else None,
        "fragment": fragment,
    }


def timing_bucket(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    if seconds < 0.1:
        return "under_100ms"
    if seconds < 1:
        return "100ms_to_1s"
    if seconds < 10:
        return "1s_to_10s"
    return "10s_or_more"


@dataclass(frozen=True)
class CleanupEvidence:
    owner_test: str
    kind: str
    name: str
    dropped: bool
    absent_after_cleanup: bool
    error_class: str | None = None
    sqlstate: str | None = None
