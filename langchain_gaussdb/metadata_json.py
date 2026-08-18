from __future__ import annotations

import datetime as dt
import json
from typing import Any


def dumps_metadata_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
