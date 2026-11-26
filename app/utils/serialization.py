from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, time
from enum import Enum
from pathlib import PurePath
from typing import Any

__all__ = ["json_safe"]


def _key(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(str(_scalar(k)) for k in key)
    value = _scalar(key)
    return value if isinstance(value, str) else str(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return _scalar(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    return value


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, str):
        return str(value)
    if type(value).__name__ in ("NAType", "NaTType"):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {_key(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [json_safe(v) for v in value]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=lambda v: str(v))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe({f.name: getattr(value, f.name) for f in dataclasses.fields(value)})
    scalar = _scalar(value)
    if scalar is not value:
        return json_safe(scalar)
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and getattr(value, "ndim", 0):
        return json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)
