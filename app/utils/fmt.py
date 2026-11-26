from __future__ import annotations

import math
from typing import Any, Optional


def fmt(value: Any, digits: int = 2, missing: str = "–", *,
        thousands: bool = False, compact: Optional[str] = None) -> str:
    if value is None:
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return missing
    if compact == "int" and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    if compact == "g" and number.is_integer():
        return f"{number:g}"
    if thousands:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1, missing: str = "–") -> str:
    if value is None:
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return missing
    return f"{number:.{digits}f}%"


def fmt_thousands(value: Any, digits: int = 1, missing: str = "–") -> str:
    return fmt(value, digits, missing, thousands=True)


def fmt_g(value: Any, digits: int = 2, missing: str = "—") -> str:
    return fmt(value, digits, missing, compact="g")
