from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() == ""


def has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def text_value(value: Any) -> Optional[str]:
    if is_blank(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_student_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>"):
        return None
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def parse_enum_value(enum_class: type, value: Any):
    if is_blank(value):
        return None
    if isinstance(value, enum_class):
        return value
    text = str(value).strip()
    for member in enum_class:
        if member.value == text:
            return member
    lowered = text.lower()
    for member in enum_class:
        if member.value.lower() == lowered or member.name.lower() == lowered:
            return member
    return None


def parse_int(value: Any) -> Optional[int]:
    if is_blank(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number or not number.is_integer():
        return None
    return int(number)


def parse_number(value: Any, *, decimal_comma: bool = True) -> Optional[float]:
    if decimal_comma:
        if not has_value(value):
            return None
        text = str(value).strip().replace(",", ".")
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
    return None if number != number else number


def parse_list_field(value: Any) -> List[str]:
    if is_blank(value):
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = [part.strip() for part in str(value).split("|")]
    return [item for item in items if item and item.lower() != "nan"]


def split_ids(value: Any) -> List[str]:
    if not has_value(value):
        return []
    ids = [normalize_student_id(part) for part in str(value).split("|")]
    return [i for i in ids if i]


def _truncate(values: List[str], max_selections: Optional[int]) -> List[str]:
    if max_selections is not None and len(values) > max_selections:
        return values[:max_selections]
    return values


def parse_multi_select_field(value: Any, max_selections: Optional[int] = None) -> List[str]:
    return _truncate(parse_list_field(value), max_selections)


def parse_multi_select(value: Any, max_selections: Optional[int] = None) -> List[str]:
    if not has_value(value):
        return []
    values = [v.strip() for v in str(value).split("|") if v.strip()]
    return _truncate(values, max_selections)


def parse_categorical_field(value: Any,
                            valid_categories: Optional[List[str]] = None) -> Optional[str]:
    if is_blank(value):
        return None
    str_value = str(value).strip()
    for category in valid_categories or []:
        if str(category).lower() == str_value.lower():
            return category
    return str_value


def parse_ordinal_field(value: Any, valid_levels: Optional[List[str]] = None) -> Optional[str]:
    if is_blank(value):
        return None
    text = str(value).strip()
    for level in valid_levels or []:
        if str(level).lower() == text.lower():
            return level
    return None


def parse_ordinal_weight(value: Any, levels: List[Dict[str, Any]]) -> Optional[float]:
    if is_blank(value):
        return None
    text = str(value).strip()
    for level in levels:
        if str(level.get("name", "")).strip().lower() == text.lower():
            return level.get("weight")
    try:
        number = float(text)
    except ValueError:
        return None
    for level in levels:
        try:
            if float(level.get("weight")) == number:
                return level.get("weight")
        except (TypeError, ValueError):
            continue
    return None
