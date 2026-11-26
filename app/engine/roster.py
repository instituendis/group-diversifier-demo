from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from app import extensions


RESERVED_COLUMNS = {
    "student_id", "identifier", "prefer_teammates", "avoid_teammates",
    "group_number", "team_id", "project_id",
}
RESERVED_PREFIXES = ("team_project_",)

KNOWN_ORDERS: Dict[str, List[str]] = {
    "skill": ["Seedling", "Sprout", "Sapling", "Oak"],
}
KNOWN_ORDERS.update(extensions.hook("known_orders", {}))

MULTI_SEPARATOR = "|"

_MISSING_STRINGS = {"", "nan", "<na>"}

NUMERIC_SHARE = 0.9
UNIQUE_SHARE = 0.9
UNIQUE_MIN_PRESENT = 8
_ID_TOKENS = {"id", "ids", "identifier", "nr", "no", "num", "number", "code", "login",
              "username", "userid", "netid", "sis", "email", "mail", "e-mail"}
_ID_TOKENS.update(extensions.hook("id_tokens", ()))
_TEXT_TOKENS = {"name", "names", "firstname", "lastname", "surname", "email", "mail",
                "comment", "comments", "note", "notes", "remark", "remarks", "description",
                "motivation", "feedback"}
_DECIMAL_COMMA = re.compile(r"^[+-]?\d+,\d+$")
_DATE_RE = re.compile(
    r"^(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"
    r"([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?$")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_cell(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    if hasattr(value, "value") and not isinstance(value, (str, int, float)):
        value = value.value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text or None


_clean = clean_cell


def _to_float(value: Any) -> Optional[float]:
    text = _clean(value)
    if text is None:
        return None
    if _DECIMAL_COMMA.match(text):
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _number_text(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def _name_tokens(name: str) -> set:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def is_reserved(column: str) -> bool:
    name = str(column).strip()
    return name in RESERVED_COLUMNS or name.startswith(RESERVED_PREFIXES)


def build_roster_frame(students: Sequence[Any], raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ordinal_keys: set = set()
    for s in students:
        row: Dict[str, Any] = {"student_id": str(s.student_id).strip()}
        typed = (s.typed_attributes() if hasattr(s, "typed_attributes") else
                 extensions.hook("attributes_of", lambda _record: {})(s))
        for name, value in typed.items():
            cell = value if isinstance(value, int) else _clean(value)
            if cell is not None:
                row[name] = cell
        for mapping in ("dynamic_categorical", "dynamic_numeric"):
            for key, value in (getattr(s, mapping, None) or {}).items():
                if not _is_missing(value):
                    row[key] = value
        for key, value in (getattr(s, "dynamic_ordinal", None) or {}).items():
            ordinal_keys.add(key)
            if key not in row and not _is_missing(value):
                row[key] = value
        for key, values in (getattr(s, "dynamic_multi_select", None) or {}).items():
            if values:
                row[key] = MULTI_SEPARATOR.join(str(v) for v in values)
        prefer = getattr(s, "prefer_teammates", None) or []
        avoid = getattr(s, "avoid_teammates", None) or []
        row["prefer_teammates"] = MULTI_SEPARATOR.join(str(p).strip() for p in prefer if str(p).strip())
        row["avoid_teammates"] = MULTI_SEPARATOR.join(str(a).strip() for a in avoid if str(a).strip())
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["student_id", "prefer_teammates", "avoid_teammates"])

    if raw_df is not None and len(raw_df) and "student_id" in raw_df.columns:
        raw = raw_df.copy()
        raw["student_id"] = raw["student_id"].map(lambda v: _clean(v) or "")
        raw = raw.drop_duplicates("student_id", keep="first").set_index("student_id")
        for column in raw.columns:
            name = str(column).strip()
            if not name or is_reserved(name):
                continue
            series = raw[column]
            if series.map(_is_missing).all():
                continue
            labels = frame["student_id"].map(
                lambda sid, col=series: None if sid not in col.index or _is_missing(col.loc[sid])
                else col.loc[sid]
            )
            if name not in frame.columns:
                frame[name] = labels
            elif name in ordinal_keys:
                frame[name] = [label if label is not None else old
                               for label, old in zip(labels, frame[name])]

    postprocess = extensions.hook("roster_frame_postprocess", None)
    if postprocess is not None:
        frame = postprocess(frame)

    frame = frame.astype(object).where(pd.notna(frame), None)
    ordered = ["student_id"] + [c for c in frame.columns if c != "student_id"]
    return frame[ordered].reset_index(drop=True)


@dataclass
class AttributeInfo:
    name: str
    kind: str
    coverage: float
    n_unique: int
    values: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)
    non_numeric: List[str] = field(default_factory=list)
    flag: str = ""

    @property
    def usable(self) -> bool:
        return self.kind in ("categorical", "numeric", "multi")


def infer_attribute(series: pd.Series, name: Optional[str] = None) -> AttributeInfo:
    name = name or str(series.name)
    n = len(series)
    cleaned = [_clean(v) for v in series.tolist()]
    present = [v for v in cleaned if v is not None]
    coverage = len(present) / n if n else 0.0
    if not present:
        return AttributeInfo(name=name, kind="empty", coverage=0.0, n_unique=0)

    is_multi = sum(MULTI_SEPARATOR in v for v in present) >= max(1, 0.2 * len(present))
    if is_multi:
        counts: Dict[str, int] = {}
        for v in present:
            for item in {p.strip() for p in v.split(MULTI_SEPARATOR) if p.strip()}:
                counts[item] = counts.get(item, 0) + 1
        values = sorted(counts, key=lambda k: (-counts[k], k))
        return AttributeInfo(name=name, kind="multi", coverage=coverage,
                             n_unique=len(values), values=values, counts=counts)

    tokens = _name_tokens(name)
    numbers = [_to_float(v) for v in present]
    parsed = [x for x in numbers if x is not None]
    if parsed and len(parsed) >= NUMERIC_SHARE * len(present):
        info = _numeric_info(name, n, present, numbers, parsed, tokens)
        if info.kind in ("numeric", "text"):
            return info

    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    values = sorted(counts, key=lambda k: (-counts[k], k))
    n_unique = len(values)
    distinct = n_unique >= UNIQUE_SHARE * len(present)
    flag = ""
    if sum(1 for v in present if _DATE_RE.match(v)) >= NUMERIC_SHARE * len(present):
        flag = "date"
    elif distinct and n_unique > 1 and (len(present) >= UNIQUE_MIN_PRESENT
                                        or tokens & (_TEXT_TOKENS | _ID_TOKENS)):
        flag = "identifier"
    if n_unique == 1:
        kind = "constant"
    elif flag == "identifier" or (n_unique > 20 and n_unique > 0.8 * len(present)):
        kind = "text"
    else:
        kind = "categorical"
    stats: Dict[str, float] = {}
    if len(parsed) == len(present):
        mean = sum(parsed) / len(parsed)
        var = sum((x - mean) ** 2 for x in parsed) / len(parsed)
        stats = {"min": min(parsed), "max": max(parsed), "mean": mean, "std": math.sqrt(var)}
    return AttributeInfo(name=name, kind=kind, coverage=coverage, n_unique=n_unique,
                         values=values, counts=counts, stats=stats, flag=flag)


def _numeric_info(name: str, n: int, present: List[str], numbers: List[Optional[float]],
                  parsed: List[float], tokens: set) -> AttributeInfo:
    bad: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for text, x in zip(present, numbers):
        if x is None:
            bad[text] = bad.get(text, 0) + 1
        else:
            key = _number_text(x)
            counts[key] = counts.get(key, 0) + 1
    values = sorted(counts, key=lambda k: (-counts[k], k))
    n_unique = len(values)
    mean = sum(parsed) / len(parsed)
    var = sum((x - mean) ** 2 for x in parsed) / len(parsed)
    stats: Dict[str, float] = {"min": min(parsed), "max": max(parsed), "mean": mean,
                               "std": math.sqrt(var)}
    if bad:
        stats["non_numeric"] = sum(bad.values())
    integers = all(float(x).is_integer() for x in parsed)
    distinct = n_unique == len(parsed)
    consecutive = integers and distinct and max(parsed) - min(parsed) == len(parsed) - 1
    flag = ""
    if (integers and distinct and len(parsed) >= UNIQUE_MIN_PRESENT
            and (tokens & _ID_TOKENS or consecutive)):
        flag = "identifier"
    if flag:
        kind = "text"
    elif n_unique > 5:
        kind = "numeric"
    else:
        kind = "categorical"
    return AttributeInfo(name=name, kind=kind, coverage=len(parsed) / n if n else 0.0,
                         n_unique=n_unique, values=values, counts=counts, stats=stats,
                         non_numeric=sorted(bad, key=lambda k: (-bad[k], k)), flag=flag)


def infer_attributes(frame: pd.DataFrame) -> Dict[str, AttributeInfo]:
    return {
        str(col): infer_attribute(frame[col], str(col))
        for col in frame.columns if not is_reserved(str(col))
    }


def default_order(attribute: str, values: Iterable[str]) -> Optional[List[str]]:
    present = {v for v in values}
    if not present:
        return None
    for key, order in KNOWN_ORDERS.items():
        if present <= set(order):
            return [v for v in order if v in present]
    return None


@dataclass
class EncodedAttribute:
    name: str
    kind: str
    categories: List[str] = field(default_factory=list)
    codes: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    multi: List[List[int]] = field(default_factory=list)

    def code_of(self, value: str) -> int:
        try:
            return self.categories.index(str(value).strip())
        except ValueError:
            return -1

    def present(self, i: int) -> bool:
        if self.kind == "categorical":
            return self.codes[i] >= 0
        if self.kind == "numeric":
            return not math.isnan(self.values[i])
        return bool(self.multi[i])


@dataclass
class EncodedCohort:
    student_ids: List[str]
    attributes: Dict[str, EncodedAttribute] = field(default_factory=dict)
    prefer: Dict[int, List[int]] = field(default_factory=dict)
    avoid: Dict[int, List[int]] = field(default_factory=dict)
    index: Dict[str, int] = field(default_factory=dict)
    _stripped_index: Dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {sid: i for i, sid in enumerate(self.student_ids)}

    @property
    def n(self) -> int:
        return len(self.student_ids)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        kinds: Optional[Dict[str, str]] = None,
        orders: Optional[Dict[str, List[str]]] = None,
        missing_as_category: Iterable[str] = (),
    ) -> "EncodedCohort":
        kinds = dict(kinds or {})
        orders = dict(orders or {})
        missing_cat = set(missing_as_category)
        ids = [(_clean(v) or "") for v in frame["student_id"].tolist()]
        cohort = cls(student_ids=ids)
        infos = infer_attributes(frame)
        for name, info in infos.items():
            kind = kinds.get(name, info.kind)
            column = frame[name].tolist()
            if kind == "categorical":
                cleaned = [_clean(v) for v in column]
                if name in missing_cat:
                    cleaned = [v if v is not None else "(missing)" for v in cleaned]
                counts: Dict[str, int] = {}
                for v in cleaned:
                    if v is not None:
                        counts[v] = counts.get(v, 0) + 1
                categories = sorted(counts, key=lambda k: (-counts[k], k))
                lookup = {c: k for k, c in enumerate(categories)}
                codes = [lookup[v] if v is not None else -1 for v in cleaned]
                cohort.attributes[name] = EncodedAttribute(name, "categorical", categories, codes=codes)
            elif kind == "numeric":
                numbers = [_to_float(v) for v in column]
                if any(x is None for x, raw in zip(numbers, column) if _clean(raw) is not None):
                    order = orders.get(name) or default_order(name, info.values)
                    if order:
                        rank = {v: float(k) for k, v in enumerate(order)}
                        numbers = [rank.get(_clean(v)) if _clean(v) is not None else None
                                   for v in column]
                values = [float("nan") if x is None else float(x) for x in numbers]
                cohort.attributes[name] = EncodedAttribute(name, "numeric", values=values)
            elif kind == "multi":
                lists = []
                counts = {}
                for v in column:
                    text = _clean(v)
                    items = sorted({p.strip() for p in text.split(MULTI_SEPARATOR) if p.strip()}) if text else []
                    lists.append(items)
                    for item in items:
                        counts[item] = counts.get(item, 0) + 1
                categories = sorted(counts, key=lambda k: (-counts[k], k))
                lookup = {c: k for k, c in enumerate(categories)}
                cohort.attributes[name] = EncodedAttribute(
                    name, "multi", categories, multi=[[lookup[x] for x in items] for items in lists]
                )

        for key, target in (("prefer_teammates", cohort.prefer), ("avoid_teammates", cohort.avoid)):
            if key not in frame.columns:
                continue
            for i, raw in enumerate(frame[key].tolist()):
                text = _clean(raw)
                if not text:
                    continue
                listed = []
                for other in text.split(MULTI_SEPARATOR):
                    j = cohort.lookup(other)
                    if j is not None and j != i and j not in listed:
                        listed.append(j)
                if listed:
                    target[i] = listed
        return cohort

    def lookup(self, student_id: Any) -> Optional[int]:
        text = _clean(student_id)
        if text is None:
            return None
        if text in self.index:
            return self.index[text]
        if not self._stripped_index and self.index:
            self._stripped_index = {}
            for sid, i in self.index.items():
                self._stripped_index.setdefault(sid.lstrip("0") or "0", i)
        return self._stripped_index.get(text.lstrip("0") or "0")

    def subset(self, student_ids: Sequence[str]) -> "EncodedCohort":
        picks = []
        for sid in student_ids:
            i = self.lookup(sid)
            if i is None:
                raise KeyError(f"Unknown student id {sid!r}")
            picks.append(i)
        remap = {old: new for new, old in enumerate(picks)}
        sub = EncodedCohort(student_ids=[self.student_ids[i] for i in picks])
        for name, attr in self.attributes.items():
            sub.attributes[name] = EncodedAttribute(
                name=name,
                kind=attr.kind,
                categories=attr.categories,
                codes=[attr.codes[i] for i in picks] if attr.codes else [],
                values=[attr.values[i] for i in picks] if attr.values else [],
                multi=[attr.multi[i] for i in picks] if attr.multi else [],
            )
        for source, target in ((self.prefer, sub.prefer), (self.avoid, sub.avoid)):
            for old, listed in source.items():
                if old in remap:
                    kept = [remap[j] for j in listed if j in remap]
                    if kept:
                        target[remap[old]] = kept
        return sub
