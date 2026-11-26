from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .roster import (
    AttributeInfo,
    EncodedCohort,
    _clean,
    _to_float,
    infer_attribute,
    infer_attributes,
    is_reserved,
)
from .sizing import plan_team_sizes

CohortLike = Union[pd.DataFrame, EncodedCohort]

IMPLICATION_MIN_SUPPORT = 3
IMPLICATION_PURITY = 0.95
SAME_INFORMATION_R = 0.999
MIN_REDUNDANCY_SAMPLE = 30
MAX_PAIRWISE_ATTRIBUTES = 60
MAX_PAIRWISE_CARDINALITY = 100
_MAX_DENSE_TABLE = 250_000


def _fmt_num(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def format_top_values(info: AttributeInfo, top: int = 3) -> str:
    if not info.values:
        return ""
    parts = [f"{v} {info.counts.get(v, 0)}" for v in info.values[:top]]
    text = " · ".join(parts)
    if len(info.values) > top:
        text += " …"
    return text


def format_stats(info: AttributeInfo) -> str:
    s = info.stats
    if not s:
        return ""
    return (f"{_fmt_num(s['min'])}–{_fmt_num(s['max'])} · mean {_fmt_num(s['mean'], 1)}"
            f" · sd {_fmt_num(s['std'], 1)}")


def _cells(frame: pd.DataFrame, name: str) -> List[Optional[str]]:
    return [_clean(v) for v in frame[name].tolist()]


def _encode_cells(cells: Sequence[Optional[str]]) -> Tuple[np.ndarray, List[str]]:
    counts = Counter(c for c in cells if c is not None)
    categories = sorted(counts, key=lambda k: (-counts[k], k))
    lookup = {c: k for k, c in enumerate(categories)}
    codes = np.fromiter((lookup[c] if c is not None else -1 for c in cells),
                        dtype=np.int64, count=len(cells))
    return codes, categories


def _numbers(frame: pd.DataFrame, name: str) -> np.ndarray:
    return np.array([np.nan if (x := _to_float(v)) is None else x
                     for v in frame[name].tolist()], dtype=float)


@dataclass
class _Column:
    name: str
    kind: str
    codes: Optional[np.ndarray] = None
    categories: Optional[List[str]] = None
    values: Optional[np.ndarray] = None

    @property
    def present(self) -> np.ndarray:
        if self.kind == "numeric":
            return ~np.isnan(self.values)
        return self.codes >= 0


def _prepare(frame: pd.DataFrame, name: str, kind: str) -> _Column:
    if kind == "numeric":
        return _Column(name, "numeric", values=_numbers(frame, name))
    codes, categories = _encode_cells(_cells(frame, name))
    return _Column(name, "categorical", codes=codes, categories=categories)


def _default_attributes(infos: Dict[str, AttributeInfo],
                        allowed: Iterable[str] = ("categorical", "numeric")) -> List[str]:
    allowed = set(allowed)
    return [name for name, info in infos.items() if info.kind in allowed]


def analysis_columns(frame: pd.DataFrame, infos: Dict[str, AttributeInfo]) -> List[str]:
    names = [n for n in _default_attributes(infos)
             if n in frame.columns
             and not (infos[n].kind == "categorical"
                      and infos[n].n_unique > MAX_PAIRWISE_CARDINALITY)]
    return names[:MAX_PAIRWISE_ATTRIBUTES]


def _analysis_kind(info: Optional[AttributeInfo]) -> Optional[str]:
    if info is None:
        return None
    if info.kind == "numeric":
        return "numeric"
    if info.kind in ("categorical", "text"):
        return "categorical"
    return None


def _contingency(a: np.ndarray, b: np.ndarray, ka: int, kb: int):
    both = (a >= 0) & (b >= 0)
    if ka * kb > _MAX_DENSE_TABLE:
        keys, counts = np.unique(a[both].astype(np.int64) * kb + b[both], return_counts=True)
        return sp.csr_matrix((counts.astype(float), (keys // kb, keys % kb)), shape=(ka, kb))
    flat = a[both] * kb + b[both]
    return np.bincount(flat, minlength=ka * kb).reshape(ka, kb).astype(float)


def cramers_v(table: np.ndarray) -> float:
    if sp.issparse(table):
        counts = table.tocoo()
        row = np.asarray(table.sum(axis=1), dtype=float).ravel()
        col = np.asarray(table.sum(axis=0), dtype=float).ravel()
        n = float(row.sum())
        r, k = int((row > 0).sum()), int((col > 0).sum())
        if n < 2 or r < 2 or k < 2:
            return 0.0
        keep = counts.data > 0
        observed = counts.data[keep].astype(float)
        expected = row[counts.row[keep]] * col[counts.col[keep]] / n
        return _bias_corrected_v(float((observed ** 2 / expected).sum() - n), n, r, k)
    table = np.asarray(table, dtype=float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n = table.sum()
    if table.ndim != 2 or n < 2:
        return 0.0
    r, k = table.shape
    if r < 2 or k < 2:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return _bias_corrected_v(chi2, n, r, k)


def _bias_corrected_v(chi2: float, n: float, r: int, k: int) -> float:
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0
    return float(min(1.0, math.sqrt(phi2_corr / denom)))


def correlation_ratio(codes: np.ndarray, values: np.ndarray) -> float:
    mask = (codes >= 0) & ~np.isnan(values)
    x = values[mask]
    g = codes[mask]
    n = len(x)
    if n < 3:
        return 0.0
    ss_total = float(((x - x.mean()) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    groups = np.unique(g)
    k = len(groups)
    if k < 2 or n - k <= 0:
        return 0.0
    sums = np.bincount(g, weights=x)
    counts = np.bincount(g)
    used = counts > 0
    means = sums[used] / counts[used]
    ss_between = float((counts[used] * (means - x.mean()) ** 2).sum())
    eta2 = ss_between / ss_total
    eta2_adj = 1.0 - (1.0 - eta2) * (n - 1) / (n - k)
    return float(math.sqrt(min(1.0, max(0.0, eta2_adj))))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return 0.0
    xs, ys = x[mask], y[mask]
    if xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _association(a: _Column, b: _Column) -> float:
    if a.kind == "categorical" and b.kind == "categorical":
        return cramers_v(_contingency(a.codes, b.codes, len(a.categories), len(b.categories)))
    if a.kind == "numeric" and b.kind == "numeric":
        return abs(_pearson(a.values, b.values))
    cat, num = (a, b) if a.kind == "categorical" else (b, a)
    return correlation_ratio(cat.codes, num.values)


def association_matrix(frame: pd.DataFrame, attributes: Optional[Sequence[str]] = None,
                       infos: Optional[Dict[str, AttributeInfo]] = None) -> pd.DataFrame:
    infos = infos or infer_attributes(frame)
    names = list(attributes) if attributes is not None else analysis_columns(frame, infos)
    columns = {}
    for name in names:
        kind = _analysis_kind(infos.get(name))
        if kind is not None and name in frame.columns:
            columns[name] = _prepare(frame, name, kind)
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        matrix.loc[a, a] = 1.0
        for b in names[i + 1:]:
            if a in columns and b in columns:
                value = _association(columns[a], columns[b])
                matrix.loc[a, b] = matrix.loc[b, a] = round(value, 4)
    return matrix


@dataclass
class _Implication:
    a: str
    b: str
    joint: int
    n_a: int
    n_b: int

    @property
    def exact(self) -> bool:
        return self.joint >= IMPLICATION_PURITY * self.n_b


@dataclass
class _Dependency:
    purity: float
    evidence: float
    improvement: float
    implications: List[_Implication]
    holds: bool


def _row_max(table, i: int) -> Tuple[int, float]:
    if sp.issparse(table):
        start, end = table.indptr[i], table.indptr[i + 1]
        cols, vals = table.indices[start:end], table.data[start:end]
        best = vals.max()
        return int(cols[vals == best].min()), float(best)
    j = int(np.argmax(table[i]))
    return j, float(table[i, j])


def _dependency(table, cats_a: List[str], cats_b: List[str]) -> _Dependency:
    if sp.issparse(table):
        table = table.tocsr()
    n = table.sum()
    row_tot = np.asarray(table.sum(axis=1), dtype=float).ravel()
    col_tot = np.asarray(table.sum(axis=0), dtype=float).ravel()
    implications: List[_Implication] = []
    support_rows = 0.0
    pure_rows = 0.0
    for i, a in enumerate(cats_a):
        if row_tot[i] < IMPLICATION_MIN_SUPPORT:
            continue
        j, top = _row_max(table, i)
        support_rows += row_tot[i]
        pure_rows += top
        if top >= IMPLICATION_PURITY * row_tot[i]:
            implications.append(_Implication(a, cats_b[j], int(top),
                                             int(row_tot[i]), int(col_tot[j])))
    if n <= 0 or support_rows <= 0 or (col_tot > 0).sum() < 2:
        return _Dependency(0.0, 0.0, 0.0, implications, False)
    purity = pure_rows / support_rows
    evidence = support_rows / n
    base = col_tot.max() / n
    improvement = (purity - base) / (1 - base) if base < 1 else 0.0
    majority = cats_b[int(np.argmax(col_tot))]
    explains_minority = any(imp.b != majority for imp in implications)
    holds = (evidence >= 0.5 and purity >= IMPLICATION_PURITY
             and improvement >= 0.5 and explains_minority)
    return _Dependency(purity, evidence, improvement, implications, holds)


def _implication_text(det: str, dep: str, imp: _Implication) -> str:
    if imp.exact:
        return (f"{dep} = {imp.b} exactly when {det} = {imp.a} "
                f"({imp.joint}/{max(imp.n_a, imp.n_b)})")
    return f"{det} = {imp.a} implies {dep} = {imp.b} ({imp.joint}/{imp.n_a})"


def _dependency_text(det: str, dep: str, d: _Dependency) -> str:
    ranked = sorted(d.implications, key=lambda m: (not m.exact, m.n_b, -m.joint))
    examples = "; ".join(_implication_text(det, dep, m) for m in ranked[:2])
    head = (f"{dep} follows from {det} for {d.purity:.0%} of students"
            if d.purity < 0.995 else f"every {det} value belongs to a single {dep}")
    return f"{examples} — {head}" if examples else head


def _binning(cat: _Column, num: _Column) -> Optional[List[Tuple[str, float, float]]]:
    mask = (cat.codes >= 0) & ~np.isnan(num.values)
    if mask.sum() < 4:
        return None
    ranges = []
    supported = 0
    for code in np.unique(cat.codes[mask]):
        xs = num.values[mask & (cat.codes == code)]
        ranges.append((cat.categories[code], float(xs.min()), float(xs.max())))
        supported += len(xs) >= IMPLICATION_MIN_SUPPORT
    if len(ranges) < 2 or supported < 2:
        return None
    ranges.sort(key=lambda r: (r[1], r[2]))
    distinct_numbers = len(np.unique(num.values[mask]))
    if distinct_numbers <= len(ranges):
        return None
    for (_, _, hi), (_, lo, _) in zip(ranges, ranges[1:]):
        if hi >= lo:
            return None
    return ranges


def _redundancy(a: _Column, b: _Column, threshold: float) -> Optional[Dict[str, Any]]:
    enough = int((a.present & b.present).sum()) >= MIN_REDUNDANCY_SAMPLE
    if a.kind == "numeric" and b.kind == "numeric":
        r = _pearson(a.values, b.values)
        if abs(r) >= SAME_INFORMATION_R and (a.present & b.present).sum() >= 3:
            mask = ~np.isnan(a.values) & ~np.isnan(b.values)
            slope, intercept = np.polyfit(a.values[mask], b.values[mask], 1)
            relation = ""
            if abs(abs(slope) - 1) < 1e-6:
                sign = "+" if slope > 0 else "−"
                relation = f" ({b.name} = {_fmt_num(round(intercept, 3))} {sign} {a.name})"
            return dict(a=a.name, b=b.name, strength=1.0, relation="same",
                        determinant=a.name, dependent=b.name,
                        explanation=f"{a.name} and {b.name} are the same information{relation}")
        if abs(r) >= threshold and enough:
            return dict(a=a.name, b=b.name, strength=abs(r), relation="correlated",
                        determinant=None, dependent=None,
                        explanation=f"{a.name} and {b.name} are strongly correlated (r = {r:+.2f})")
        return None

    if not enough:
        return None
    strength = _association(a, b)
    if a.kind != b.kind:
        cat, num = (a, b) if a.kind == "categorical" else (b, a)
        ranges = _binning(cat, num)
        if ranges is not None:
            bins = " · ".join(f"{label}: {_fmt_num(lo)}–{_fmt_num(hi)}" for label, lo, hi in ranges)
            return dict(a=num.name, b=cat.name, strength=1.0, relation="binning",
                        determinant=num.name, dependent=cat.name,
                        explanation=f"{cat.name} is a binning of {num.name} ({bins})")
        if strength >= threshold:
            return dict(a=cat.name, b=num.name, strength=strength, relation="associated",
                        determinant=None, dependent=None,
                        explanation=(f"{num.name} differs strongly between {cat.name} groups "
                                     f"(eta = {strength:.2f})"))
        return None

    table = _contingency(a.codes, b.codes, len(a.categories), len(b.categories))
    ab = _dependency(table, a.categories, b.categories)
    ba = _dependency(table.T, b.categories, a.categories)
    if ab.holds and ba.holds:
        pairs = ", ".join(f"{m.a} ↔ {m.b}" for m in sorted(ab.implications, key=lambda m: -m.joint)[:3])
        return dict(a=a.name, b=b.name, strength=max(strength, min(ab.purity, ba.purity)),
                    relation="same", determinant=a.name, dependent=b.name,
                    explanation=(f"{a.name} and {b.name} are the same grouping under different "
                                 f"labels ({pairs})"))
    if ab.holds or ba.holds:
        det, dep, d = (a, b, ab) if ab.holds else (b, a, ba)
        return dict(a=det.name, b=dep.name, strength=max(strength, d.purity),
                    relation="determines", determinant=det.name, dependent=dep.name,
                    explanation=_dependency_text(det.name, dep.name, d))
    if strength >= threshold:
        return dict(a=a.name, b=b.name, strength=strength, relation="associated",
                    determinant=None, dependent=None,
                    explanation=f"{a.name} and {b.name} are strongly associated "
                                f"(Cramér's V = {strength:.2f})")
    return None


def redundant_pairs(frame: pd.DataFrame, threshold: float = 0.85, *,
                    attributes: Optional[Sequence[str]] = None,
                    infos: Optional[Dict[str, AttributeInfo]] = None) -> List[Dict[str, Any]]:
    infos = infos or infer_attributes(frame)
    names = list(attributes) if attributes is not None else analysis_columns(frame, infos)
    columns: List[_Column] = []
    for name in names:
        kind = _analysis_kind(infos.get(name))
        if kind is not None and name in frame.columns:
            columns.append(_prepare(frame, name, kind))
    found = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            record = _redundancy(a, b, threshold)
            if record is not None:
                record["strength"] = round(float(record["strength"]), 4)
                found.append(record)
    order = {"same": 0, "binning": 1, "determines": 2, "correlated": 3, "associated": 4}
    found.sort(key=lambda r: (order[r["relation"]], -r["strength"]))
    return found


def unavoidable_count_violations(matching: int, sizes: Sequence[int],
                                 min_count: Optional[int] = None,
                                 max_count: Optional[int] = None) -> int:
    units = 0
    if min_count is not None and min_count > 0:
        need = sum(min(min_count, s) for s in sizes)
        units += max(0, need - matching) + sum(max(0, min_count - s) for s in sizes)
    if max_count is not None:
        slots = sum(min(max(0, max_count), s) for s in sizes)
        units += max(0, matching - slots)
    return int(units)


def tightest_feasible_max(matching: int, sizes: Sequence[int]) -> int:
    if not sizes or matching <= 0:
        return 0
    for cap in range(0, max(sizes) + 1):
        if unavoidable_count_violations(matching, sizes, max_count=cap) == 0:
            return cap
    return max(sizes)


def loosest_feasible_min(matching: int, sizes: Sequence[int]) -> int:
    if not sizes:
        return 0
    best = 0
    for floor in range(1, min(sizes) + 1):
        if unavoidable_count_violations(matching, sizes, min_count=floor) == 0:
            best = floor
        else:
            break
    return best


def _value_counts(source: CohortLike, attribute: str,
                  infos: Optional[Dict[str, AttributeInfo]] = None) -> Tuple[Dict[str, int], int, str]:
    if isinstance(source, EncodedCohort):
        attr = source.attributes.get(attribute)
        if attr is None:
            raise KeyError(f"Unknown attribute {attribute!r}")
        counts: Dict[str, int] = {}
        if attr.kind == "categorical":
            for code in attr.codes:
                if code >= 0:
                    counts[attr.categories[code]] = counts.get(attr.categories[code], 0) + 1
        elif attr.kind == "multi":
            for codes in attr.multi:
                for code in set(codes):
                    counts[attr.categories[code]] = counts.get(attr.categories[code], 0) + 1
        return counts, source.n, attr.kind
    if attribute not in source.columns:
        raise KeyError(f"Unknown attribute {attribute!r}")
    info = (infos or {}).get(attribute) or infer_attribute(source[attribute], attribute)
    kind = info.kind if info.kind in ("multi", "numeric") else "categorical"
    if kind == "numeric":
        return {}, len(source), kind
    return dict(info.counts), len(source), kind


def _numeric_values(source: CohortLike, attribute: str) -> List[float]:
    if isinstance(source, EncodedCohort):
        attr = source.attributes.get(attribute)
        if attr is None:
            raise KeyError(f"Unknown attribute {attribute!r}")
        return [v for v in attr.values if not math.isnan(v)]
    return [x for x in (_to_float(v) for v in source[attribute].tolist()) if x is not None]


def expected_team_counts(cohort_or_frame: CohortLike, attribute: str,
                         team_size: float = 5) -> Dict[str, float]:
    counts, n, kind = _value_counts(cohort_or_frame, attribute)
    if kind == "numeric" or n == 0:
        return {}
    ordered = sorted(counts, key=lambda k: (-counts[k], k))
    return {value: counts[value] / n * team_size for value in ordered}


def _sizes_for(n: int, team_size: int, num_teams: Optional[int],
               team_sizes: Optional[Sequence[int]]) -> List[int]:
    if team_sizes:
        return [int(s) for s in team_sizes]
    return plan_team_sizes(n, team_size, team_size, team_size, num_teams or 0)


def _range_label(attribute: str, lo: Optional[float], hi: Optional[float]) -> str:
    if lo is not None and hi is not None:
        return f"{attribute} {_fmt_num(lo)}–{_fmt_num(hi)}"
    if lo is not None:
        return f"{attribute} ≥ {_fmt_num(lo)}"
    if hi is not None:
        return f"{attribute} ≤ {_fmt_num(hi)}"
    return f"any {attribute}"


def _target_row(label: str, value: Any, matching: int, n: int, sizes: List[int]) -> Dict[str, Any]:
    teams = len(sizes)
    typical = max(set(sizes), key=sizes.count) if sizes else 0
    per_team = matching / n * typical if n else 0.0
    max_ok = tightest_feasible_max(matching, sizes)
    min_ok = loosest_feasible_min(matching, sizes)
    parts = [f"{label}: {matching} student{'s' if matching != 1 else ''} ≈ "
             f"{per_team:.2f} per team of {typical}"]
    advice = []
    if matching == 0:
        advice.append("no students have this value")
    else:
        advice.append(f"'at most {max_ok}' is feasible")
        if max_ok >= 1:
            missed = unavoidable_count_violations(matching, sizes, max_count=max_ok - 1)
            advice.append(f"'at most {max_ok - 1}' would leave ≥ {missed} violations")
        if min_ok >= 1:
            advice.append(f"'at least {min_ok}' is feasible")
        if min_ok < typical:
            short = unavoidable_count_violations(matching, sizes, min_count=min_ok + 1)
            if short:
                advice.append(f"'at least {min_ok + 1}' would leave ≥ {short} violations")
    hint = parts[0] + " → " + ", ".join(advice)
    return {
        "value": value,
        "count": matching,
        "share": matching / n if n else 0.0,
        "teams": teams,
        "team_size": typical,
        "expected_per_team": per_team,
        "feasible_max": max_ok,
        "feasible_min": min_ok,
        "violations_if_max_lower": (
            unavoidable_count_violations(matching, sizes, max_count=max_ok - 1)
            if max_ok >= 1 else 0),
        "hint": hint,
    }


def count_rule_targets(cohort_or_frame: CohortLike, attribute: str, team_size: int = 5, *,
                       num_teams: Optional[int] = None,
                       team_sizes: Optional[Sequence[int]] = None,
                       values: Optional[Sequence[str]] = None,
                       numeric_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                       top: Optional[int] = None) -> List[Dict[str, Any]]:
    if numeric_range is not None:
        numbers = _numeric_values(cohort_or_frame, attribute)
        n = cohort_or_frame.n if isinstance(cohort_or_frame, EncodedCohort) else len(cohort_or_frame)
        lo, hi = (list(numeric_range) + [None, None])[:2]
        matching = sum(1 for x in numbers
                       if (lo is None or x >= lo) and (hi is None or x <= hi))
        sizes = _sizes_for(n, team_size, num_teams, team_sizes)
        return [_target_row(_range_label(attribute, lo, hi), (lo, hi), matching, n, sizes)]

    counts, n, kind = _value_counts(cohort_or_frame, attribute)
    if kind == "numeric":
        raise ValueError(f"{attribute!r} is numeric: pass numeric_range=(lo, hi)")
    sizes = _sizes_for(n, team_size, num_teams, team_sizes)
    ordered = list(values) if values is not None else sorted(counts, key=lambda k: (-counts[k], k))
    if top is not None:
        ordered = ordered[:top]
    return [_target_row(str(v), v, counts.get(v, 0), n, sizes) for v in ordered]


def attribute_overview(frame: pd.DataFrame,
                       infos: Optional[Dict[str, AttributeInfo]] = None,
                       *, team_size: int = 5) -> List[Dict[str, Any]]:
    from .defaults import suggest_attribute_rules

    infos = infos or infer_attributes(frame)
    suggestions = suggest_attribute_rules(frame, infos, team_size=team_size)
    rows = []
    for name, info in infos.items():
        if is_reserved(name):
            continue
        s = suggestions.get(name)
        rows.append({
            "name": name,
            "kind": info.kind,
            "coverage": round(info.coverage, 4),
            "n_unique": info.n_unique,
            "top_values": format_top_values(info),
            "stats": format_stats(info),
            "stats_values": dict(info.stats),
            "role": s.role if s else "",
            "suggested_goal": s.goal if s else "ignore",
            "suggested_weight": s.weight if s else 0.0,
            "why": s.why if s else "",
        })
    return rows

