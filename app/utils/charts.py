from __future__ import annotations

import colorsys
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.ids import team_sort_key
from .roster_view import MISSING_LABEL

BASE_COLORS: Tuple[str, ...] = (
    "#171C54",
    "#2871AD",
    "#5D8972",
    "#D17B58",
    "#BD0226",
    "#E1AE67",
    "#B62D7A",
    "#8A9A2B",
    "#7B5EA7",
    "#2A9D8F",
)
MISSING_COLOR = "#A0A4A8"
MISSING_VALUES = frozenset({MISSING_LABEL.lower(), "missing", "unknown", "(unknown)", "n/a", ""})
_SHADE_OFFSETS = (0.22, -0.15, 0.34, -0.25, 0.11, -0.08, 0.28, -0.20)
_LIGHTNESS_RANGE = (0.16, 0.84)
_MIN_DISTANCE = 38.0
MAX_PALETTE_COLORS = 64


def natural_sort_key(label: object) -> Tuple:
    return team_sort_key(str(label))


def natural_sorted(labels: Iterable[object]) -> List:
    return sorted(dict.fromkeys(labels), key=natural_sort_key)


def is_missing_value(value: object) -> bool:
    return value is None or str(value).strip().casefold() in MISSING_VALUES


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    text = color.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02X}" for c in rgb)


def _distance(a: str, b: str) -> float:
    return sum((x - y) ** 2 for x, y in zip(_hex_to_rgb(a), _hex_to_rgb(b))) ** 0.5


def _shade(color: str, offset: float, hue_shift: float = 0.0) -> str:
    r, g, b = (c / 255.0 for c in _hex_to_rgb(color))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    low, high = _LIGHTNESS_RANGE
    l = min(high, max(low, l + offset))
    r, g, b = colorsys.hls_to_rgb((h + hue_shift) % 1.0, l, s)
    return _rgb_to_hex((r * 255, g * 255, b * 255))


def _candidates() -> Iterator[str]:
    yield from BASE_COLORS
    for offset in _SHADE_OFFSETS:
        for base in BASE_COLORS:
            yield _shade(base, offset)
    depth = 1
    while True:
        steps = 2 ** depth
        for k in range(1, steps, 2):
            fraction = k / steps
            for sign in (1, -1):
                offset = sign * 0.32 * fraction
                shift = sign * 0.04 * fraction
                for base in BASE_COLORS:
                    yield _shade(base, offset, shift)
        depth += 1


@lru_cache(maxsize=64)
def _palette(n: int) -> Tuple[str, ...]:
    chosen: List[str] = []
    reserved = [MISSING_COLOR]
    threshold = _MIN_DISTANCE
    while len(chosen) < n:
        tried = 0
        for candidate in _candidates():
            if len(chosen) >= n or tried > max(4000, 8 * n):
                break
            tried += 1
            if candidate in chosen:
                continue
            if all(_distance(candidate, other) >= threshold for other in chosen + reserved):
                chosen.append(candidate)
        threshold = threshold / 2 if threshold > 1 else 0.5
    return tuple(chosen)


def category_colors(n: int) -> List[str]:
    n = max(0, int(n))
    if n <= MAX_PALETTE_COLORS:
        return list(_palette(n))
    base = _palette(MAX_PALETTE_COLORS)
    return [base[i % MAX_PALETTE_COLORS] for i in range(n)]


def category_color_map(values: Iterable[object], *, order: Optional[Sequence[object]] = None
                       ) -> Dict[str, str]:
    labels = [str(v) for v in dict.fromkeys(values) if v is not None]
    if order is not None:
        ranked = [str(v) for v in order if str(v) in labels]
        ranked += natural_sorted(v for v in labels if v not in ranked)
    else:
        ranked = natural_sorted(labels)
    real = [v for v in ranked if not is_missing_value(v)]
    colors = dict(zip(real, category_colors(len(real))))
    return {v: (MISSING_COLOR if is_missing_value(v) else colors[v]) for v in ranked}


HEATMAP_COLOR_MODES = {
    "single": "One gradient",
    "per_project": "A gradient per project",
    "uploaded_vs_upcoming": "Uploaded vs upcoming",
}
HEATMAP_GRADIENTS: List[Tuple[str, List[str]]] = [
    ("blue", ["#DCE8F5", "#2871AD", "#171C54"]),
    ("orange", ["#F7E3D8", "#D17B58", "#7A3418"]),
    ("green", ["#E0EDE5", "#5D8972", "#22402F"]),
    ("purple", ["#E9E1F2", "#7B5BA6", "#3A2457"]),
    ("red", ["#F6D9DE", "#BD0226", "#5C0012"]),
    ("gold", ["#F7ECD6", "#E1AE67", "#7D5617"]),
    ("teal", ["#D8EFEF", "#2A9D9A", "#0F4A48"]),
    ("slate", ["#E4E7EE", "#7A859E", "#2E3547"]),
]
REPEAT_HEATMAP_MAX_STUDENTS = 2000
REPEAT_HEATMAP_COLOR_MODES = {
    "single": "One gradient",
    "per_project": "By project",
    "uploaded_vs_upcoming": "Uploaded vs upcoming",
}


def heatmap_color_groups(imported: Sequence[bool], mode: str) -> List[Tuple[List[int], int]]:
    columns = list(range(len(imported)))
    if mode == "per_project":
        return [([i], i % len(HEATMAP_GRADIENTS)) for i in columns]
    if mode == "uploaded_vs_upcoming":
        groups = [([i for i in columns if imported[i]], 0), ([i for i in columns if not imported[i]], 1)]
        return [g for g in groups if g[0]]
    return [(columns, 0)] if columns else []


def rotation_heatmap_traces(
    z_columns: Sequence[Sequence[float]],
    labels: Sequence[str],
    student_ids: Sequence[str],
    imported: Sequence[bool],
    mode: str,
) -> List[go.Heatmap]:
    traces = []
    for cols, gradient in heatmap_color_groups(imported, mode):
        z = [[z_columns[i][row] for i in cols] for row in range(len(student_ids))]
        values = [v for i in cols for v in z_columns[i] if not np.isnan(v)]
        lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        stops = HEATMAP_GRADIENTS[gradient][1]
        traces.append(go.Heatmap(
            z=z,
            x=[labels[i] for i in cols],
            y=list(student_ids),
            zmin=lo,
            zmax=hi,
            colorscale=[[k / (len(stops) - 1), c] for k, c in enumerate(stops)],
            showscale=False,
            hovertemplate="Student %{y} · %{x}: team %{z}<extra></extra>",
        ))
    return traces


def repeat_color_groups(met: Dict[Tuple[int, int], List[int]], imported: Sequence[bool],
                        mode: str) -> List[Tuple[List[Tuple[int, int]], int, str]]:
    pairs = [pair for pair, where in met.items() if where]
    if mode == "per_project":
        by_project: Dict[int, List[Tuple[int, int]]] = {}
        for pair in pairs:
            by_project.setdefault(max(met[pair]), []).append(pair)
        return [(by_project[k], k % len(HEATMAP_GRADIENTS), str(k)) for k in sorted(by_project)]
    if mode == "uploaded_vs_upcoming":
        uploaded = [p for p in pairs if all(imported[k] for k in met[p])]
        upcoming = [p for p in pairs if not all(imported[k] for k in met[p])]
        return [g for g in ((uploaded, 0, "uploaded"), (upcoming, 1, "upcoming")) if g[0]]
    return [(pairs, 0, "all")] if pairs else []


def repeat_pair_hover(count: int, where: Sequence[int], labels: Sequence[str]) -> str:
    return f"{count} team(s) · " + ", ".join(labels[k] for k in dict.fromkeys(where))


def repeat_heatmap_trace(
    student_ids: Sequence[str],
    cohort_ids: Sequence[str],
    counts: Dict[Tuple[int, int], int],
    met: Dict[Tuple[int, int], List[int]],
    labels: Sequence[str],
    imported: Sequence[bool],
    mode: str,
    why: Optional[Dict[Tuple[int, int], str]] = None,
) -> Tuple[go.Heatmap, List[Tuple[List[Tuple[int, int]], int, str]]]:
    n = len(student_ids)
    if n > REPEAT_HEATMAP_MAX_STUDENTS:
        placeholder = go.Heatmap(z=[[np.nan]], x=[""], y=[""], showscale=False,
                                 hoverinfo="skip")
        return placeholder, []
    position = {sid: i for i, sid in enumerate(student_ids)}
    top = max([2] + list(counts.values()))
    span = top + 1
    groups = repeat_color_groups({p: w for p, w in met.items() if counts.get(p, 0) > 0},
                                 imported, mode)
    z = np.full((n, n), np.nan)
    hover = np.full((n, n), "", dtype=object)
    for band, (pairs, _gradient, _name) in enumerate(groups):
        for a, b in pairs:
            ia, ib = position.get(cohort_ids[a]), position.get(cohort_ids[b])
            if ia is None or ib is None:
                continue
            value = band * span + counts[(a, b)]
            text = repeat_pair_hover(counts[(a, b)], met[(a, b)], labels)
            if why and why.get((a, b)):
                text += f" · why: {why[(a, b)]}"
            z[ia, ib] = z[ib, ia] = value
            hover[ia, ib] = hover[ib, ia] = text
    zmax = (max(len(groups), 1) - 1) * span + top
    colorscale = []
    for band, (_pairs, gradient, _name) in enumerate(groups or [([], 0, "all")]):
        stops = HEATMAP_GRADIENTS[gradient][1]
        for k, colour in enumerate(stops):
            colorscale.append([(band * span + top * k / (len(stops) - 1)) / zmax, colour])
    trace = go.Heatmap(
        z=z,
        x=[str(s) for s in student_ids],
        y=[str(s) for s in student_ids],
        customdata=hover,
        zmin=0,
        zmax=zmax,
        colorscale=colorscale,
        showscale=len(groups) <= 1 and mode == "single",
        colorbar=dict(title="Teams"),
        hoverongaps=False,
        hovertemplate="Student %{y} with %{x}: %{customdata}<extra></extra>",
    )
    return trace, groups


HEATMAP_LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white")
HEATMAP_AXIS = dict(showgrid=False, zeroline=False)


HISTORY_OFF_NOTE = ("History is switched off in the allocation rules, so these are not counted "
                    "as repeats (Allocation Rules › 🔁 History & preferences).")
ALL_PAIRS_LABEL = "Pairs sharing 2+ teams (all projects, imported included)"
ALL_RATE_LABEL = "Share of pairs sharing 2+ teams (all projects, imported included)"


def reach_timeline_figure(timeline: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline["Project"], y=timeline["Maximum possible (no repeats)"],
                             name="Maximum possible", mode="lines+markers",
                             line=dict(color="#9AA3B5", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=timeline["Project"], y=timeline["Mean distinct teammates"],
                             name="Mean distinct teammates", mode="lines+markers",
                             line=dict(color="#171C54", width=2), marker=dict(size=9)))
    fig.update_layout(height=320, margin=dict(t=30), yaxis_title="Teammates (cumulative)",
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return fig


def cross_group_figure(exposure: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=exposure["Project"], y=exposure["Cross-group share"],
                         name="Cross-group share", marker_color="#2871AD",
                         customdata=exposure[["Cross-group pairs", "Team pairs"]].values,
                         hovertemplate="%{x}: %{y:.0%} (%{customdata[0]} of %{customdata[1]} "
                                       "pairs)<extra></extra>"))
    fig.add_trace(go.Scatter(x=exposure["Project"], y=exposure["Random expectation"],
                             name="Random teams", mode="markers",
                             marker=dict(symbol="diamond", size=12, color="#D17B58",
                                         line=dict(color="#FFFFFF", width=2)),
                             hovertemplate="random: %{y:.0%}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(t=30), yaxis=dict(tickformat=".0%",
                      range=[0, 1], title="Share of team pairs"),
                      legend=dict(orientation="h", y=1.12))
    return fig


def category_pair_figure(labels: Sequence[str], ratio: np.ndarray,
                         observed: np.ndarray) -> go.Figure:
    text = np.array([[f"{r:.2f}×" if np.isfinite(r) else "" for r in row]
                     for row in np.asarray(ratio, dtype=float)], dtype=object)
    fig = go.Figure(go.Heatmap(
        z=np.log2(np.clip(np.nan_to_num(ratio, nan=1.0), 0.125, 8.0)),
        x=labels, y=labels, zmid=0, zmin=-2, zmax=2,
        colorscale=[[0, "#BD0226"], [0.5, "#EEF0F4"], [1, "#2871AD"]],
        text=text, texttemplate="%{text}", customdata=observed,
        hovertemplate="%{y} × %{x}: %{customdata:.0f} pairings, %{text} of random"
                      "<extra></extra>",
        colorbar=dict(title="vs random", tickvals=[-2, -1, 0, 1, 2],
                      ticktext=["¼×", "½×", "1×", "2×", "4×"]),
    ))
    fig.update_layout(height=max(320, 36 * len(labels) + 120), margin=dict(t=30),
                      yaxis=dict(autorange="reversed"))
    return fig
