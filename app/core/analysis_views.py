from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Container, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from app.utils.charts import natural_sort_key
from app.utils.roster_view import MISSING_LABEL

from .. import extensions
from .diversity_metrics import DiversityMetrics
from .models import Project, Student, Team

STANDARD_ATTRIBUTES: Dict[str, Tuple[str, str]] = dict(
    extensions.hook("standard_attributes", {}))
OUTLIER_STD_FACTOR = 1.5
TOP_CONNECTED = 10
SPRING_K = 2
SPRING_ITERATIONS = 50
SPRING_SEED = 42


def attribute_value(student: Student, attr: str) -> Optional[str]:
    value = student.get_attribute_value(attr)
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value) or None
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return str(value)


def available_attributes(members: List[Student]) -> Dict[str, tuple]:
    options: Dict[str, tuple] = {}
    for attr, (label, score_key) in STANDARD_ATTRIBUTES.items():
        if any(attribute_value(m, attr) is not None for m in members):
            options[attr] = (label, score_key)
    dynamic = sorted({k for m in members for k in m.dynamic_categorical})
    for attr in dynamic:
        if attr not in options and any(attribute_value(m, attr) is not None for m in members):
            options[attr] = (attr.replace("_", " ").title(), attr)
    return options


def deep_dive_team_entries(projects: List[Project],
                           project: Optional[Project] = None) -> List[Tuple[str, Team, bool]]:
    if project is not None:
        return [(team.team_id, team, project.imported)
                for team in sorted(project.teams, key=lambda t: natural_sort_key(t.team_id))]
    return [
        (
            f"{proj.name} (imported) · {team.team_id}" if proj.imported
            else f"{proj.name} · {team.team_id}",
            team,
            proj.imported,
        )
        for proj in projects
        for team in sorted(proj.teams, key=lambda t: natural_sort_key(t.team_id))
    ]


MAX_DEEP_DIVE_VALUES = 40
OTHER_LABEL = "Other"


def capped_values(values: Iterable[str],
                  limit: int = MAX_DEEP_DIVE_VALUES) -> Optional[Set[str]]:
    counts = Counter(v for v in values if v != MISSING_LABEL)
    if len(counts) <= limit:
        return None
    ranked = sorted(counts, key=lambda v: (-counts[v], natural_sort_key(v)))
    return set(ranked[:limit]) | {MISSING_LABEL}


def bucket_value(value: str, keep: Optional[Container[str]]) -> str:
    if keep is None or value == MISSING_LABEL or value in keep:
        return value
    return OTHER_LABEL


def distribution_frame(entries: List[Tuple[str, Team, bool]], attr: str, *,
                       keep: Optional[Container[str]] = None) -> pd.DataFrame:
    rows = []
    for team_label, team, imported in entries:
        values = [bucket_value(attribute_value(m, attr) or MISSING_LABEL, keep)
                  for m in team.members]
        for val in sorted(set(values)):
            rows.append({
                "Team": team_label,
                "Imported": "imported" if imported else "generated",
                "Value": val,
                "Count": values.count(val),
            })
    return pd.DataFrame(rows)


def diversity_score_frame(entries: List[Tuple[str, Team, bool]], metrics: DiversityMetrics,
                          attr: str, score_key: str) -> pd.DataFrame:
    rows = []
    for team_label, team, imported in entries:
        team_div = metrics.calculate_team_diversity(team)
        rows.append({
            "Team": team_label,
            "Imported": "imported" if imported else "generated",
            "Diversity Score": team_div.get(score_key, team_div.get(attr, 0.0)),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FairnessView:
    fairness: Dict[str, Any]
    student_avgs: List[float]
    avg_score: Optional[float] = None
    std_score: Optional[float] = None
    outliers_low: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    outliers_high: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)


def fairness_view(metrics: DiversityMetrics, projects: List[Project]) -> FairnessView:
    fairness = metrics.get_fairness_metrics(projects)
    student_avgs = [stats["average"] for stats in fairness.get("student_stats", {}).values()]
    if not student_avgs:
        return FairnessView(fairness=fairness, student_avgs=[])
    avg_score = float(np.mean(student_avgs))
    std_score = float(np.std(student_avgs))
    low = avg_score - OUTLIER_STD_FACTOR * std_score
    high = avg_score + OUTLIER_STD_FACTOR * std_score
    outliers_low = [(sid, stats) for sid, stats in fairness["student_stats"].items()
                    if stats["average"] < low]
    outliers_high = [(sid, stats) for sid, stats in fairness["student_stats"].items()
                     if stats["average"] > high]
    return FairnessView(fairness=fairness, student_avgs=student_avgs, avg_score=avg_score,
                        std_score=std_score, outliers_low=outliers_low,
                        outliers_high=outliers_high)


@dataclass(frozen=True)
class NetworkView:
    positions: Dict[str, Tuple[float, float]]
    edges: List[Tuple[str, str, int]]
    degrees: Dict[str, int]
    density: float
    top_connected: List[Tuple[str, int]]

    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def network_view(students: List[Student], overlap_matrix: Dict[tuple, int]) -> NetworkView:
    import networkx as nx

    graph = nx.Graph()
    for student in students:
        graph.add_node(student.student_id)
    for (s1, s2), count in overlap_matrix.items():
        if count > 0:
            graph.add_edge(s1, s2, weight=count)
    pos = nx.spring_layout(graph, k=SPRING_K, iterations=SPRING_ITERATIONS, seed=SPRING_SEED)
    return NetworkView(
        positions={n: (float(pos[n][0]), float(pos[n][1])) for n in graph.nodes()},
        edges=[(a, b, int(data.get("weight", 1))) for a, b, data in graph.edges(data=True)],
        degrees={n: int(d) for n, d in graph.degree()},
        density=float(nx.density(graph)),
        top_connected=sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:TOP_CONNECTED],
    )


def __getattr__(name: str) -> Any:
    extras = extensions.hook("analysis_views_extras", {})
    if name in extras:
        return extras[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
