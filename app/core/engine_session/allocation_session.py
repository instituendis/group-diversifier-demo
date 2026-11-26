from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ...engine.adapters import pair_counts
from .. import run_history as rh
from ..models import Configuration, FormationResult, Project, Student
from .plain_form import fingerprint, projects_to_ids
from .scoring_session import set_active_reports
from .state import META_KEY, REPORTS_KEY, SCORED_POLICY_KEY
from .state import session as _session
from .summaries import summarize_reports, violations_from_reports


def build_formation_result(projects: List[Project], config: Configuration,
                           students: Sequence[Student], reports: Dict[str, Dict[str, Any]], *,
                           execution_time: float = 0.0,
                           algorithm_stats: Optional[Dict[str, Any]] = None) -> FormationResult:
    from ..diversity_metrics import DiversityMetrics

    snapshot = copy.deepcopy(config)
    metrics = DiversityMetrics(snapshot, cohort_students=list(students))
    live = set(metrics.live_attributes())
    per_attr: Dict[str, List[float]] = {}
    total_score = 0.0
    for project in projects:
        stats = metrics.calculate_project_diversity(project)
        for attr, score in (stats.get("average_scores") or {}).items():
            per_attr.setdefault(attr, []).append(score)
        for team in project.teams:
            total_score += metrics.calculate_weighted_score(metrics.calculate_team_diversity(team))
    diversity = {a: sum(v) / len(v) for a, v in per_attr.items() if v and a in live}
    stats = dict(algorithm_stats or {})
    per_project = stats.get("per_project")
    stats.setdefault("engine_summary", summarize_reports(
        reports, per_project if isinstance(per_project, dict) else None))
    return FormationResult(
        projects=projects,
        config=snapshot,
        total_score=float(total_score) if math.isfinite(total_score) else 0.0,
        diversity_scores=diversity,
        constraint_violations=violations_from_reports(projects, reports),
        overlap_matrix=pair_counts(projects),
        execution_time=float(execution_time),
        algorithm_stats=stats,
    )


def activate_allocation(projects: List[Project], config: Configuration,
                        students: Sequence[Student], reports: Dict[str, Dict[str, Any]], *,
                        execution_time: float = 0.0,
                        algorithm_stats: Optional[Dict[str, Any]] = None,
                        run_id: Optional[str] = None, state: Any = None,
                        scored_with: Any = None) -> FormationResult:
    state = _session() if state is None else state
    result = build_formation_result(projects, config, students, reports,
                                    execution_time=execution_time,
                                    algorithm_stats=algorithm_stats)
    state["projects"] = projects
    state["formation_result"] = result
    set_active_reports(reports, scored_with, state)
    if run_id is not None:
        state["active_run_id"] = run_id
    return result


def active_reports(state: Any = None) -> Dict[str, Dict[str, Any]]:
    state = _session() if state is None else state
    return state.get(REPORTS_KEY) or {}


@dataclass(frozen=True)
class ActiveAllocation:
    projects: List[Project]
    reports: Dict[str, Dict[str, Any]]
    formation_result: FormationResult
    stats_source: str
    run_id: Optional[str]
    unsaved_edits: bool


def active_allocation(state: Any = None) -> Optional[ActiveAllocation]:
    state = _session() if state is None else state
    result = state.get("formation_result")
    if result is None:
        return None
    stats = getattr(result, "algorithm_stats", None) or {}
    return ActiveAllocation(
        projects=state.get("projects") or [],
        reports=state.get(REPORTS_KEY) or {},
        formation_result=result,
        stats_source=stats.get("source") or "",
        run_id=state.get(rh.ACTIVE_KEY),
        unsaved_edits=unsaved_edits(state),
    )


def has_active_allocation(state: Any = None) -> bool:
    state = _session() if state is None else state
    return state.get("formation_result") is not None


def unsaved_edits(state: Any = None) -> bool:
    state = _session() if state is None else state
    result = state.get("formation_result")
    if result is None:
        return False
    stats = getattr(result, "algorithm_stats", None) or {}
    if stats.get("source") != "edited":
        return False
    record = rh.active_run(state)
    if record is None:
        return True
    return fingerprint(projects_to_ids(state.get("projects") or [])) \
        != fingerprint(record.projects)


def clear_active_allocation(state: Any = None) -> None:
    state = _session() if state is None else state
    for key in ("projects", "formation_result", REPORTS_KEY, SCORED_POLICY_KEY,
                META_KEY, rh.ACTIVE_KEY):
        try:
            state.pop(key, None)
        except Exception:  # pragma: no cover - mapping without pop
            if key in state:
                del state[key]


def set_result_meta(meta: Dict[str, Any], state: Any = None) -> None:
    state = _session() if state is None else state
    state[META_KEY] = meta
