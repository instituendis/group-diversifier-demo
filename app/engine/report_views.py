from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional

from .runner import EngineResult


def _section(report: Any, key: str) -> Dict[str, Any]:
    value = report.get(key) if isinstance(report, Mapping) else None
    return value if isinstance(value, dict) else {}


def _pick(raw: Mapping[str, Any], cls: Any, skip: tuple = ()) -> Dict[str, Any]:
    wanted = {f.name for f in fields(cls)} - {"raw", *skip}
    return {k: v for k, v in raw.items() if k in wanted}


@dataclass(frozen=True)
class BaselineView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    mean: Optional[float] = None
    std: Optional[float] = None
    best: Optional[float] = None
    percentile_of_solution: Optional[float] = None
    samples: Optional[int] = None


def _baseline(raw: Any) -> BaselineView:
    raw = raw if isinstance(raw, dict) else {}
    return BaselineView(raw=raw, **_pick(raw, BaselineView))


@dataclass(frozen=True)
class OverallView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    objective: Optional[float] = None
    mean_team_penalty: Optional[float] = None
    worst_team_penalty: Optional[float] = None
    std_team_penalty: Optional[float] = None
    hard_units: Optional[int] = None
    avoidable_hard_units: Optional[int] = None
    unavoidable_hard_units: Optional[float] = None
    quality_index: Optional[float] = None
    quality_index_soft: Optional[float] = None
    soft_objective: Optional[float] = None
    n_teams: Optional[int] = None
    n_students: Optional[int] = None
    fairness_weight: Optional[float] = None
    outlier_team: Optional[Dict[str, Any]] = None
    random_baseline: BaselineView = field(default_factory=BaselineView)
    random_baseline_soft: BaselineView = field(default_factory=BaselineView)


@dataclass(frozen=True)
class PairView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    a: Optional[str] = None
    b: Optional[str] = None
    team: Optional[int] = None
    times_before: Optional[int] = None
    weight: Optional[float] = None
    met_in: Optional[List[str]] = None
    over_cap: Optional[bool] = None


def _pair(raw: Any) -> PairView:
    raw = raw if isinstance(raw, dict) else {}
    return PairView(raw=raw, **_pick(raw, PairView))


@dataclass(frozen=True)
class HistoryView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    available: Optional[bool] = None
    enabled: Optional[bool] = None
    cap: Optional[int] = None
    repeated_pairs: Optional[int] = None
    over_cap_pairs: Optional[int] = None
    max_times: Optional[int] = None
    contact_pairs: Optional[int] = None
    section_reunions: Optional[int] = None
    section_pairs: List[PairView] = field(default_factory=list)
    pairs: List[PairView] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.raw)


@dataclass(frozen=True)
class PreferencesView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    used: Optional[bool] = None
    prefer_satisfied: Optional[int] = None
    prefer_total: Optional[int] = None
    avoid_violated: Optional[int] = None
    avoid_total: Optional[int] = None
    violations: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class TeamView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    index: Optional[int] = None
    name: Optional[str] = None
    members: Optional[List[str]] = None
    size: Optional[int] = None
    penalty: Optional[float] = None
    hard_units: Optional[int] = None
    breakdown: Optional[Dict[str, float]] = None
    violations: Optional[List[str]] = None
    issues: Optional[List[Dict[str, Any]]] = None
    composition: Optional[Dict[str, Dict[str, int]]] = None
    numeric_means: Optional[Dict[str, Optional[float]]] = None


@dataclass(frozen=True)
class RuleView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    label: Optional[str] = None
    kind: Optional[str] = None
    hard: Optional[bool] = None
    weight: Optional[float] = None
    attribute: Optional[str] = None
    goal: Optional[str] = None
    teams_compliant: Optional[int] = None
    teams_total: Optional[int] = None
    compliance_pct: Optional[float] = None
    quality_pct: Optional[float] = None
    score_normalised: Optional[float] = None
    units: Optional[float] = None
    unavoidable_units: Optional[float] = None
    excess_sd: Optional[float] = None
    points: Optional[float] = None
    skipped: Optional[bool] = None


def overall(report: Mapping[str, Any]) -> OverallView:
    raw = _section(report, "overall")
    return OverallView(raw=raw, **_pick(raw, OverallView, skip=("random_baseline",
                                                                "random_baseline_soft")),
                       random_baseline=_baseline(raw.get("random_baseline")),
                       random_baseline_soft=_baseline(raw.get("random_baseline_soft")))


def history_section(report: Mapping[str, Any]) -> HistoryView:
    raw = _section(report, "history")
    pairs = raw.get("pairs")
    sections = raw.get("section_pairs")
    return HistoryView(
        raw=raw, **_pick(raw, HistoryView, skip=("pairs", "section_pairs")),
        pairs=[_pair(p) for p in pairs] if isinstance(pairs, list) else [],
        section_pairs=[_pair(p) for p in sections] if isinstance(sections, list) else [],
    )


def preferences_section(report: Mapping[str, Any]) -> PreferencesView:
    raw = _section(report, "preferences")
    return PreferencesView(raw=raw, **_pick(raw, PreferencesView))


def teams(report: Mapping[str, Any]) -> List[TeamView]:
    raw = report.get("teams") if isinstance(report, Mapping) else None
    items = raw if isinstance(raw, list) else []
    return [TeamView(raw=t, **_pick(t, TeamView)) if isinstance(t, dict) else TeamView()
            for t in items]


def team_report(report: Mapping[str, Any], team: Any) -> Optional[TeamView]:
    views = teams(report)
    if isinstance(team, bool):
        return None
    if isinstance(team, int):
        return views[team] if 0 <= team < len(views) else None
    if isinstance(team, str):
        match = re.fullmatch(r"team[_ ]?(\d+)", team.strip(), re.IGNORECASE)
        if match:
            index = int(match.group(1)) - 1
            return views[index] if 0 <= index < len(views) else None
        for view in views:
            if view.name == team:
                return view
    return None


def _rule_views(report: Mapping[str, Any], key: str) -> List[RuleView]:
    raw = report.get(key) if isinstance(report, Mapping) else None
    items = raw if isinstance(raw, list) else []
    return [RuleView(raw=r, **_pick(r, RuleView)) if isinstance(r, dict) else RuleView()
            for r in items]


def rule_entries(report: Mapping[str, Any]) -> List[RuleView]:
    return _rule_views(report, "rules")


def attribute_rules(report: Mapping[str, Any]) -> List[RuleView]:
    return _rule_views(report, "attribute_rules")


def warnings(report: Mapping[str, Any]) -> List[str]:
    raw = report.get("warnings") if isinstance(report, Mapping) else None
    return list(raw) if isinstance(raw, (list, tuple)) else []


@dataclass(frozen=True)
class MilpInfoView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    status: Optional[str] = None
    gap: Optional[float] = None
    elapsed_s: Optional[float] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class StatsView:
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    method: Optional[str] = None
    method_used: Optional[str] = None
    iterations: Optional[int] = None
    evaluations: Optional[int] = None
    accepted: Optional[int] = None
    improvements: Optional[int] = None
    restarts: Optional[int] = None
    restarts_planned: Optional[int] = None
    restarts_completed: Optional[int] = None
    best_restart: Optional[int] = None
    seeds: Optional[List[Any]] = None
    per_restart: Optional[List[float]] = None
    elapsed_s: Optional[float] = None
    time_limit_s: Optional[float] = None
    time_limited: Optional[bool] = None
    cancelled: Optional[bool] = None
    warm_start_ran: Optional[bool] = None
    warm_start_won: Optional[bool] = None
    workers: Optional[int] = None
    parallel: Optional[bool] = None
    milp: Optional[MilpInfoView] = None
    objective_gap: Optional[float] = None
    pair_overlap: Optional[Any] = None


def stats_view(stats: Optional[Mapping[str, Any]]) -> StatsView:
    raw = stats if isinstance(stats, dict) else {}
    milp = raw.get("milp")
    milp = milp if isinstance(milp, dict) else {}
    return StatsView(raw=raw, **_pick(raw, StatsView, skip=("milp",)),
                     milp=MilpInfoView(raw=milp, **_pick(milp, MilpInfoView)))


def run_summary(result: EngineResult, *, stopped: bool = False) -> Dict[str, Dict[str, Any]]:
    solver = (result.policy or {}).get("solver") or {}
    workers = int(solver.get("workers") or 0)
    out: Dict[str, Dict[str, Any]] = {}
    for pr in result.projects:
        stats = pr.stats or {}
        out[str(pr.project_id)] = {
            "score": pr.objective,
            "objective": pr.objective,
            "hard_units": pr.hard_units,
            "iterations": stats.get("iterations"),
            "restarts": stats.get("restarts"),
            "restarts_completed": stats.get("restarts_completed"),
            "restarts_planned": stats.get("restarts_planned"),
            "time_limited": bool(stats.get("time_limited", False)),
            "execution_time": stats.get("elapsed_s"),
            "converged": not (stopped or stats.get("time_limited", False)
                              or stats.get("cancelled", False)),
            "method_used": stats.get("method_used", result.method),
            "seeds": stats.get("seeds"),
            "actual_cores_used": workers,
        }
    return out
