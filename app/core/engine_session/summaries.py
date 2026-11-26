from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...engine.report_views import history_section, overall, preferences_section, stats_view, teams
from ..models import Project


def tagged_violations(projects: Sequence[Project],
                      reports: Dict[str, Dict[str, Any]]) -> List[Tuple[bool, str]]:
    names = {str(p.project_id): p for p in projects}
    hard, soft = [], []
    for pid, report in reports.items():
        project = names.get(pid)
        label = project.name if project else f"Project {pid}"
        team_ids = [t.team_id for t in project.teams] if project else []
        for team in teams(report):
            k = team.index or 0
            tid = team_ids[k] if k < len(team_ids) else team.name or f"team_{k + 1}"
            for issue in team.issues or []:
                text = f"{label} · {tid}: {issue.get('text', '')}"
                (hard if issue.get("hard") else soft).append(text)
    return [(True, text) for text in hard] + [(False, text) for text in soft]


def violations_from_reports(projects: Sequence[Project],
                            reports: Dict[str, Dict[str, Any]]) -> List[str]:
    return [text for _, text in tagged_violations(projects, reports)]


def summarize_reports(reports: Dict[str, Dict[str, Any]],
                      stats: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    overall_views = [overall(r) for r in reports.values()]
    history = [history_section(r) for r in reports.values()]
    prefs = [preferences_section(r) for r in reports.values()]
    given = {str(k): v for k, v in (stats or {}).items()}
    run_stats = [stats_view(given.get(str(pid)) or r.get("stats"))
                 for pid, r in reports.items()]

    def total(items, key):
        return float(sum(float(getattr(o, key) or 0.0) for o in items))

    def mean(items, key):
        vals = [float(getattr(o, key)) for o in items if getattr(o, key) is not None]
        return sum(vals) / len(vals) if vals else None

    worst = [float(o.worst_team_penalty or 0.0) for o in overall_views]
    summary: Dict[str, Any] = {
        "projects": len(reports),
        "teams": int(sum(int(o.n_teams or 0) for o in overall_views)),
        "objective": total(overall_views, "objective"),
        "soft_objective": total(overall_views, "soft_objective"),
        "hard_units": int(total(overall_views, "hard_units")),
        "worst_team_penalty": max(worst) if worst else 0.0,
        "quality_index_soft": mean(overall_views, "quality_index_soft"),
        "repeated_pairs": int(total(history, "repeated_pairs")),
        "over_cap_pairs": int(total(history, "over_cap_pairs")),
        "prefer_satisfied": int(total(prefs, "prefer_satisfied")),
        "prefer_total": int(total(prefs, "prefer_total")),
        "avoid_violated": int(total(prefs, "avoid_violated")),
        "violations": int(sum(len(t.violations or []) for r in reports.values()
                              for t in teams(r))),
        "section_reunions": int(total(history, "section_reunions")),
        "history_available": has_earlier_projects(reports),
    }
    for key in ("restarts_completed", "restarts_planned"):
        values = [int(getattr(s, key)) for s in run_stats if getattr(s, key) is not None]
        if values:
            summary[key] = int(sum(values))
    if any("time_limited" in s.raw for s in run_stats):
        summary["time_limited"] = any(bool(s.time_limited) for s in run_stats)
    return summary


def has_earlier_projects(reports: Dict[str, Dict[str, Any]]) -> bool:
    return any(bool(history_section(r).available) for r in reports.values())


def section_reunions(summary: Optional[Dict[str, Any]],
                     reports: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    if (summary or {}).get("section_reunions") is not None:
        return int(summary["section_reunions"])
    total = 0
    for report in (reports or {}).values():
        history = history_section(report)
        if history.section_reunions is not None:
            total += int(history.section_reunions)
        else:
            total += max(0, int(history.contact_pairs or 0)
                         - int(history.repeated_pairs or 0))
    return total


def show_section_reunions(summary: Optional[Dict[str, Any]],
                          reports: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
    if section_reunions(summary, reports):
        return True
    if (summary or {}).get("history_available") is not None:
        return bool(summary["history_available"])
    return has_earlier_projects(reports or {})
