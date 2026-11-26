from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ...engine.policy import Policy
from .. import import_sync
from ..ids import IdMatcher
from ..models import Configuration, Project, Student
from .allocation_session import activate_allocation
from .engine_runs import evaluate_projects
from .plain_form import _plain_teams, projects_from_ids, projects_to_ids
from .roster_cache import current_policy, roster_frame


def departed_students(projects: Sequence[Any], roster_ids: Sequence[Any]) -> Dict[str, List[str]]:
    matcher = IdMatcher(roster_ids)
    out: Dict[str, List[str]] = {}
    for project in projects:
        if import_sync.is_imported(project):
            continue
        pid, teams = _plain_teams(project)
        gone = list(dict.fromkeys(sid for team in teams for sid in team
                                  if matcher.resolve(sid) is None))
        if gone:
            out[pid] = gone
    return out


def formed_on_other_dataset(projects: Sequence[Any], roster_ids: Sequence[Any]) -> bool:
    matcher = IdMatcher(roster_ids)
    formed: List[str] = []
    for project in projects:
        if import_sync.is_imported(project):
            continue
        _, teams = _plain_teams(project)
        members = list(dict.fromkeys(sid for team in teams for sid in team))
        if members and all(matcher.resolve(sid) is None for sid in members):
            return True
        formed.extend(members)
    formed = list(dict.fromkeys(formed))
    off = sum(1 for sid in formed if matcher.resolve(sid) is None)
    return bool(formed) and off * 2 > len(formed)


OTHER_DATASET_TEXT = ("These results were formed on a different dataset than the one currently "
                      "loaded")


def roster_notice(projects: Sequence[Any],
                  roster_ids: Sequence[Any]) -> Optional[Tuple[str, str]]:
    departed = departed_students(projects, roster_ids)
    if not departed:
        return None
    if formed_on_other_dataset(projects, roster_ids):
        formed = {sid for project in projects if not import_sync.is_imported(project)
                  for team in _plain_teams(project)[1] for sid in team}
        missing = len({sid for ids in departed.values() for sid in ids})
        return "dataset", (f"{OTHER_DATASET_TEXT}: {missing} of the {len(formed)} students in "
                           "the generated projects are not in it. Load the dataset the teams "
                           "were formed with, or form new teams for this one.")
    names = {}
    for project in projects:
        pid, _ = _plain_teams(project)
        name = project.name if isinstance(project, Project) else (
            project.get("name") if isinstance(project, dict) else None)
        names[pid] = name or f"Project {pid}"
    where = "; ".join(f"{len(ids)} departed student(s) still in {names.get(pid, pid)}"
                      for pid, ids in departed.items())
    return "departed", f"{where} — remove them or repair in Team Editor › Roster changes."


def departed_notice(projects: Sequence[Any], roster_ids: Sequence[Any]) -> Optional[str]:
    notice = roster_notice(projects, roster_ids)
    return notice[1] if notice else None


def refresh_imported_projects(state: Any, students: Sequence[Student], *,
                              config: Optional[Configuration] = None,
                              raw_df: Optional[pd.DataFrame] = None,
                              policy: Optional[Policy] = None,
                              frame: Optional[pd.DataFrame] = None) -> List[str]:
    projects = list(state.get("projects") or [])
    config = config if config is not None else state.get("config")
    if not projects or config is None:
        return []
    fresh = import_sync.refreshed_imports(projects_to_ids(projects),
                                          state.get("imported_assignments"), students)
    if not fresh:
        return []
    by_id = {str(s.student_id): s for s in students}
    matcher = IdMatcher(by_id)
    projects = [projects_from_ids([fresh[str(p.project_id)]], students)[0]
                if p.imported and str(p.project_id) in fresh
                else import_sync.project_on_roster(p, by_id, matcher)
                for p in projects]
    frame = frame if frame is not None else roster_frame(students, raw_df)
    policy = policy if policy is not None else current_policy(config, frame)
    reports = evaluate_projects(projects, config, students, policy=policy, raw_df=raw_df,
                                frame=frame)
    previous = state.get("formation_result")
    stats = dict(getattr(previous, "algorithm_stats", None) or {})
    edited = stats.get("source") == "edited"
    activate_allocation(projects, config, students, reports,
                        execution_time=float(getattr(previous, "execution_time", 0.0) or 0.0),
                        algorithm_stats={
                            "source": "edited", "method": "edited",
                            "base_run_id": (stats.get("base_run_id") if edited
                                            else state.get("active_run_id")),
                            "edits": int(stats.get("edits") or 0) if edited else 0,
                            "unsaved": True,
                        }, state=state, scored_with=policy)
    return list(fresh)
