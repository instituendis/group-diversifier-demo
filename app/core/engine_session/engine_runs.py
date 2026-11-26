from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ... import hosting
from ...engine.adapters import prior_from_legacy, teams_from_legacy
from ...engine.history import PriorProject
from ...engine.policy import Policy
from ...engine.runner import EngineResult, ProjectSpec, evaluate_allocation
from ...engine.sizing import project_config, resolve_sizes
from ..models import Configuration, Project, Student
from .roster_cache import current_policy, roster_frame
from .state import facade
from .state import session as _session


def imported_prior_projects(imported_assignments: Any = None) -> List[PriorProject]:
    imp = imported_assignments
    if imp is None:
        imp = _session().get("imported_assignments")
    if imp is None:
        return []
    prior = []
    for pid in imp.imported_project_ids:
        teams = imp.teams_by_project.get(pid, {})
        prior.append(PriorProject(project_id=str(pid), teams=[list(m) for m in teams.values()]))
    return prior


def project_specs(config: Configuration, roster_ids: Sequence[str], *,
                  participants: Any = None, imported_count: int = 0,
                  locks: Optional[Dict[str, Dict[str, int]]] = None,
                  name_teams: bool = False) -> List[ProjectSpec]:
    specs = []
    for idx in range(imported_count, int(config.num_projects)):
        pid = str(idx + 1)
        pool = participants.pool_for(pid) if participants is not None else None
        ids = list(pool) if pool is not None else list(roster_ids)
        eff = project_config(config, idx)
        size_min, size_max = int(eff.team_size_min), int(eff.team_size_max)
        size_target, num_teams = int(eff.team_size_target), int(eff.num_teams or 0)
        team_names = None
        if name_teams:
            planned = resolve_sizes(len(ids), size_min, size_max, size_target, num_teams)
            team_names = [f"team_{k + 1}" for k in range(len(planned))]
        specs.append(ProjectSpec(
            project_id=pid,
            student_ids=ids,
            size_min=size_min,
            size_max=size_max,
            size_target=size_target,
            num_teams=num_teams,
            locks=dict((locks or {}).get(pid, {})),
            team_names=team_names,
        ))
    return specs


def capped_policy(policy: Policy) -> Policy:
    workers = int(getattr(policy.solver, "workers", 1) or 1)
    clamped_workers = hosting.clamp_workers(workers)
    if hosting.max_time_limit() is None:
        clamped_limit = None
        scale_off = False
    else:
        requested = getattr(policy.solver, "time_limit_s", None)
        clamped_limit = hosting.clamp_time_limit(requested)
        try:
            unchanged = float(requested) == clamped_limit
        except (TypeError, ValueError):
            unchanged = False
        if unchanged:
            clamped_limit = None
        scale_off = bool(getattr(policy.solver, "scale_time_limit", False))
    if clamped_workers == workers and clamped_limit is None and not scale_off:
        return policy
    capped = copy.deepcopy(policy)
    capped.solver.workers = clamped_workers
    if clamped_limit is not None:
        capped.solver.time_limit_s = clamped_limit
    if scale_off:
        capped.solver.scale_time_limit = False
    return capped


def run_engine(config: Configuration, students: Sequence[Student], *,
               raw_df: Optional[pd.DataFrame] = None,
               imported_assignments: Any = None, participants: Any = None,
               policy: Optional[Policy] = None,
               locks: Optional[Dict[str, Dict[str, int]]] = None,
               frame: Optional[pd.DataFrame] = None,
               prior: Optional[Sequence[PriorProject]] = None,
               name_teams: bool = False,
               progress: Optional[Callable[[float, float, str], None]] = None,
               cancel: Optional[Callable[[], bool]] = None,
               apply_server_caps: bool = True,
               ) -> Tuple[EngineResult, List[PriorProject]]:
    frame = frame if frame is not None else roster_frame(students, raw_df)
    policy = policy or current_policy(config, frame)
    prior = list(prior) if prior is not None else imported_prior_projects(imported_assignments)
    specs = project_specs(config, frame["student_id"].tolist(), participants=participants,
                          imported_count=len(prior), locks=locks, name_teams=name_teams)
    run_policy = capped_policy(policy) if apply_server_caps else policy
    result = facade().run_allocation(frame, run_policy, specs, prior_projects=prior,
                                     progress=progress, cancel=cancel)
    return result, prior


def evaluate_projects(projects: Sequence[Project], config: Configuration,
                      students: Sequence[Student], *, policy: Optional[Policy] = None,
                      raw_df: Optional[pd.DataFrame] = None,
                      project_ids: Optional[Sequence[str]] = None,
                      frame: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
    frame = frame if frame is not None else roster_frame(students, raw_df)
    policy = policy or current_policy(config, frame)
    roster = set(frame["student_id"].tolist())
    reports: Dict[str, Dict[str, Any]] = {}
    for idx, project in enumerate(projects):
        if project.imported or (project_ids is not None and str(project.project_id) not in project_ids):
            continue
        teams = teams_from_legacy(project)
        pool = [sid for team in teams for sid in team if sid in roster]
        if not pool:
            continue
        eff = project_config(config, idx)
        spec = ProjectSpec(project_id=str(project.project_id), student_ids=pool,
                           team_sizes=[len([s for s in t if s in roster]) for t in teams],
                           size_min=int(eff.team_size_min), size_max=int(eff.team_size_max),
                           size_target=int(eff.team_size_target))
        prior = prior_from_legacy(projects[:idx])
        result = evaluate_allocation(frame, policy, spec, teams, prior_projects=prior)
        reports[str(project.project_id)] = result.report
    return reports
