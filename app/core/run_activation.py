from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ..engine.adapters import to_legacy_projects
from ..engine.policy import Policy
from ..engine.report_views import run_summary
from ..utils.serialization import json_safe
from . import engine_session as es
from . import run_history as rh
from .assignment_import import _MAX_IMPORTED_GROUP, build_imported_projects
from .edit_scoring import roster_signature
from .ids import IdMatcher
from .engine_session import (active_allocation, activate_allocation, evaluate_projects,
                             projects_from_ids, projects_to_ids, report_matches_teams,
                             set_result_meta, summarize_reports, unmatched_ids)
from .models import Configuration, Project, Student

if TYPE_CHECKING:
    from ..engine.runner import EngineResult


class ActivationError(ValueError): ...


MAX_STORED_TEAM_SIZE = _MAX_IMPORTED_GROUP


def oversized_team(projects_plain: Any) -> Optional[Tuple[str, int, int]]:
    if not isinstance(projects_plain, (list, tuple)):
        return None
    for entry in projects_plain:
        teams = entry.get("teams") if isinstance(entry, dict) else None
        if not isinstance(teams, (list, tuple)):
            continue
        for t, team in enumerate(teams, start=1):
            if isinstance(team, (list, tuple)) and len(team) > MAX_STORED_TEAM_SIZE:
                return str(entry.get("project_id")), t, len(team)
    return None


def oversized_team_message(found: Tuple[str, int, int], whose: str) -> str:
    pid, position, size = found
    return (f"Team {position} of project {pid} in {whose} has {size} students; this app "
            f"accepts teams of at most {MAX_STORED_TEAM_SIZE} students.")


def _session():
    return es._session()


def roster_identity(students: Sequence[Student], raw_df: Any = None) -> str:
    return roster_signature(es.roster_frame(students, raw_df))


def record_run(*, label: str, source: str, run_config: Configuration, policy: Any,
               projects: Sequence[Project], reports: Dict[str, Dict[str, Any]],
               extra_summary: Dict[str, Any], parent: Optional[str] = None,
               activate: bool = True, notes: str = "",
               state: Any = None, roster: str = "") -> rh.RunRecord:
    state = _session() if state is None else state
    summary = {**(summarize_reports(reports) if reports else {}), **extra_summary}
    record = rh.make_record(
        label=label, source=source, config=run_config.to_dict(),
        policy=policy.to_dict(), projects=projects_to_ids(projects),
        reports=json_safe(reports), summary=json_safe(summary), notes=notes,
        parent_run_id=parent, roster_signature=roster,
    )
    return rh.add_run(record, activate=activate, state=state)


def engine_run_meta(result: EngineResult, per_project: Dict[str, Dict[str, Any]],
                    policy: Policy, *, stopped: bool = False) -> Dict[str, Any]:
    seeds = [s for entry in per_project.values() for s in (entry["seeds"] or [])]
    warnings = list(result.warnings)
    if stopped:
        warnings.insert(0, "Stopped early on request: these teams are the best allocation "
                           "found until then (recorded as a run marked 'stopped early').")
    return {
        "method": result.method,
        "elapsed_s": result.elapsed_s,
        "warnings": warnings,
        "seed": policy.solver.seed,
        "seeds": seeds[:5],
        "policy_name": policy.name,
        "source": "engine",
        "stopped_early": stopped,
    }


def _run_extra_summary(per_project: Dict[str, Dict[str, Any]], policy: Policy, *,
                       method: str, elapsed_s: Any,
                       stopped: bool) -> Dict[str, Any]:
    extra = {"method": method, "elapsed_s": elapsed_s, "seed": policy.solver.seed,
             "algorithm": "engine",
             "time_limited": any(bool(entry.get("time_limited"))
                                 for entry in per_project.values())}
    for key in ("restarts_completed", "restarts_planned"):
        values = [int(entry[key]) for entry in per_project.values()
                  if entry.get(key) is not None]
        if values:
            extra[key] = sum(values)
    if stopped:
        extra["stopped_early"] = True
    return extra


def finish_engine_run(result: EngineResult, *, config: Configuration, policy: Policy,
                      students: Sequence[Student], imported: Any = None,
                      n_generated: int = 0, stopped: bool = False,
                      state: Any = None, raw_df: Any = None) -> Tuple[rh.RunRecord, Dict[str, Any]]:
    state = _session() if state is None else state
    students_by_id = {s.student_id: s for s in students}
    projects: List[Project] = []
    if imported:
        projects.extend(build_imported_projects(imported, students_by_id))
    projects.extend(to_legacy_projects(result, students))
    reports = {str(pr.project_id): pr.report for pr in result.projects}
    per_project = {pid: json_safe(entry)
                   for pid, entry in run_summary(result, stopped=stopped).items()}
    meta = engine_run_meta(result, per_project, policy, stopped=stopped)
    record = record_run(
        label=f"{policy.name} · {result.method}" + (" · stopped early" if stopped else ""),
        source="engine", run_config=config, policy=policy, projects=projects,
        reports=reports,
        extra_summary=_run_extra_summary(per_project, policy, method=result.method,
                                         elapsed_s=result.elapsed_s, stopped=stopped),
        state=state, roster=roster_identity(students, raw_df),
    )
    meta["run_id"] = record.run_id
    activate_allocation(
        projects, config, students, reports,
        execution_time=float(result.elapsed_s),
        algorithm_stats={"engine": json_safe(meta), "per_project": per_project},
        run_id=record.run_id, scored_with=policy, state=state,
    )
    set_result_meta(meta, state)
    summary = {
        "total_time": float(result.elapsed_s),
        "teams_formed": sum(len(p.teams) for p in projects if not p.imported),
        "n_projects": len(projects),
        "n_generated": int(n_generated),
        "algorithm": "engine",
        "num_cores": int(policy.solver.workers),
        "mp_failed": False,
        "per_project": per_project,
        "run_label": rh.run_title(record),
        "run_id": record.run_id,
        "stopped_early": stopped,
    }
    return record, summary


def record_rescored_allocation(new_reports: Dict[str, Dict[str, Any]],
                               students: Sequence[Student],
                               *, state: Any = None, raw_df: Any = None) -> rh.RunRecord:
    state = _session() if state is None else state
    alloc = active_allocation(state)
    if alloc is None:
        raise RuntimeError("record_rescored_allocation without an active allocation")
    result = alloc.formation_result
    projects = alloc.projects
    config = state["config"]
    parent = rh.active_run(state)
    plain = projects_to_ids(projects)
    base = rh.display_label(parent.label) if parent is not None else "Allocation"
    edited = parent is not None and alloc.unsaved_edits
    if edited:
        base = re.sub(r" \(edited\)$", "", base)
        label, source = f"{base} (edited, re-scored with the current rules)", "edited"
        notes = (f"The teams of {parent.run_id} with unsaved Team Editor edits, scored with "
                 "the rules of the Allocation Rules page.")
    else:
        label, source = f"{base} (re-scored with the current rules)", "evaluated"
        notes = (f"The teams of {parent.run_id}, scored with the rules of the Allocation Rules "
                 "page." if parent is not None else "Scored with the current rules.")
    summary = {**summarize_reports(new_reports), "method": "re-scored"}
    record = rh.make_record(
        label=label, source=source,
        config=result.config.to_dict(), policy=copy.deepcopy(config.policy or {}),
        projects=plain, reports=new_reports, summary=summary, notes=notes,
        parent_run_id=parent.run_id if parent is not None else None,
        roster_signature=roster_identity(students, raw_df))
    stored = rh.add_run(record, activate=True, state=state)
    stats = {k: v for k, v in (result.algorithm_stats or {}).items()
             if k not in ("engine_summary", "unsaved")}
    stats["run_id"] = stored.run_id
    activate_allocation(projects, result.config, students, new_reports,
                        execution_time=float(result.execution_time or 0.0),
                        algorithm_stats=stats, run_id=stored.run_id,
                        scored_with=stored.policy, state=state)
    return stored


def _edited_stats(*, base_run_id: Optional[str] = None, edits: Optional[int] = None,
                  unsaved: Optional[bool] = None, run_id: Optional[str] = None,
                  parent_run_id: Optional[str] = None) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"source": "edited", "method": "edited"}
    for key, value in (("run_id", run_id), ("parent_run_id", parent_run_id),
                       ("base_run_id", base_run_id), ("edits", edits),
                       ("unsaved", unsaved)):
        if value is not None:
            stats[key] = value
    return stats


def activate_edited_allocation(projects: Sequence[Project], config: Configuration,
                               students: Sequence[Student],
                               reports: Dict[str, Dict[str, Any]], *, policy: Any,
                               base_run_id: Optional[str] = None,
                               edits: Optional[int] = None, unsaved: Optional[bool] = None,
                               run_id: Optional[str] = None,
                               parent_run_id: Optional[str] = None,
                               state: Any = None) -> Any:
    state = _session() if state is None else state
    return activate_allocation(
        list(projects), config, students, reports,
        algorithm_stats=_edited_stats(base_run_id=base_run_id, edits=edits, unsaved=unsaved,
                                      run_id=run_id, parent_run_id=parent_run_id),
        run_id=run_id, scored_with=policy, state=state,
    )


def save_edited_run(record: rh.RunRecord, *, config: Configuration,
                    students: Sequence[Student], policy: Any,
                    state: Any = None) -> Tuple[rh.RunRecord, List[Project]]:
    state = _session() if state is None else state
    stored = rh.add_run(record, activate=True, state=state)
    legacy = projects_from_ids(record.projects, students)
    activate_edited_allocation(legacy, config, students, record.reports, policy=policy,
                               run_id=stored.run_id, parent_run_id=record.parent_run_id,
                               state=state)
    return stored, legacy


@dataclass(frozen=True)
class StoredActivation:
    warnings: List[str] = field(default_factory=list)
    reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rescored_missing: List[str] = field(default_factory=list)
    rescored_changed: List[str] = field(default_factory=list)
    activated: bool = True


def _one_team_per_project(projects_plain: Sequence[Dict[str, Any]], roster_ids: List[str],
                          warnings: List[str], *, tolerant: bool) -> List[Dict[str, Any]]:
    matcher = IdMatcher(roster_ids)
    result: List[Dict[str, Any]] = []
    for entry in projects_plain:
        pid = str(entry.get("project_id"))
        seen: set = set()
        teams: List[List[Any]] = []
        for team in entry.get("teams", []):
            kept: List[Any] = []
            for sid in team:
                match = matcher.resolve(sid)
                if match is None:
                    kept.append(sid)
                    continue
                if match in seen:
                    warnings.append(f"Student {match} was listed in two teams of project {pid} "
                                    "and only the first was kept" + ("." if tolerant else ""))
                    continue
                seen.add(match)
                kept.append(sid)
            teams.append(kept)
        result.append({**entry, "teams": teams})
    return result


def activate_stored_projects(projects_plain: Sequence[Dict[str, Any]],
                             *, config: Configuration, students: Sequence[Student],
                             raw_df: Any = None, frame: Any = None, state: Any = None,
                             reports: Optional[Dict[str, Dict[str, Any]]] = None,
                             scored: Any = None, fallback_policy: Any = None,
                             run_id: Optional[str] = None,
                             stats: Optional[Dict[str, Any]] = None,
                             execution_time: float = 0.0, whose: str = "this run",
                             roster_hint: str = "the run was formed with before activating it.",
                             tolerant: bool = False) -> StoredActivation:
    state = _session() if state is None else state
    found = oversized_team(projects_plain)
    if found is not None:
        raise ActivationError(oversized_team_message(found, whose))
    warnings: List[str] = []
    roster_ids =[str(s.student_id) for s in students]
    placed = {str(s) for entry in projects_plain for team in entry.get("teams", [])
              for s in team}
    missing_ids = unmatched_ids(projects_plain, roster_ids)
    if placed and len(missing_ids) * 2 > len(placed):
        raise ActivationError(f"Only {len(placed) - len(missing_ids)} of the {len(placed)} "
                              f"students of {whose} are in the loaded roster. Load the "
                              f"roster {roster_hint}")
    if not placed:
        warnings.append(f"No student of {whose} is in the loaded roster, so no allocation "
                        "was activated.")
        return StoredActivation(warnings=warnings, activated=False)
    if missing_ids:
        warnings.append(f"{len(missing_ids)} student(s) of {whose} are not in the loaded "
                        f"roster and were left out (e.g. {', '.join(missing_ids[:5])})"
                        + ("." if tolerant else ""))
    projects_plain = _one_team_per_project(projects_plain, roster_ids, warnings,
                                           tolerant=tolerant)
    projects = projects_from_ids(projects_plain, students)
    by_id = {str(p.project_id): p for p in projects}
    generated = [str(p.project_id) for p in projects if not p.imported]
    given = {str(pid): rep for pid, rep in (reports or {}).items() if pid in by_id}
    kept = {pid: rep for pid, rep in given.items()
            if pid not in generated or report_matches_teams(by_id[pid], rep)}
    rescored_missing = [pid for pid in generated if pid not in given]
    rescored_changed = [pid for pid in generated if pid in given and pid not in kept]
    stale = rescored_missing + rescored_changed
    final: Dict[str, Dict[str, Any]] = dict(kept)
    scored_with = scored if isinstance(scored, dict) and scored else None
    if stale:
        rules: Optional[Policy] = None
        if scored_with:
            try:
                rules = Policy.from_dict(copy.deepcopy(scored_with))
            except Exception:
                if not tolerant:
                    raise
                rules = None
        if rules is None and scored_with is None and fallback_policy:
            rules = (fallback_policy if isinstance(fallback_policy, Policy)
                     else Policy.from_dict(fallback_policy))
        if rules is None:
            frame = frame if frame is not None else es.roster_frame(students, raw_df)
            rules = es.current_policy(config, frame)
            scored_with = rules.to_dict()
        if tolerant:
            try:
                fresh = evaluate_projects(projects, config, students, policy=rules,
                                          raw_df=raw_df, project_ids=stale, frame=frame)
            except ValueError as exc:
                warnings.append(f"The restored allocation could not be scored: {exc}")
                fresh = {}
        else:
            fresh = evaluate_projects(projects, config, students, policy=rules,
                                      raw_df=raw_df, project_ids=stale, frame=frame)
        final.update(fresh)
        if not tolerant:
            if rescored_missing:
                warnings.append(f"Scores of project(s) {', '.join(rescored_missing)} "
                                "were re-evaluated")
            if rescored_changed:
                warnings.append(f"Project(s) {', '.join(rescored_changed)} were scored "
                                "again for the loaded roster")
    activate_allocation(projects, config, students, final,
                        execution_time=float(execution_time),
                        algorithm_stats=dict(stats or {}), run_id=run_id,
                        scored_with=scored_with, state=state)
    return StoredActivation(warnings=warnings, reports=final,
                            rescored_missing=rescored_missing,
                            rescored_changed=rescored_changed)
