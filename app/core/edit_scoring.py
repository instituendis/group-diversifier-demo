from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..engine import repair as engine_repair
from ..engine.explain import suggest_moves
from ..engine.history import PriorProject, build_pair_history
from ..engine.metrics import build_report, team_violations
from ..engine.objective import ObjectiveModel
from ..engine.policy import Policy
from ..engine.report_views import history_section, overall
from ..engine.roster import infer_attributes
from ..engine.runner import ProjectSpec, encode_roster, run_allocation
from . import engine_session as es
from .edit_session import (
    EditError,
    EditSession,
    Op,
    describe_op,
    normalise_projects,
    parse_op,
)
from .engine_session import as_policy, digest
from .ids import IdMatcher
from .job_runner import CancelFn, ProgressFn, SlotWaitCancelled, run_slot

_EPS = 1e-9


def _slot_waiter(progress: Optional[ProgressFn]):
    if progress is None:
        return None
    return lambda text: progress(0.0, math.nan, text)


class RepairUnavailable(EditError): ...


def roster_signature(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame.astype(str), index=False).to_numpy()
    return digest([list(map(str, frame.columns)), hashlib.sha1(values.tobytes()).hexdigest()])


@dataclass
class ProjectModel:
    project_id: str
    model: ObjectiveModel
    teams: List[List[str]]
    team_names: List[str]
    index: Dict[str, int]
    pool_ids: List[str]
    members: List[List[int]]
    assignment: List[int]
    penalties: List[Optional[float]]
    hard_exact: List[float]
    objective: float
    hard_units: int
    unknown: List[str]
    prior: List[PriorProject]
    upcoming_index: int = 0
    roster_id: Dict[str, str] = field(default_factory=dict)

    def session_id(self, roster_id: str) -> str:
        if not hasattr(self, "_by_roster"):
            self._by_roster = {rid: sid for sid, rid in self.roster_id.items()}
        return self._by_roster.get(roster_id, roster_id)

    def worst_team(self) -> Optional[int]:
        scored = [(p, t) for t, p in enumerate(self.penalties) if p is not None]
        return max(scored)[1] if scored else None


def _aggregate(model: ObjectiveModel, penalties: Sequence[Optional[float]]) -> float:
    return model.aggregate([p for p in penalties if p is not None])


def _matcher(roster: Any) -> IdMatcher:
    return roster if isinstance(roster, IdMatcher) else IdMatcher(roster)


def _rostered_projects(projects: Sequence[Dict[str, Any]], matcher: IdMatcher
                       ) -> List[Dict[str, Any]]:
    out = []
    for entry in projects:
        teams = [[m for m in (matcher.resolve(s) for s in team) if m is not None]
                 for team in entry["teams"]]
        out.append({**entry, "teams": teams})
    return out


def history_for(session: EditSession, project_id: str, roster: Any) -> List[PriorProject]:
    k = session.project_index(project_id)
    return es.other_projects_history(_rostered_projects(session.projects, _matcher(roster)), k)


def history_position(session: EditSession, project_id: str, roster: Any) -> int:
    k = session.project_index(project_id)
    return es.history_position(_rostered_projects(session.projects, _matcher(roster)), k)


def project_model(session: EditSession, frame: pd.DataFrame, policy: Any,
                  project_id: Optional[str] = None) -> ProjectModel:
    policy = as_policy(policy)
    pid = session._pid(project_id)
    entry = session.project(pid)
    problems = policy.validate({n: i.kind for n, i in infer_attributes(frame).items()})
    if problems:
        raise EditError("The allocation rules are invalid: " + "; ".join(problems))
    matcher = IdMatcher(frame["student_id"].astype(str).tolist())
    teams = entry["teams"]
    roster_id = {sid: rid for team in teams for sid in team
                 if (rid := matcher.resolve(sid)) is not None}
    pool_ids = [sid for team in teams for sid in team if sid in roster_id]
    if not pool_ids:
        raise EditError(f"No student of {entry['name']} is in the loaded roster")
    unknown = [sid for team in teams for sid in team if sid not in roster_id]
    cohort, _ = encode_roster(frame, policy)
    pool = cohort.subset([roster_id[sid] for sid in pool_ids])
    index = {sid: k for k, sid in enumerate(pool_ids)}
    members = [[index[s] for s in team if s in index] for team in teams]
    assignment = [0] * len(pool_ids)
    for t, ms in enumerate(members):
        for i in ms:
            assignment[i] = t
    prior = history_for(session, pid, matcher)
    upcoming = history_position(session, pid, matcher)
    history = build_pair_history(prior, policy.history, pool, upcoming_index=upcoming)
    model = ObjectiveModel(pool, policy, [max(1, len(m)) for m in members], history=history,
                           project_id=pid)
    penalties = [model.team_penalty(m) if m else None for m in members]
    hard_exact = [model.team_hard_units_exact(m) if m else 0.0 for m in members]
    return ProjectModel(
        project_id=pid, model=model, teams=[list(t) for t in teams],
        team_names=session.display_names(pid), index=index, pool_ids=pool_ids,
        members=members, assignment=assignment, penalties=penalties, hard_exact=hard_exact,
        objective=_aggregate(model, penalties), hard_units=int(round(sum(hard_exact))),
        unknown=unknown, prior=prior, upcoming_index=upcoming, roster_id=roster_id,
    )


def team_cards(pm: ProjectModel) -> List[Dict[str, Any]]:
    ids = pm.pool_ids
    cards = []
    for t, team in enumerate(pm.teams):
        ms = pm.members[t]
        issues = [{**v, "students": [ids[i] for i in v["students"]]}
                  for v in team_violations(pm.model, ms)] if ms else []
        cards.append({
            "index": t,
            "name": pm.team_names[t],
            "members": list(team),
            "size": len(team),
            "penalty": pm.penalties[t],
            "hard_units": pm.model.hard_units(ms) if ms else 0,
            "issues": issues,
            "breakdown": pm.model.team_breakdown(ms) if ms else {},
        })
    return cards


_CHIP_KINDS = ("categorical", "numeric", "multi")
_CHIP_MAX_VALUES = 12


def chip_attributes(frame: pd.DataFrame) -> List[str]:
    return [name for name, info in infer_attributes(frame).items() if info.kind in _CHIP_KINDS]


def default_chip_attributes(frame: pd.DataFrame, policy: Any, k: int = 3) -> List[str]:
    infos = infer_attributes(frame)

    def usable(name: str) -> bool:
        info = infos.get(name)
        return info is not None and info.kind in _CHIP_KINDS and (
            info.kind == "numeric" or info.n_unique <= _CHIP_MAX_VALUES)

    rules = sorted((r for r in as_policy(policy).attributes if r.enabled),
                   key=lambda r: -float(r.weight))
    chosen: List[str] = []
    for name in [r.attribute for r in rules] + list(infos):
        if name not in chosen and usable(name):
            chosen.append(name)
        if len(chosen) >= k:
            break
    return chosen


def _chip_text(value: Any, max_len: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, (list, tuple, set)):
        text = ", ".join(str(v) for v in value) or "–"
    elif isinstance(value, float):
        text = str(int(value)) if value.is_integer() else f"{value:.1f}"
    else:
        text = str(value).strip() or "–"
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def chip_values(frame: pd.DataFrame, attributes: Sequence[str],
                max_len: int = 18) -> Dict[str, List[str]]:
    columns = [a for a in attributes if a in frame.columns]
    out: Dict[str, List[str]] = {}
    for row in frame[["student_id"] + columns].itertuples(index=False, name=None):
        out[str(row[0])] = [_chip_text(v, max_len) for v in row[1:]]
    return out


def run_wide_totals(projects: Sequence[Dict[str, Any]], frame: pd.DataFrame, policy: Any,
                    *, start: int = 0,
                    cache: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict[str, int]:
    policy = as_policy(policy)
    cohort, _ = encode_roster(frame, policy)
    matcher = IdMatcher(cohort.student_ids)
    plain = _rostered_projects(normalise_projects(projects), matcher)
    cache = {} if cache is None else cache
    hard = repeats = 0
    for k, entry in enumerate(plain):
        if entry["imported"]:
            continue
        if k < start and k in cache:
            h, r = cache[k]
        else:
            h, r = _project_totals(cohort, policy, entry, plain[:k])
            cache[k] = (h, r)
        hard += h
        repeats += r
    return {"hard_units": int(hard), "repeated_pairs": int(repeats)}


def _project_totals(cohort: Any, policy: Policy, entry: Dict[str, Any],
                    before: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    teams = [team for team in entry["teams"] if team]
    pool_ids = [sid for team in teams for sid in team]
    if not pool_ids:
        return 0, 0
    pool = cohort.subset(pool_ids)
    prior = [PriorProject(project_id=p["project_id"], teams=[t for t in p["teams"] if t])
             for p in before]
    history = build_pair_history(prior, policy.history, pool)
    model = ObjectiveModel(pool, policy, [len(t) for t in teams], history=history,
                           project_id=entry["project_id"])
    assignment = [t for t, team in enumerate(teams) for _ in team]
    report = build_report(model, assignment, baseline_samples=0)
    return (int(overall(report).hard_units or 0),
            int(history_section(report).repeated_pairs or 0))


def run_wide_delta(before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]],
                   frame: pd.DataFrame, policy: Any) -> Dict[str, int]:
    first = next((k for k, (a, b) in enumerate(zip(before, after))
                  if [t for t in a["teams"] if t] != [t for t in b["teams"] if t]),
                 min(len(before), len(after)))
    cache: Dict[int, Tuple[int, int]] = {}
    old = run_wide_totals(before, frame, policy, cache=cache)
    new = run_wide_totals(after, frame, policy, start=first, cache=cache)
    return {"hard_units_before": old["hard_units"], "hard_units_after": new["hard_units"],
            "repeated_pairs_before": old["repeated_pairs"],
            "repeated_pairs_after": new["repeated_pairs"]}


def run_wide_warning(run_wide: Optional[Dict[str, int]]) -> Optional[str]:
    if not run_wide:
        return None
    gained = run_wide["hard_units_after"] - run_wide["hard_units_before"]
    if gained <= 0:
        return None
    return (f"All generated projects together: hard-rule violations would rise from "
            f"{run_wide['hard_units_before']} to {run_wide['hard_units_after']} "
            f"(repeat pairs {run_wide['repeated_pairs_before']} → "
            f"{run_wide['repeated_pairs_after']}); later projects are affected too")


def _reason(name: str, term: str, change: float) -> str:
    sign = "−" if change < 0 else "+"
    return f"{name} · {term} {sign}{abs(change):.2f}"


def _violation_texts(model: ObjectiveModel, members: List[int]) -> List[Tuple[str, bool]]:
    return [(v["text"], bool(v["hard"])) for v in team_violations(model, members)] if members \
        else []


def preview(session: EditSession, frame: pd.DataFrame, policy: Any, op: Op, *,
            project_id: Optional[str] = None, size_bounds: Optional[Tuple[int, int]] = None,
            pm: Optional[ProjectModel] = None, run_wide: bool = True) -> Dict[str, Any]:
    pid = session._pid(project_id)
    kind, students, target = parse_op(op)
    if kind == "swap":
        sa, sb, ta, tb = session.check_swap(students[0], students[1], pid)
        moves = [(sa, ta, tb), (sb, tb, ta)]
    else:
        sid, source, target = session.check_move(students[0], target, pid)
        moves = [(sid, source, target)]
    pm = pm if pm is not None and pm.project_id == pid else project_model(
        session, frame, policy, pid)
    model = pm.model
    new_members = {t: list(pm.members[t]) for _, a, b in moves for t in (a, b)}
    for sid, source, dest in moves:
        if sid not in pm.index:
            raise EditError(f"Student {sid} is not in the loaded roster; repair the "
                            f"allocation first")
        i = pm.index[sid]
        new_members[source].remove(i)
        new_members[dest].append(i)

    penalties = list(pm.penalties)
    hard = list(pm.hard_exact)
    team_rows, term_changes = [], []
    added, resolved = [], []
    for t, ms in sorted(new_members.items()):
        before_bd = model.team_breakdown(pm.members[t]) if pm.members[t] else {}
        after_bd = model.team_breakdown(ms) if ms else {}
        penalties[t] = model.team_penalty(ms) if ms else None
        hard[t] = model.team_hard_units_exact(ms) if ms else 0.0
        name = pm.team_names[t]
        for label in after_bd.keys() | before_bd.keys():
            b, a = before_bd.get(label, 0.0), after_bd.get(label, 0.0)
            if abs(a - b) > 1e-6:
                term_changes.append({"team": t, "name": name, "term": label, "before": b,
                                     "after": a, "change": a - b})
        old_v = _violation_texts(model, pm.members[t])
        new_v = _violation_texts(model, ms)
        added += [(f"{name}: {txt}", h) for txt, h in new_v if (txt, h) not in old_v]
        resolved += [(f"{name}: {txt}", h) for txt, h in old_v if (txt, h) not in new_v]
        pb, pa = pm.penalties[t], penalties[t]
        team_rows.append({"team": t, "name": name, "penalty_before": pb, "penalty_after": pa,
                          "change": (pa or 0.0) - (pb or 0.0),
                          "size_before": len(pm.teams[t]),
                          "size_after": len(pm.teams[t]) + len(ms) - len(pm.members[t])})
    objective_after = _aggregate(model, penalties)
    term_changes.sort(key=lambda c: -abs(c["change"]))
    warnings = []
    if size_bounds is not None and kind == "move":
        lo, hi = size_bounds
        for row in team_rows:
            if row["size_after"] and not lo <= row["size_after"] <= hi:
                warnings.append(f"{row['name']} would have {row['size_after']} members "
                                f"(allowed {lo}–{hi})")
    for row in team_rows:
        if row["size_before"] and not row["size_after"]:
            warnings.append(f"{row['name']} would be empty")
    wide = None
    if run_wide:
        after = copy.deepcopy(session.projects)
        teams_after = next(e for e in after if e["project_id"] == pid)["teams"]
        for sid, source, dest in moves:
            teams_after[source].remove(sid)
            teams_after[dest].append(sid)
        wide = run_wide_delta(session.projects, after, frame, policy)
        warning = run_wide_warning(wide)
        if warning:
            warnings.append(warning)
    sources = [session.team_of(s, pid) for s in students]
    return {
        "kind": kind,
        "students": students,
        "description": describe_op(op if kind == "swap" else {**op, "to_team": target},
                                   pm.team_names, sources),
        "objective_before": pm.objective,
        "objective_after": objective_after,
        "delta": objective_after - pm.objective,
        "hard_before": pm.hard_units,
        "hard_after": int(round(sum(hard))),
        "teams": team_rows,
        "term_changes": term_changes,
        "reasons": [_reason(c["name"], c["term"], c["change"]) for c in term_changes[:6]],
        "violations_added": [{"text": txt, "hard": h} for txt, h in added],
        "violations_resolved": [{"text": txt, "hard": h} for txt, h in resolved],
        "warnings": warnings,
        "run_wide": wide,
    }


def suggestions(session: EditSession, frame: pd.DataFrame, policy: Any, *, k: int = 10,
                focus_student: Optional[str] = None, focus_team: Optional[int] = None,
                allow_moves: bool = False, size_bounds: Optional[Tuple[int, int]] = None,
                project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    pid = session._require_editable(project_id)
    pm = project_model(session, frame, policy, pid)
    locked_teams = set(session.locked_team_indices(pid))
    locks = {pm.index[s]: t for s, t in session.effective_locks(pid).items() if s in pm.index}
    focus = None
    if focus_student is not None:
        sid = str(focus_student).strip()
        if sid not in pm.index:
            raise EditError(f"Student {sid} is not a rostered member of this project")
        if sid in session.effective_locks(pid):
            raise EditError(f"Student {sid} is locked; unlock them to get suggestions")
        focus = [pm.index[sid]]
    if focus_team is not None:
        session._team_index(focus_team, pid)
    has_empty = any(not m for m in pm.members)
    want = int(k) * 4 + 10 if (locked_teams or has_empty) else int(k)
    raw = suggest_moves(pm.model, pm.assignment, k=want, locks=locks, focus_students=focus,
                        focus_team=focus_team, allow_moves=allow_moves,
                        size_bounds=size_bounds, team_names=pm.team_names)
    out = []
    for item in raw:
        if item["kind"] == "move" and item["to_team"] in locked_teams:
            continue
        item = {**item, "students": [pm.session_id(s) for s in item["students"]]}
        op = {"kind": item["kind"], "students": list(item["students"]),
              "from_team": item["from_team"], "to_team": item["to_team"]}
        try:
            pv = preview(session, frame, policy, op, project_id=pid, size_bounds=size_bounds,
                         pm=pm, run_wide=False)
        except EditError:
            continue
        if pv["delta"] >= -_EPS:
            continue
        out.append({**item, "delta": pv["delta"], "objective_after": pv["objective_after"],
                    "reasons": pv["reasons"], "description": pv["description"],
                    "preview": pv})
    out.sort(key=lambda s: (s["delta"], s["students"]))
    return out[:max(0, int(k))]


def polish(session: EditSession, frame: pd.DataFrame, policy: Any, *, max_steps: int = 10,
           allow_moves: bool = False, size_bounds: Optional[Tuple[int, int]] = None,
           project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    pid = session._require_editable(project_id)
    trial = EditSession(projects=copy.deepcopy(session.projects), project_id=pid,
                        locks=copy.deepcopy(session.locks),
                        locked_teams=copy.deepcopy(session.locked_teams))
    applied: List[Dict[str, Any]] = []
    for _ in range(max(0, int(max_steps))):
        best = suggestions(trial, frame, policy, k=1, allow_moves=allow_moves,
                           size_bounds=size_bounds, project_id=pid)
        if not best:
            break
        trial.apply_suggestion(best[0], pid)
        applied.append(best[0])
    if applied:
        total = sum(s["delta"] for s in applied)
        session.set_teams(trial.teams(pid), project_id=pid,
                          label=f"Polished: {len(applied)} improving change(s), "
                                f"objective {total:+.2f}")
    return applied


@dataclass
class ReoptimizeOutcome:
    applied: bool
    objective_before: float
    objective_after: float
    moved: List[str] = field(default_factory=list)
    locked: int = 0
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    run_wide: Optional[Dict[str, int]] = None

    @property
    def delta(self) -> float:
        return self.objective_after - self.objective_before if self.applied else 0.0


def _align_free_teams(candidate: List[List[str]], current: Sequence[Sequence[str]],
                      free: Sequence[int]) -> List[List[str]]:
    from scipy.optimize import linear_sum_assignment

    out = [list(team) for team in candidate]
    by_size: Dict[int, List[int]] = {}
    for t in free:
        by_size.setdefault(len(candidate[t]), []).append(t)
    for group in by_size.values():
        if len(group) < 2:
            continue
        old = [set(current[t]) for t in group]
        overlap = [[-len(set(candidate[c]) & old[k]) for k in range(len(group))] for c in group]
        rows, cols = linear_sum_assignment(overlap)
        for r, c in zip(rows, cols):
            out[group[c]] = list(candidate[group[r]])
    return out


def reoptimize_unlocked(session: EditSession, frame: pd.DataFrame, policy: Any, *,
                        time_limit_s: float = 5.0, size_min: Optional[int] = None,
                        size_max: Optional[int] = None, project_id: Optional[str] = None,
                        progress: Optional[ProgressFn] = None,
                        cancel: Optional[CancelFn] = None) -> ReoptimizeOutcome:
    pid = session._require_editable(project_id)
    pm = project_model(session, frame, policy, pid)
    empty = [pm.team_names[t] for t, m in enumerate(pm.members) if not m]
    if empty:
        raise EditError(f"{', '.join(empty)} has no rostered students. Re-optimisation keeps "
                        f"the current team sizes: move students in or remove the team first")
    locks = {sid: t for sid, t in session.effective_locks(pid).items() if sid in pm.index}
    if len(locks) >= len(pm.pool_ids):
        raise EditError("Every student is locked; there is nothing to re-optimise")
    tuned = Policy.from_dict(as_policy(policy).to_dict())
    tuned.solver.time_limit_s = max(0.5, float(time_limit_s))
    tuned.solver.flexible_sizes = False
    tuned.solver.joint_projects = False
    tuned = es.capped_policy(tuned)
    sizes = [len(m) for m in pm.members]
    spec = ProjectSpec(
        project_id=pid, student_ids=list(pm.pool_ids), team_sizes=sizes,
        size_min=int(size_min if size_min is not None else min(sizes)),
        size_max=int(size_max if size_max is not None else max(sizes)),
        size_target=int(round(sum(sizes) / len(sizes))), locks=locks,
        team_names=session.team_names(pid),
    )
    def distance(k: int) -> int:
        up = pm.upcoming_index
        return up - k if k < up else k - up + 1

    prior = [pm.prior[k] for k in sorted(range(len(pm.prior)), key=lambda k: (-distance(k), k))]
    try:
        with run_slot(cancel=cancel, on_wait=_slot_waiter(progress)):
            result = run_allocation(frame, tuned, [spec], prior_projects=prior,
                                    progress=progress, cancel=cancel)
    except SlotWaitCancelled as exc:
        raise EditError(str(exc)) from exc
    except ValueError as exc:
        raise EditError(f"Re-optimisation failed: {exc}") from exc
    project = result.projects[0]
    pinned_teams = set(locks.values())
    solved = [[pm.session_id(sid) for sid in team] for team in project.teams]
    candidate = _align_free_teams(solved, pm.teams,
                                  [t for t in range(len(sizes)) if t not in pinned_teams])
    candidate_members = [[pm.index[sid] for sid in team if sid in pm.index]
                         for team in candidate]
    objective_after = _aggregate(pm.model, [pm.model.team_penalty(m) if m else None
                                            for m in candidate_members])
    for t, team in enumerate(pm.teams):
        candidate[t].extend(sid for sid in team if sid not in pm.index)
    before = {sid: t for t, team in enumerate(pm.teams) for sid in team}
    moved = [sid for t, team in enumerate(candidate) for sid in team if before.get(sid) != t]
    outcome = ReoptimizeOutcome(applied=False, objective_before=pm.objective,
                                objective_after=float(objective_after), moved=moved,
                                locked=len(locks), warnings=list(result.warnings),
                                elapsed_s=float(result.elapsed_s))
    if objective_after < pm.objective - _EPS and moved:
        previous = copy.deepcopy(session.projects)
        session.set_teams(candidate, project_id=pid,
                          label=f"Re-optimised unlocked students ({len(moved)} moved, "
                                f"{len(locks)} locked)")
        outcome.applied = True
        outcome.run_wide = run_wide_delta(previous, session.projects, frame, policy)
        warning = run_wide_warning(outcome.run_wide)
        if warning:
            outcome.warnings.append(warning)
    else:
        outcome.moved = []
        outcome.objective_after = pm.objective
    return outcome


def evaluate_plain(projects: Sequence[Dict[str, Any]], config: Any, students: Sequence[Any], *,
                   policy: Any = None, raw_df: Optional[pd.DataFrame] = None,
                   project_ids: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    legacy = es.projects_from_ids(list(projects), students)
    pol = as_policy(policy) if policy is not None else None
    ids = None if project_ids is None else [str(p) for p in project_ids]
    return es.evaluate_projects(legacy, config, students, policy=pol, raw_df=raw_df,
                                project_ids=ids)


def refresh_reports(session: EditSession, config: Any, students: Sequence[Any], *,
                    policy: Any = None, raw_df: Optional[pd.DataFrame] = None,
                    project_ids: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    editable = session.editable_project_ids()
    wanted = list(session.stale if project_ids is None else project_ids)
    ids = [pid for pid in editable if pid in wanted]
    fresh: Dict[str, Dict[str, Any]] = {}
    if ids:
        fresh = evaluate_plain(session.export_projects(), config, students, policy=policy,
                               raw_df=raw_df, project_ids=ids)
        reports = dict(session.reports)
        for pid in ids:
            if pid in fresh:
                reports[pid] = fresh[pid]
            else:
                reports.pop(pid, None)
        session.reports = reports
        changed = set(session.changed_project_ids())
        for entry in session.projects:
            pid = entry["project_id"]
            if pid in changed:
                break
            if pid in fresh and pid not in session.original_reports:
                session.original_reports[pid] = fresh[pid]
    session.stale = [pid for pid in session.stale if pid not in ids and pid in editable]
    return fresh


def rescore(session: EditSession, config: Any, students: Sequence[Any], *,
            policy: Any = None, raw_df: Optional[pd.DataFrame] = None) -> None:
    session.invalidate_reports()
    session.original_reports = evaluate_plain(session.original, config, students,
                                              policy=policy, raw_df=raw_df)
    refresh_reports(session, config, students, policy=policy, raw_df=raw_df)


@dataclass
class RosterDiff:
    project_id: str
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


def roster_diff(session: EditSession, roster_ids: Sequence[str], *,
                pool: Optional[Sequence[str]] = None,
                project_id: Optional[str] = None) -> RosterDiff:
    pid = session._pid(project_id)
    roster = IdMatcher(roster_ids)
    expected = []
    for sid in (pool if pool is not None else roster.ids):
        match = roster.resolve(sid)
        if match is not None and match not in expected:
            expected.append(match)
    expected_set = set(expected)
    current = session.student_ids(pid)
    in_project = IdMatcher(current)
    return RosterDiff(project_id=pid,
                      added=[s for s in expected if in_project.resolve(s) is None],
                      removed=[s for s in current if roster.resolve(s) not in expected_set])


@dataclass
class RepairOutcome:
    moved: List[str]
    added: List[str]
    removed: List[str]
    objective: float
    warnings: List[str] = field(default_factory=list)
    run_wide: Optional[Dict[str, int]] = None


def repair(session: EditSession, frame: pd.DataFrame, policy: Any, *,
           pool: Optional[Sequence[str]] = None, size_min: int = 3, size_max: int = 8,
           size_target: int = 5, move_weight: float = 5.0, keep_team_count: bool = True,
           project_id: Optional[str] = None, progress: Optional[ProgressFn] = None,
           cancel: Optional[CancelFn] = None) -> RepairOutcome:
    pid = session._require_editable(project_id)
    roster = [str(s) for s in frame["student_id"].tolist()]
    diff = roster_diff(session, roster, pool=pool, project_id=pid)
    if not diff.changed:
        raise EditError("The project already matches the roster; there is nothing to repair")
    matcher = IdMatcher(roster)
    expected_set = set(roster) if pool is None else \
        {m for m in (matcher.resolve(s) for s in pool) if m is not None}
    expected = [s for s in roster if s in expected_set]
    if not expected:
        raise EditError("No participant of this project is in the loaded roster")
    teams = session.teams(pid)
    locks = {sid: t for sid, t in session.effective_locks(pid).items()
             if matcher.resolve(sid) in expected_set}
    spec = ProjectSpec(project_id=pid, student_ids=expected, size_min=int(size_min),
                       size_max=int(size_max), size_target=int(size_target),
                       num_teams=len(teams) if keep_team_count else 0, locks=locks,
                       team_names=session.team_names(pid))
    prior = history_for(session, pid, matcher)
    try:
        with run_slot(cancel=cancel, on_wait=_slot_waiter(progress)):
            result = engine_repair.repair_allocation(
                frame, es.capped_policy(as_policy(policy)), spec, [list(t) for t in teams],
                prior_projects=prior, move_weight=float(move_weight),
                keep_team_count=bool(keep_team_count), progress=progress, cancel=cancel)
    except SlotWaitCancelled as exc:
        raise EditError(str(exc)) from exc
    except NotImplementedError as exc:
        raise RepairUnavailable(
            "Automatic repair is not available in this version of the engine yet. Place the "
            "new students and remove the departed ones by hand below, or run Team Formation "
            "again.") from exc
    except (ValueError, KeyError) as exc:
        raise EditError(f"Repair failed: {exc}") from exc
    stats = result.stats or {}
    previous = copy.deepcopy(session.projects)
    session.set_teams(result.teams, project_id=pid,
                      team_names=result.team_names if keep_team_count else None,
                      label=f"Repaired after roster change (+{len(diff.added)} / "
                            f"−{len(diff.removed)})")
    warnings = list(result.warnings)
    run_wide = run_wide_delta(previous, session.projects, frame, policy)
    warning = run_wide_warning(run_wide)
    if warning:
        warnings.append(warning)
    return RepairOutcome(moved=list(stats.get("moved", [])),
                         added=list(stats.get("added", diff.added)),
                         removed=list(stats.get("removed", diff.removed)),
                         objective=float(result.objective), warnings=warnings,
                         run_wide=run_wide)
