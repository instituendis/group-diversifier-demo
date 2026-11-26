from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .feasibility import check_feasibility
from .history import PairHistoryData, PriorProject, build_pair_history
from .metrics import build_report, default_team_names
from .objective import ObjectiveModel
from .policy import Policy
from .roster import EncodedCohort, infer_attributes
from .sizing import (all_sizes_listed, describe_sizes, resolve_sizes, size_bounds_for,
                     size_order_key,
                     size_problems)
from .solver import effective_time_limit, optimize

ProgressFn = Callable[[float, float, str], None]
CancelFn = Callable[[], bool]

JOINT_ROUNDS = 3
_JOINT_ROUNDS = JOINT_ROUNDS
MILP_TIME_SHARE = 0.5
_SHOW_IDS = 8


@dataclass
class ProjectSpec:
    project_id: str
    student_ids: List[str]
    team_sizes: Optional[List[int]] = None
    size_min: int = 3
    size_max: int = 8
    size_target: int = 5
    num_teams: int = 0
    locks: Dict[str, int] = field(default_factory=dict)
    team_names: Optional[List[str]] = None


@dataclass
class ProjectResult:
    project_id: str
    teams: List[List[str]]
    team_names: List[str]
    objective: float
    team_penalties: List[float]
    hard_units: int
    report: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    unknown_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngineResult:
    projects: List[ProjectResult]
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    policy: Dict[str, Any] = field(default_factory=dict)
    method: str = "anneal"
    feasibility: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_objective(self) -> float:
        return float(sum(p.objective for p in self.projects))

    def project(self, project_id: str) -> Optional[ProjectResult]:
        for p in self.projects:
            if str(p.project_id) == str(project_id):
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def encode_roster(roster_frame: pd.DataFrame, policy: Policy) -> Tuple[EncodedCohort, List[str]]:
    if roster_frame is None or "student_id" not in roster_frame.columns:
        raise ValueError("The roster needs a 'student_id' column")
    infos = infer_attributes(roster_frame)
    kinds: Dict[str, str] = {}
    orders: Dict[str, List[str]] = {}
    missing: List[str] = []
    warnings: List[str] = []
    for name in policy.referenced_attributes():
        info = infos.get(name)
        if info is None:
            continue
        if info.kind == "empty":
            warnings.append(f"Attribute '{name}' has no values; rules on it are ignored")
        elif info.kind in ("text", "constant"):
            kinds[name] = "categorical"
    for rule in policy.attributes:
        if not rule.enabled:
            continue
        if rule.kind:
            kinds[rule.attribute] = rule.kind
        if rule.order:
            orders[rule.attribute] = list(rule.order)
        if rule.missing_as_category:
            missing.append(rule.attribute)
    cohort = EncodedCohort.from_frame(roster_frame, kinds=kinds, orders=orders,
                                      missing_as_category=missing)
    return cohort, warnings


def _attribute_kinds(roster_frame: pd.DataFrame) -> Dict[str, str]:
    return {name: info.kind for name, info in infer_attributes(roster_frame).items()}


def _check_policy(roster_frame: pd.DataFrame, policy: Policy) -> None:
    problems = policy.validate(_attribute_kinds(roster_frame))
    if problems:
        raise ValueError("Invalid policy: " + "; ".join(problems))


@dataclass
class _Prepared:
    spec: ProjectSpec
    pool: EncodedCohort
    sizes: List[int]
    locks: Dict[int, int]
    team_names: List[str]
    bounds: Tuple[int, int]


def _canonical_pool(cohort: EncodedCohort, ids: Sequence[str], where: str,
                    warnings: List[str]) -> List[str]:
    pool: List[str] = []
    unknown: List[str] = []
    seen = set()
    duplicates = 0
    for raw in ids:
        i = cohort.lookup(raw)
        if i is None:
            unknown.append(str(raw))
            continue
        if i in seen:
            duplicates += 1
            continue
        seen.add(i)
        pool.append(cohort.student_ids[i])
    if unknown:
        shown = ", ".join(unknown[:10]) + (" ..." if len(unknown) > 10 else "")
        raise ValueError(
            f"{where}: {len(unknown)} participant id(s) are not in the roster: {shown}")
    if duplicates:
        warnings.append(f"{where}: {duplicates} duplicate participant id(s) ignored")
    pool.sort(key=lambda sid: cohort.index[sid])
    return pool


def _prepare(cohort: EncodedCohort, spec: ProjectSpec, policy: Policy,
             warnings: List[str]) -> _Prepared:
    where = f"Project {spec.project_id}"
    pool_ids = _canonical_pool(cohort, list(spec.student_ids), where, warnings)
    n = len(pool_ids)
    if n == 0:
        raise ValueError(f"{where}: the participant pool is empty")
    if spec.team_sizes:
        sizes = [int(s) for s in spec.team_sizes]
        if any(s <= 0 for s in sizes):
            raise ValueError(f"{where}: team sizes must be positive")
        if sum(sizes) != n:
            raise ValueError(f"{where}: team sizes sum to {sum(sizes)} but the pool has "
                             f"{n} students")
    else:
        sizes = resolve_sizes(n, spec.size_min, spec.size_max, spec.size_target, spec.num_teams)
    breakdown = ("" if all_sizes_listed(sizes, spec.size_min, spec.size_max)
                 else f" ({describe_sizes(sizes)})")
    for problem in size_problems(sizes, spec.size_min, spec.size_max):
        warnings.append(f"{where}: {problem}{breakdown}")
    pool = cohort.subset(pool_ids)
    locks: Dict[int, int] = {}
    per_team = [0] * len(sizes)
    for sid, team in (spec.locks or {}).items():
        i = pool.lookup(sid)
        if i is None:
            raise ValueError(f"{where}: locked student {sid!r} is not a participant")
        team = int(team)
        if not 0 <= team < len(sizes):
            raise ValueError(f"{where}: student {sid!r} is locked to team {team + 1}, but there "
                             f"are only {len(sizes)} teams")
        locks[i] = team
        per_team[team] += 1
    for t, k in enumerate(per_team):
        if k > sizes[t]:
            raise ValueError(f"{where}: {k} students are locked to team {t + 1}, which has "
                             f"{sizes[t]} places")
    names = list(spec.team_names or [])[:len(sizes)]
    names += default_team_names(len(sizes))[len(names):]
    bounds = size_bounds_for(sizes, spec.size_min, spec.size_max, policy.solver.flexible_sizes)
    return _Prepared(spec=spec, pool=pool, sizes=sizes, locks=locks, team_names=names,
                     bounds=bounds)


def _history(prior: Sequence[PriorProject], policy: Policy, pool: EncodedCohort,
             upcoming_index: Optional[int] = None) -> PairHistoryData:
    return build_pair_history(prior, policy.history, pool, upcoming_index)


def _id_list(ids: Sequence[str]) -> str:
    return ", ".join(ids[:_SHOW_IDS]) + (" …" if len(ids) > _SHOW_IDS else "")


def _unknown_id_warnings(cohort: EncodedCohort, policy: Policy,
                         prior: Sequence[PriorProject]) -> List[str]:
    out: List[str] = []
    for rule in policy.pairs:
        if not rule.enabled:
            continue
        unknown = [str(sid) for sid in dict.fromkeys(rule.students)
                   if cohort.lookup(sid) is None]
        if not unknown:
            continue
        known = len(set(rule.students)) - len(unknown)
        effect = ("the rule is skipped" if known < 2
                  else "the rule applies to the other students")
        name = rule.label or ", ".join(map(str, rule.students))
        out.append(f"Pair rule '{name}': {len(unknown)} student id(s) are not in the roster "
                   f"({_id_list(unknown)}); {effect}. Check the ids.")
    if policy.history.enabled:
        unknown_hist: List[str] = []
        seen = set()
        for project in prior:
            for team in project.teams:
                for sid in team:
                    text = str(sid).strip()
                    if text and text not in seen and cohort.lookup(text) is None:
                        seen.add(text)
                        unknown_hist.append(text)
        if unknown_hist:
            out.append(f"{len(unknown_hist)} student id(s) in earlier projects are not in the "
                       f"roster and are ignored for repeat checks ({_id_list(unknown_hist)}). "
                       "If they should match, check the id format.")
    return out


def _valid_assignment(assignment: Optional[Sequence[int]], prep: _Prepared,
                      flexible: bool) -> bool:
    if assignment is None or len(assignment) != prep.pool.n:
        return False
    T = len(prep.sizes)
    counts = [0] * T
    for t in assignment:
        if t is None or not 0 <= int(t) < T:
            return False
        counts[int(t)] += 1
    if not flexible and counts != prep.sizes:
        return False
    if flexible and any(c < prep.bounds[0] or c > prep.bounds[1] for c in counts):
        return False
    return all(int(assignment[i]) == t for i, t in prep.locks.items())


def _cancelled(cancel: Optional[CancelFn]) -> bool:
    try:
        return bool(cancel is not None and cancel())
    except Exception:
        return False


def _gap_text(gap: Any) -> str:
    return "n/a" if gap is None else f"{100 * float(gap):.1f}%"


def _solve(model: ObjectiveModel, prep: _Prepared, policy: Policy, warnings: List[str],
           progress: Optional[ProgressFn], cancel: Optional[CancelFn],
           initial: Optional[List[int]] = None) -> Tuple[List[int], Dict[str, Any]]:
    settings = policy.solver
    method = settings.method
    where = f"Project {prep.spec.project_id}"
    stats: Dict[str, Any] = {"method": method, "sizes": list(prep.sizes)}
    limit = effective_time_limit(settings, prep.pool.n)
    started = time.perf_counter()
    extra: Optional[List[int]] = None
    milp_note: Optional[str] = None
    milp_limited = False
    anneal_progress = progress
    if method in ("milp", "milp+anneal") and initial is None:
        milp_limit = limit * MILP_TIME_SHARE
        milp_progress = anneal_progress = None
        if progress is not None:
            def milp_progress(fraction: float, message: str) -> None:
                progress(MILP_TIME_SHARE * fraction, float("inf"), message)

            def anneal_progress(fraction: float, best: float, message: str) -> None:
                progress(MILP_TIME_SHARE + (1 - MILP_TIME_SHARE) * fraction, best, message)

        milp_assignment, milp_info = _run_milp(model, prep, policy, milp_limit, cancel,
                                               milp_progress)
        stats["milp"] = milp_info
        status = milp_info.get("status")
        milp_limited = status in ("feasible", "timeout")
        if status == "cancelled":
            stats["cancelled"] = True
        elif _valid_assignment(milp_assignment, prep, settings.flexible_sizes):
            assignment = [int(t) for t in milp_assignment]
            if method == "milp" and status == "optimal":
                stats.update(method_used="milp", time_limited=False, restarts_completed=0,
                             restarts_planned=0, seeds=[])
                return assignment, stats
            extra = assignment
            if method == "milp":
                milp_note = (
                    f"{where}: the MILP solver did not prove its solution optimal within "
                    f"{milp_limit:g} s (status {status}, gap {_gap_text(milp_info.get('gap'))})")
        else:
            warnings.append(
                f"{where}: MILP solver returned no usable allocation "
                f"({status}: {milp_info.get('message') or 'no details'}); "
                f"used annealing instead"
            )
    remaining = max(0.0, limit - (time.perf_counter() - started))
    result = optimize(model, settings=settings, initial=initial, extra_start=extra,
                      locks=prep.locks, size_bounds=prep.bounds, progress=anneal_progress,
                      cancel=cancel, time_limit_s=remaining)
    used = extra is not None and bool(result.stats.get("warm_start_ran"))
    if extra is not None and not used:
        lead = f"{milp_note}; its solution" if milp_note else f"{where}: the MILP solution"
        warnings.append(f"{lead} was not used because the search stopped before its annealing "
                        f"restart could run.")
    elif milp_note:
        warnings.append(
            f"{milp_note}; the solution was polished by annealing as with 'milp+anneal'. "
            f"Time-limited MILP results may vary between machines.")
    cancelled_before = bool(stats.get("cancelled"))
    stats.update(result.stats)
    stats["cancelled"] = bool(result.stats.get("cancelled")) or cancelled_before
    stats["time_limit_s"] = round(limit, 3)
    for message in result.stats.get("warnings", []):
        if f"{where}: {message}" not in warnings:
            warnings.append(f"{where}: {message}")
    stats["method_used"] = "milp+anneal" if used else "anneal"
    won = bool(result.stats.get("warm_start_won"))
    stats["time_limited"] = bool(result.stats.get("time_limited") or
                                 (milp_limited and (method == "milp" or won)))
    if result.stats.get("time_limited") and not stats["cancelled"]:
        warnings.append(
            f"{where}: search stopped at the time limit ({limit:g} s) after "
            f"{result.stats.get('restarts_completed', 0)} of "
            f"{result.stats.get('restarts_planned', 0)} restarts; results may vary between "
            f"machines and runs. Raise the time limit or the number of workers for a "
            f"complete, reproducible search.")
    elif milp_limited and won and method == "milp+anneal":
        warnings.append(
            f"{where}: the best allocation grew from a time-limited MILP solution "
            f"(gap {_gap_text((stats.get('milp') or {}).get('gap'))}); results may vary between "
            f"machines.")
    return result.assignment, stats


def _run_milp(model: ObjectiveModel, prep: _Prepared, policy: Policy, time_limit_s: float,
              cancel: Optional[CancelFn] = None,
              progress: Optional[Callable[[float, str], None]] = None,
              ) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    from .milp import solve_milp

    if _cancelled(cancel):
        return None, {"status": "cancelled", "message": "cancelled before solving",
                      "elapsed_s": 0.0, "objective": None, "gap": None}
    hp = policy.history
    forbidden: List[Tuple[int, int]] = []
    pair_weights: Dict[Tuple[int, int], float] = {}
    if model.history is not None and hp.enabled:
        if hp.repeat_weight:
            pair_weights.update(model.history.pair_weights(hp.repeat_weight))
        cap_pairs = model.history.forbidden_pairs(hp.max_times_together)
        if hp.hard_cap:
            forbidden.extend(cap_pairs)
        elif hp.cap_weight:
            for pair in cap_pairs:
                pair_weights[pair] = pair_weights.get(pair, 0.0) + float(hp.cap_weight)
    if policy.preferences.use_student_preferences and policy.preferences.avoid_hard:
        for i, listed in prep.pool.avoid.items():
            for j in listed:
                forbidden.append((min(i, j), max(i, j)))
    forbidden = sorted(set(forbidden))
    try:
        res = solve_milp(
            prep.pool, policy, prep.sizes, forbidden_pairs=forbidden,
            pair_weights=pair_weights or None, locks=prep.locks, time_limit_s=time_limit_s,
            mip_gap=policy.solver.mip_gap, project_id=prep.spec.project_id,
            cancel=cancel, progress=progress,
        )
    except Exception as exc:
        return None, {"status": "error", "message": str(exc)}
    info = {"status": res.status, "objective": res.objective, "gap": res.gap,
            "message": res.message, "elapsed_s": res.elapsed_s,
            "time_limit_s": round(float(time_limit_s), 3)}
    return res.assignment, info


def _project_result(model: ObjectiveModel, prep: _Prepared, assignment: List[int],
                    stats: Dict[str, Any], warnings: List[str]) -> ProjectResult:
    teams_idx = model.teams_of(assignment)
    ids = prep.pool.student_ids
    penalties = [model.team_penalty(m) for m in teams_idx]
    report = build_report(model, assignment, team_names=prep.team_names, locks=prep.locks)
    outlier = (report.get("overall") or {}).get("outlier_team")
    if outlier:
        text = f"Project {prep.spec.project_id}: {outlier['text']}"
        if text not in warnings:
            warnings = list(warnings) + [text]
    return ProjectResult(
        project_id=str(prep.spec.project_id),
        teams=[[ids[i] for i in members] for members in teams_idx],
        team_names=list(prep.team_names),
        objective=model.aggregate(penalties),
        team_penalties=penalties,
        hard_units=model.assignment_hard_units(assignment),
        report=report,
        stats=stats,
        warnings=list(warnings),
    )


class _ProgressScaler:
    def __init__(self, progress: Optional[ProgressFn]) -> None:
        self.progress = progress

    def slice(self, lo: float, hi: float, label: str) -> Optional[ProgressFn]:
        if self.progress is None:
            return None

        def report(fraction: float, best: float, message: str) -> None:
            self.progress(lo + (hi - lo) * fraction, best, f"{label}: {message}")
        return report


def run_allocation(roster_frame: pd.DataFrame, policy: Any, projects: List[ProjectSpec],
                   prior_projects: Sequence[PriorProject] = (),
                   progress: Optional[ProgressFn] = None,
                   cancel: Optional[CancelFn] = None) -> EngineResult:
    started = time.perf_counter()
    policy = Policy.from_dict(policy) if not isinstance(policy, Policy) else policy
    if not projects:
        raise ValueError("No projects to allocate")
    ids_seen = [str(p.project_id) for p in projects]
    if len(set(ids_seen)) != len(ids_seen):
        raise ValueError("Project ids must be unique")
    _check_policy(roster_frame, policy)
    cohort, warnings = encode_roster(roster_frame, policy)
    prepared = [_prepare(cohort, spec, policy, warnings) for spec in projects]
    imported = list(prior_projects or [])
    warnings.extend(_unknown_id_warnings(cohort, policy, imported))
    feasibility = _feasibility(roster_frame, policy, projects, imported, warnings)
    settings = policy.solver
    joint = bool(settings.joint_projects) and len(prepared) > 1
    scaler = _ProgressScaler(progress)
    seq_share = 0.5 if joint else 1.0
    K = len(prepared)

    assignments: List[List[int]] = []
    stats_list: List[Dict[str, Any]] = []
    project_warnings: List[List[str]] = []
    generated: List[PriorProject] = []
    for k, prep in enumerate(prepared):
        own: List[str] = []
        history = _history(imported + generated, policy, prep.pool)
        model = ObjectiveModel(prep.pool, policy, prep.sizes, history=history,
                               project_id=prep.spec.project_id)
        own.extend(f"Project {prep.spec.project_id}: {w}" for w in model.warnings)
        sub = scaler.slice(seq_share * k / K, seq_share * (k + 1) / K,
                           f"Project {prep.spec.project_id}")
        assignment, stats = _solve(model, prep, policy, own, sub, cancel)
        assignments.append(assignment)
        stats_list.append(stats)
        project_warnings.append(own)
        generated.append(PriorProject(str(prep.spec.project_id),
                                      _teams_as_ids(prep, assignment)))

    if joint:
        joint_stats = _joint_rounds(prepared, imported, policy, assignments, scaler,
                                    seq_share, cancel)
        for stats in stats_list:
            stats["joint"] = joint_stats

    assignments = [_target_first(prep, a) for prep, a in zip(prepared, assignments)]
    results: List[ProjectResult] = []
    generated = []
    for k, prep in enumerate(prepared):
        history = _history(imported + generated, policy, prep.pool)
        model = ObjectiveModel(prep.pool, policy, prep.sizes, history=history,
                               project_id=prep.spec.project_id)
        result = _project_result(model, prep, assignments[k], stats_list[k], project_warnings[k])
        results.append(result)
        generated.append(PriorProject(result.project_id, result.teams))
        warnings.extend(w for w in result.warnings if w not in warnings)
    if cancel is not None and any(s.get("cancelled") for s in stats_list):
        warnings.append("Allocation was cancelled; results are the best found so far")
    if progress is not None:
        progress(1.0, sum(r.objective for r in results), "done")
    return EngineResult(
        projects=results,
        warnings=warnings,
        elapsed_s=round(time.perf_counter() - started, 4),
        policy=policy.to_dict(),
        method=settings.method,
        feasibility=feasibility,
    )


_FEASIBILITY_PROBE_S = 0.2


def _feasibility(roster_frame: pd.DataFrame, policy: Policy, projects: List[ProjectSpec],
                 prior: Sequence[PriorProject], warnings: List[str]) -> Dict[str, Any]:
    try:
        report = check_feasibility(roster_frame, policy, projects, prior,
                                   probe_time_s=_FEASIBILITY_PROBE_S)
    except Exception as exc:
        warnings.append(f"Feasibility check failed: {exc}")
        return {}
    for finding in report.errors:
        if finding.area in ("sizes", "pool"):
            continue
        text = f"Feasibility: {finding.message}"
        if text not in warnings:
            warnings.append(text)
    return report.to_dict()


def _target_first(prep: _Prepared, assignment: List[int]) -> List[int]:
    spec = prep.spec
    if spec.team_sizes or prep.locks or spec.team_names:
        return assignment
    counts = [0] * len(prep.sizes)
    for t in assignment:
        counts[t] += 1
    order = sorted(range(len(counts)),
                   key=lambda t: (size_order_key(counts[t], spec.size_target), t))
    if order == list(range(len(counts))):
        return assignment
    new_index = {old: new for new, old in enumerate(order)}
    return [new_index[t] for t in assignment]


def _teams_as_ids(prep: _Prepared, assignment: Sequence[int]) -> List[List[str]]:
    teams: List[List[str]] = [[] for _ in prep.sizes]
    for i, t in enumerate(assignment):
        teams[t].append(prep.pool.student_ids[i])
    return teams


def _sequential_total(prepared: List[_Prepared], imported: List[PriorProject], policy: Policy,
                      assignments: List[List[int]]) -> float:
    total = 0.0
    generated: List[PriorProject] = []
    for prep, assignment in zip(prepared, assignments):
        model = ObjectiveModel(prep.pool, policy, prep.sizes,
                               history=_history(imported + generated, policy, prep.pool),
                               project_id=prep.spec.project_id)
        total += model.evaluate(assignment)
        generated.append(PriorProject(str(prep.spec.project_id), _teams_as_ids(prep, assignment)))
    return total


def _joint_rounds(prepared: List[_Prepared], imported: List[PriorProject], policy: Policy,
                  assignments: List[List[int]], scaler: _ProgressScaler, seq_share: float,
                  cancel: Optional[CancelFn]) -> Dict[str, Any]:
    K = len(prepared)
    total = _sequential_total(prepared, imported, policy, assignments)
    info: Dict[str, Any] = {"rounds": 0, "accepted": 0, "initial_total": total}
    for rnd in range(_JOINT_ROUNDS):
        changed = False
        for k, prep in enumerate(prepared):
            if cancel is not None and cancel():
                info["final_total"] = total
                return info
            others = [PriorProject(str(p.spec.project_id), _teams_as_ids(p, assignments[m]))
                      for m, p in enumerate(prepared) if m != k]
            history = _history(imported + others, policy, prep.pool,
                               upcoming_index=len(imported) + k)
            model = ObjectiveModel(prep.pool, policy, prep.sizes, history=history,
                                   project_id=prep.spec.project_id)
            lo = seq_share + (1 - seq_share) * (rnd * K + k) / (_JOINT_ROUNDS * K)
            hi = seq_share + (1 - seq_share) * (rnd * K + k + 1) / (_JOINT_ROUNDS * K)
            sub = scaler.slice(lo, hi, f"Joint round {rnd + 1}, project {prep.spec.project_id}")
            ignored: List[str] = []
            candidate, _ = _solve(model, prep, policy, ignored, sub, cancel,
                                  initial=list(assignments[k]))
            if candidate == assignments[k]:
                continue
            trial = list(assignments)
            trial[k] = candidate
            new_total = _sequential_total(prepared, imported, policy, trial)
            if new_total <= total + 1e-9:
                if new_total < total - 1e-9:
                    changed = True
                assignments[k] = candidate
                total = new_total
                info["accepted"] += 1
        info["rounds"] = rnd + 1
        if not changed:
            break
    info["final_total"] = total
    return info


def evaluate_allocation(roster_frame: pd.DataFrame, policy: Any, project: ProjectSpec,
                        teams: List[List[str]],
                        prior_projects: Sequence[PriorProject] = ()) -> ProjectResult:
    setup = evaluation_setup(roster_frame, policy, project, teams, prior_projects)
    result = _project_result(setup.model, setup.prep, setup.assignment, {"method": "evaluate"},
                             setup.warnings)
    result.unknown_ids = setup.unknown_ids
    return result


@dataclass
class EvaluationSetup:
    model: ObjectiveModel
    prep: _Prepared
    assignment: List[int]
    warnings: List[str]
    unknown_ids: List[str]


def evaluation_setup(roster_frame: pd.DataFrame, policy: Any, project: ProjectSpec,
                     teams: List[List[str]],
                     prior_projects: Sequence[PriorProject] = ()) -> EvaluationSetup:
    policy = Policy.from_dict(policy) if not isinstance(policy, Policy) else policy
    _check_policy(roster_frame, policy)
    cohort, warnings = encode_roster(roster_frame, policy)
    where = f"Project {project.project_id}"
    unknown: List[str] = []
    seen: Dict[int, int] = {}
    clean_teams: List[List[str]] = []
    names_in = list(project.team_names or [])
    names: List[str] = []
    duplicates = 0
    for t, members in enumerate(teams):
        kept: List[str] = []
        for raw in members:
            i = cohort.lookup(raw)
            if i is None:
                if str(raw) not in unknown:
                    unknown.append(str(raw))
                continue
            if i in seen:
                duplicates += 1
                continue
            seen[i] = t
            kept.append(cohort.student_ids[i])
        if kept:
            clean_teams.append(kept)
            names.append(names_in[t] if t < len(names_in) else f"Team {t + 1}")
        else:
            warnings.append(f"{where}: team {t + 1} has no known members and is left out")
    if unknown:
        warnings.append(f"{where}: {len(unknown)} id(s) are not in the roster and were ignored: "
                        + ", ".join(unknown[:10]) + (" ..." if len(unknown) > 10 else ""))
    if duplicates:
        warnings.append(f"{where}: {duplicates} student(s) appear in more than one team; "
                        f"the first team was kept")
    if not clean_teams:
        raise ValueError(f"{where}: no team has members that are in the roster")
    in_teams = {sid for team in clean_teams for sid in team}
    missing = []
    for raw in project.student_ids:
        i = cohort.lookup(raw)
        if i is not None and cohort.student_ids[i] not in in_teams:
            missing.append(cohort.student_ids[i])
    if missing:
        warnings.append(f"{where}: {len(missing)} participant(s) are not in any team and were "
                        f"left out of the evaluation")
    team_of = {sid: t for t, team in enumerate(clean_teams) for sid in team}
    pool_ids = sorted(team_of, key=lambda sid: cohort.index[sid])
    pool = cohort.subset(pool_ids)
    sizes = [len(team) for team in clean_teams]
    assignment = [team_of[sid] for sid in pool_ids]
    prior = [p for p in (prior_projects or []) if str(p.project_id) != str(project.project_id)]
    warnings.extend(_unknown_id_warnings(cohort, policy, prior))
    history = _history(prior, policy, pool)
    model = ObjectiveModel(pool, policy, sizes, history=history, project_id=project.project_id)
    warnings.extend(f"{where}: {w}" for w in model.warnings)
    spec = ProjectSpec(project_id=str(project.project_id), student_ids=pool_ids,
                       team_sizes=sizes, size_min=project.size_min, size_max=project.size_max,
                       size_target=project.size_target, team_names=names)
    prep = _Prepared(spec=spec, pool=pool, sizes=sizes, locks={}, team_names=names,
                     bounds=(min(sizes), max(sizes)))
    return EvaluationSetup(model=model, prep=prep, assignment=assignment, warnings=warnings,
                           unknown_ids=unknown)
