from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from .history import PriorProject
from .objective import ObjectiveModel
from .policy import Policy
from .runner import (
    ProjectResult,
    ProjectSpec,
    _canonical_pool,
    _check_policy,
    _history,
    _prepare,
    _project_result,
    encode_roster,
)
from .search_state import SearchState
from .sizing import order_sizes_target_first, plan_team_sizes, resolve_sizes
from .solver import optimize

REPAIR_RESTARTS = 2


def repair_allocation(
    roster_frame: pd.DataFrame,
    policy: Policy,
    project: ProjectSpec,
    current_teams: List[List[str]],
    *,
    prior_projects: Sequence[PriorProject] = (),
    move_weight: float = 5.0,
    keep_team_count: bool = True,
    progress: Optional[Callable[[float, float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> ProjectResult:
    policy = Policy.from_dict(policy) if not isinstance(policy, Policy) else policy
    _check_policy(roster_frame, policy)
    cohort, warnings = encode_roster(roster_frame, policy)
    where = f"Project {project.project_id}"
    pool_ids = _canonical_pool(cohort, list(project.student_ids), where, warnings)
    if not pool_ids:
        raise ValueError(f"{where}: the participant pool is empty")
    in_pool = set(pool_ids)

    original: Dict[str, int] = {}
    removed: List[str] = []
    unknown: List[str] = []
    counts = [0] * len(current_teams)
    for t, members in enumerate(current_teams):
        for raw in members:
            i = cohort.lookup(raw)
            if i is None:
                if str(raw) not in unknown:
                    unknown.append(str(raw))
                    removed.append(str(raw))
                continue
            sid = cohort.student_ids[i]
            if sid in original or sid in removed:
                continue
            if sid in in_pool:
                original[sid] = t
                counts[t] += 1
            else:
                removed.append(sid)
    added = [sid for sid in pool_ids if sid not in original]
    if unknown:
        warnings.append(f"{where}: {len(unknown)} id(s) in the current teams are not in the "
                        f"roster and were dropped")

    sizes = _repair_sizes(project, len(pool_ids), len(current_teams), counts, keep_team_count)
    spec = dataclasses.replace(project, student_ids=list(pool_ids), team_sizes=sizes,
                               team_names=list(project.team_names or []) or None)
    prep = _prepare(cohort, spec, policy, warnings)
    T = len(prep.sizes)
    ids = prep.pool.student_ids
    anchor = [original.get(sid, -1) for sid in ids]
    anchor = [a if a < T else -1 for a in anchor]
    prior = [p for p in (prior_projects or []) if str(p.project_id) != str(project.project_id)]
    history = _history(prior, policy, prep.pool)
    plain = ObjectiveModel(prep.pool, policy, prep.sizes, history=history,
                           project_id=prep.spec.project_id)
    warnings.extend(f"{where}: {w}" for w in plain.warnings)
    weight = max(0.0, float(move_weight))
    steered = ObjectiveModel(prep.pool, policy, prep.sizes, history=history,
                             project_id=prep.spec.project_id, anchor=anchor,
                             anchor_weight=weight) if weight > 0 else plain

    start = _repaired_start(steered, anchor, prep.locks)
    settings = dataclasses.replace(policy.solver, method="anneal",
                                   restarts=min(max(1, policy.solver.restarts), REPAIR_RESTARTS))
    result = optimize(steered, settings=settings, initial=start, locks=prep.locks,
                      size_bounds=prep.bounds, progress=progress, cancel=cancel)
    assignment = result.assignment
    moved = [ids[i] for i, t in enumerate(assignment)
             if ids[i] in original and original[ids[i]] != t]
    stats = dict(result.stats)
    stats.update(
        method="repair", method_used="anneal", sizes=list(prep.sizes),
        moved=moved, added=list(added), removed=list(removed),
        kept=sum(1 for sid in original if sid not in moved),
        move_weight=weight, anchor_objective=round(steered.evaluate(assignment), 6),
        team_count_kept=T == len(current_teams),
    )
    for message in result.stats.get("warnings", []):
        warnings.append(f"{where}: {message}")
    if result.stats.get("time_limited") and not result.stats.get("cancelled"):
        limit = result.stats.get("time_limit_s", settings.time_limit_s)
        warnings.append(f"{where}: repair stopped at the time limit ({limit:g} s); results "
                        f"may vary between machines")
    out = _project_result(plain, prep, assignment, stats, warnings)
    out.unknown_ids = unknown
    return out


def _repair_sizes(project: ProjectSpec, n: int, current_count: int, counts: List[int],
                  keep_team_count: bool) -> List[int]:
    if project.team_sizes and sum(project.team_sizes) == n and (
            not keep_team_count or len(project.team_sizes) == current_count):
        return [int(s) for s in project.team_sizes]
    if keep_team_count and current_count > 0:
        planned = plan_team_sizes(n, project.size_min, project.size_max, project.size_target,
                                  num_teams=min(current_count, n))
    else:
        planned = resolve_sizes(n, project.size_min, project.size_max, project.size_target,
                                project.num_teams)
    order = sorted(range(len(planned)),
                   key=lambda t: (-(counts[t] if t < len(counts) else 0), t))
    ranked = sorted(planned, reverse=True)
    sizes = [0] * len(planned)
    for t, size in zip(order, ranked):
        sizes[t] = size
    groups: Dict[int, List[int]] = {}
    for t in range(len(planned)):
        groups.setdefault(counts[t] if t < len(counts) else 0, []).append(t)
    for members in groups.values():
        for t, size in zip(members, order_sizes_target_first([sizes[t] for t in members],
                                                             project.size_target)):
            sizes[t] = size
    return sizes


def _repaired_start(model: ObjectiveModel, anchor: List[int],
                    locks: Dict[int, int]) -> List[int]:
    T = model.n_teams
    sizes = model.team_sizes
    assignment = [-1] * model.n
    for i, t in locks.items():
        assignment[i] = t
    for i, t in enumerate(anchor):
        if assignment[i] < 0 and 0 <= t < T:
            assignment[i] = t
    state = SearchState(model, [-1] * model.n, locks)
    for i, t in enumerate(assignment):
        if t >= 0:
            state.apply_add(i, t)
    for t in range(T):
        while state.sizes[t] > sizes[t]:
            candidates = [i for i in state.members[t] if i not in locks]
            if not candidates:
                break
            current = state.penalties[t]
            victim = min(candidates,
                         key=lambda i: (state._penalty_with(t, i, None) - current, -i))
            state._remove(t, victim)
            state._refresh_team(t)
            state._refresh_objective()
    for i in range(model.n):
        if state.assignment[i] >= 0:
            continue
        best_t, best_d = -1, float("inf")
        for t in range(T):
            if state.sizes[t] >= sizes[t]:
                continue
            d = state.delta_add(i, t)
            if d < best_d - 1e-12:
                best_t, best_d = t, d
        state.apply_add(i, best_t)
    return list(state.assignment)
