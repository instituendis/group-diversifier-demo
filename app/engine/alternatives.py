from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .history import PriorProject
from .objective import ObjectiveModel
from .policy import Policy
from .runner import (
    ProjectResult,
    ProjectSpec,
    _check_policy,
    _history,
    _prepare,
    _ProgressScaler,
    _project_result,
    _solve,
    encode_roster,
)
from .solver import optimize

Pair = Tuple[int, int]

MAX_ATTEMPTS = 5
_SEED_STRIDE = 101


def teammate_pairs(assignment: Sequence[int]) -> Set[Pair]:
    teams: Dict[int, List[int]] = {}
    for i, t in enumerate(assignment):
        teams.setdefault(int(t), []).append(i)
    pairs: Set[Pair] = set()
    for members in teams.values():
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1:]:
                pairs.add((a, b))
    return pairs


def pair_overlap(pairs: Set[Pair], earlier: Set[Pair]) -> float:
    if not pairs:
        return 0.0
    return len(pairs & earlier) / len(pairs)


def generate_alternatives(
    roster_frame: pd.DataFrame,
    policy: Policy,
    project: ProjectSpec,
    *,
    prior_projects: Sequence[PriorProject] = (),
    k: int = 3,
    min_difference: float = 0.4,
    progress: Optional[Callable[[float, float, str], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> List[ProjectResult]:
    policy = Policy.from_dict(policy) if not isinstance(policy, Policy) else policy
    k = max(1, int(k))
    target = 1.0 - min(1.0, max(0.0, float(min_difference)))
    _check_policy(roster_frame, policy)
    cohort, warnings = encode_roster(roster_frame, policy)
    prep = _prepare(cohort, project, policy, warnings)
    prior = [p for p in (prior_projects or []) if str(p.project_id) != str(project.project_id)]
    history = _history(prior, policy, prep.pool)
    model = ObjectiveModel(prep.pool, policy, prep.sizes, history=history,
                           project_id=prep.spec.project_id)
    warnings.extend(f"Project {prep.spec.project_id}: {w}" for w in model.warnings)
    scaler = _ProgressScaler(progress)

    def cancelled() -> bool:
        try:
            return bool(cancel is not None and cancel())
        except Exception:
            return False

    first_warnings = list(warnings)
    assignment, stats = _solve(model, prep, policy, first_warnings,
                               scaler.slice(0.0, 1.0 / k, "Alternative 1"), cancel)
    stats.update(alternative=0, pair_overlap={}, objective_gap=0.0, overlap_met=True,
                 penalty_weight=0.0, attempts=1)
    first = _project_result(model, prep, assignment, stats, first_warnings)
    results = [first]
    chosen: List[Set[Pair]] = [teammate_pairs(assignment)]
    soft = float(first.report.get("overall", {}).get("soft_objective", first.objective) or 0.0)
    base = max(0.05, abs(soft) / max(1, len(chosen[0])))
    anneal = dataclasses.replace(policy.solver, method="anneal")

    for a in range(1, k):
        if cancelled():
            break
        seed = None if anneal.seed is None else int(anneal.seed) + _SEED_STRIDE * a
        settings = dataclasses.replace(anneal, seed=seed)
        best: Optional[Tuple[float, List[int], Dict[str, object], float]] = None
        weight = base
        attempts = 0
        for attempt in range(MAX_ATTEMPTS):
            attempts += 1
            penalties: Dict[Pair, float] = {}
            for pairs in chosen:
                for pair in pairs:
                    penalties[pair] = penalties.get(pair, 0.0) + weight
            steered = ObjectiveModel(prep.pool, policy, prep.sizes, history=history,
                                     project_id=prep.spec.project_id, pair_penalties=penalties)
            lo = (a + attempt / MAX_ATTEMPTS) / k
            hi = (a + (attempt + 1) / MAX_ATTEMPTS) / k
            sub = scaler.slice(lo, hi, f"Alternative {a + 1}")
            result = optimize(steered, settings=settings, locks=prep.locks,
                              size_bounds=prep.bounds, progress=sub, cancel=cancel)
            pairs = teammate_pairs(result.assignment)
            worst = max(pair_overlap(pairs, earlier) for earlier in chosen)
            if best is None or worst < best[0] - 1e-12:
                best = (worst, result.assignment, dict(result.stats), weight)
            if worst <= target + 1e-12 or result.stats.get("cancelled"):
                break
            weight *= 2.0
        assert best is not None
        worst, assignment, stats, weight = best
        pairs = teammate_pairs(assignment)
        overlaps = {j: round(pair_overlap(pairs, earlier), 6) for j, earlier in enumerate(chosen)}
        met = worst <= target + 1e-12
        own = list(warnings)
        if stats.get("time_limited") and not stats.get("cancelled"):
            own.append(f"Project {prep.spec.project_id}: alternative {a + 1} stopped at the time "
                       f"limit ({stats.get('time_limit_s', anneal.time_limit_s):g} s); results "
                       f"may vary between machines")
        if not met:
            own.append(f"Project {prep.spec.project_id}: alternative {a + 1} shares "
                       f"{worst:.0%} of its teammate pairs with an earlier alternative "
                       f"(target at most {target:.0%}); best effort after {attempts} attempts")
        stats.update(method="anneal", method_used="anneal", sizes=list(prep.sizes),
                     alternative=a, pair_overlap=overlaps, overlap_met=met,
                     penalty_weight=round(weight, 6), attempts=attempts)
        res = _project_result(model, prep, assignment, stats, own)
        res.stats["objective_gap"] = round(res.objective - first.objective, 6)
        results.append(res)
        chosen.append(pairs)
        if stats.get("cancelled"):
            break
    if progress is not None:
        try:
            progress(1.0, results[0].objective, "done")
        except Exception:
            pass
    return results
