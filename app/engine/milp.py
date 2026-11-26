from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp

from ._milp_model import (
    MilpModel,
    build_model,
    hard_violations,
    linearised_objective,
    validate_inputs,
)
from .policy import FairnessPolicy, Policy
from .roster import EncodedCohort

_MIN_TIME_LIMIT_S = 1e-3

_TWO_PHASE_MIN_S = 2.0

_PHASE1_SHARE = 0.7


@dataclass
class MilpResult:
    assignment: Optional[List[int]]
    status: str
    objective: Optional[float] = None
    gap: Optional[float] = None
    message: str = ""
    elapsed_s: float = 0.0


def solve_milp(
    cohort: EncodedCohort,
    policy: Policy,
    team_sizes: List[int],
    *,
    forbidden_pairs: Iterable[Tuple[int, int]] = (),
    pair_weights: Optional[Dict[Tuple[int, int], float]] = None,
    locks: Optional[Dict[int, int]] = None,
    time_limit_s: float = 30.0,
    mip_gap: float = 0.01,
    project_id: Optional[str] = None,
    cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> MilpResult:
    start = time.perf_counter()
    forbidden, weights, norm_locks = validate_inputs(
        cohort, team_sizes, forbidden_pairs, pair_weights, locks)
    sizes = [int(s) for s in team_sizes]
    if cohort.n == 0:
        return MilpResult(assignment=[], status="optimal", objective=0.0, gap=0.0,
                          message="empty pool", elapsed_s=time.perf_counter() - start)

    def build(rules: Policy) -> MilpModel:
        return build_model(cohort, rules, sizes, forbidden_pairs=forbidden,
                           pair_weights=weights, locks=norm_locks, project_id=project_id)

    try:
        model = build(policy)
    except Exception as exc:
        return MilpResult(assignment=None, status="error",
                          message=f"MILP model construction failed: {exc}",
                          elapsed_s=time.perf_counter() - start)

    run = _Run(start, [f"MILP model: {model.describe()}, built in {model.build_s:.2f}s"])

    def stopped() -> bool:
        try:
            return bool(cancel is not None and cancel())
        except Exception:
            return False

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            try:
                progress(fraction, message)
            except Exception:
                pass

    if model.infeasible_reasons:
        run.say("infeasible before solving: " + "; ".join(model.infeasible_reasons))
        run.notes(model.notes)
        return run.result(None, "infeasible")

    time_limit = max(float(time_limit_s), _MIN_TIME_LIMIT_S)
    gap_tol = max(float(mip_gap), 0.0)
    incumbent: Optional[Tuple[List[int], float]] = None
    if stopped():
        run.say("cancelled before solving")
        return run.result(None, "cancelled")
    two_phase = model.has_worst_team and time_limit >= _TWO_PHASE_MIN_S
    if two_phase:
        report(0.0, "MILP phase 1 (total penalty)")
        try:
            outcome = _total_only_phase(build, policy, model, time_limit * _PHASE1_SHARE,
                                        gap_tol, run)
        except Exception as exc:
            run.say(f"phase 1 skipped: {exc}")
            outcome = None
        if outcome == "infeasible":
            run.notes(model.notes)
            return run.result(None, "infeasible")
        incumbent = outcome if isinstance(outcome, tuple) else None
        if stopped():
            run.say("cancelled after phase 1")
            run.notes(model.notes)
            if incumbent is None:
                return run.result(None, "cancelled")
            return run.result(incumbent[0], "feasible", incumbent[1])

    report(_PHASE1_SHARE if two_phase else 0.0,
           "MILP phase 2 (full objective)" if two_phase else "MILP solving")
    remaining = max(time_limit - (time.perf_counter() - start), _MIN_TIME_LIMIT_S)
    try:
        res, solve_s = _highs(model, remaining, gap_tol,
                              cutoff=incumbent[1] if incumbent else None)
    except Exception as exc:
        run.say(f"HiGHS failed: {exc}")
        if incumbent is None:
            return run.result(None, "error")
        run.notes(model.notes)
        return run.result(incumbent[0], "feasible", incumbent[1])
    label = "phase 2 (full objective)" if incumbent else "solved"
    run.say(f"{label} in {solve_s:.2f}s (HiGHS: {res.message})")
    report(1.0, "MILP done")
    run.notes(model.notes)

    assignment, problem = _extract(model, res)
    if problem:
        run.say(problem)
    bound = getattr(res, "mip_dual_bound", None)
    bound = float(bound) if bound is not None and np.isfinite(bound) else None

    if incumbent is None:
        if res.status == 2:
            return run.result(None, "infeasible")
        if res.status == 1 and res.x is None:
            return run.result(None, "timeout")
        if assignment is None or res.status not in (0, 1):
            return run.result(None, "error")
        gap = getattr(res, "mip_gap", None)
        gap = float(gap) if gap is not None and np.isfinite(gap) else None
        return run.result(assignment, "optimal" if res.status == 0 else "feasible",
                          float(res.fun), gap)

    best, best_obj = incumbent
    if assignment is not None and res.status in (0, 1) and float(res.fun) < best_obj:
        best, best_obj = assignment, float(res.fun)
    if res.status == 2:
        return run.result(best, "optimal", best_obj, 0.0)
    status = "optimal" if res.status == 0 and assignment is not None else "feasible"
    return run.result(best, status, best_obj, _gap(best_obj, bound))


class _Run:
    def __init__(self, start: float, parts: List[str]) -> None:
        self.start = start
        self.parts = parts

    def say(self, text: str) -> None:
        self.parts.append(text)

    def notes(self, notes: List[str]) -> None:
        self.parts.extend(f"note: {n}" for n in notes)

    def result(self, assignment: Optional[List[int]], status: str,
               objective: Optional[float] = None, gap: Optional[float] = None) -> MilpResult:
        return MilpResult(assignment=assignment, status=status, objective=objective, gap=gap,
                          message=". ".join(self.parts),
                          elapsed_s=time.perf_counter() - self.start)


def _highs(model: MilpModel, time_limit: float, mip_gap: float,
           cutoff: Optional[float] = None) -> Tuple[OptimizeResult, float]:
    A, lo, hi = model.A, model.row_lo, model.row_hi
    if cutoff is not None:
        A = sparse.vstack([A, sparse.csr_matrix(model.c.reshape(1, -1))], format="csr")
        lo = np.append(lo, -np.inf)
        hi = np.append(hi, cutoff + 1e-6 * max(1.0, abs(cutoff)))
    began = time.perf_counter()
    res = milp(
        model.c,
        constraints=LinearConstraint(A, lo, hi),
        integrality=model.integrality,
        bounds=Bounds(model.lb, model.ub),
        options={"time_limit": max(time_limit, _MIN_TIME_LIMIT_S),
                 "mip_rel_gap": mip_gap, "disp": False},
    )
    return res, time.perf_counter() - began


def _extract(model: MilpModel, res: OptimizeResult) -> Tuple[Optional[List[int]], str]:
    if res.x is None or res.status not in (0, 1):
        return None, ""
    x = np.asarray(res.x[: model.n * model.n_teams]).reshape(model.n, model.n_teams)
    if np.any(x.max(axis=1) < 0.5):
        return None, "solution is not integral for some students"
    assignment = [int(t) for t in np.argmax(x, axis=1)]
    problems = hard_violations(model, assignment)
    if problems:
        return None, "extracted solution failed verification: " + "; ".join(problems)
    return assignment, ""


def _total_only_phase(build: Callable[[Policy], MilpModel], policy: Policy, model: MilpModel,
                      time_limit: float, mip_gap: float,
                      run: _Run) -> Union[None, str, Tuple[List[int], float]]:
    sum_model = build(dataclasses.replace(policy, fairness=FairnessPolicy(worst_team_weight=0.0)))
    res, solve_s = _highs(sum_model, time_limit, mip_gap)
    run.say(f"phase 1 (total penalty only) in {solve_s:.2f}s (HiGHS: {res.message})")
    if res.status == 2:
        return "infeasible"
    assignment, problem = _extract(sum_model, res)
    if problem:
        run.say(f"phase 1: {problem}")
    if assignment is None:
        return None
    value = linearised_objective(model, assignment)
    return None if value is None else (assignment, value)


def _gap(objective: float, bound: Optional[float]) -> Optional[float]:
    if bound is None:
        return None
    return max(0.0, objective - bound) / max(abs(objective), 1e-9)
