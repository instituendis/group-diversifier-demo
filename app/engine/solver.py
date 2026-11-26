from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import random
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .objective import ObjectiveModel, model_from_args
from .policy import SolverSettings
from .search_state import SearchState
from .sizing import auto_iterations, polish_evaluations  # noqa: F401 (re-exported)

ProgressFn = Callable[[float, float, str], None]
CancelFn = Callable[[], bool]

_P_START_COLD = 0.2
_P_START_WARM = 0.05
_T_END_FRACTION = 1e-3
_RELAX_SHARE = 0.3
_RELAX_FACTOR = 1.0
_RELAX_STEPS = 5
_PAIR_SWAP_SHARE = 0.2
_CALIBRATION_SAMPLES = 300
_MOVE_SHARE = 0.25
_FOCUS_SHARE = 0.15
_CHECK_EVERY = 256
_PROGRESS_INTERVAL_S = 0.05
_IMPROVE_EPS = 1e-9
_POOL_POLL_S = 0.05
_COUNTERS = ("iterations", "evaluations", "accepted", "improvements")


@dataclass
class SolveResult:
    assignment: List[int]
    objective: float
    hard_units: int
    stats: Dict[str, Any] = field(default_factory=dict)


_TIME_LIMIT_STUDENTS = 200


def effective_time_limit(settings: SolverSettings, n: int) -> float:
    limit = float(settings.time_limit_s)
    if getattr(settings, "scale_time_limit", True) and n > _TIME_LIMIT_STUDENTS:
        limit *= n / _TIME_LIMIT_STUDENTS
    return limit


def restart_seed(seed: Optional[int], r: int) -> Optional[int]:
    if seed is None:
        return None
    digest = hashlib.blake2b(f"{int(seed)}:{int(r)}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class _Stopper:
    def __init__(self, time_limit_s: float, progress: Optional[ProgressFn],
                 cancel: Optional[CancelFn]) -> None:
        self.start = time.perf_counter()
        self.deadline = self.start + max(0.0, float(time_limit_s))
        self.progress = progress
        self.cancel = cancel
        self.last_progress = 0.0
        self.time_limited = False
        self.cancelled = False

    @property
    def stopped(self) -> bool:
        return self.time_limited or self.cancelled

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def check(self) -> bool:
        now = time.perf_counter()
        if now >= self.deadline:
            self.time_limited = True
        if self.cancel is not None and not self.cancelled:
            try:
                if self.cancel():
                    self.cancelled = True
            except Exception:
                pass
        return self.stopped

    def report(self, fraction: float, best: float, message: str, force: bool = False) -> None:
        if self.progress is None:
            return
        now = time.perf_counter()
        if not force and now - self.last_progress < _PROGRESS_INTERVAL_S:
            return
        self.last_progress = now
        try:
            self.progress(max(0.0, min(1.0, fraction)), best, message)
        except Exception:
            pass


def _validate(model: ObjectiveModel, locks: Dict[int, int], sizes: List[int]) -> None:
    n, T = model.n, model.n_teams
    if T == 0 and n:
        raise ValueError("No teams to allocate students into")
    if sum(sizes) != n:
        raise ValueError(f"Team sizes sum to {sum(sizes)} but there are {n} students")
    per_team = [0] * T
    for i, t in locks.items():
        if not 0 <= i < n:
            raise ValueError(f"Locked student index {i} is out of range")
        if not 0 <= t < T:
            raise ValueError(f"Lock of student index {i} refers to team {t + 1}, but there "
                             f"are only {T} teams")
        per_team[t] += 1
    for t, k in enumerate(per_team):
        if k > sizes[t]:
            raise ValueError(f"{k} students are locked to team {t + 1}, which has only "
                             f"{sizes[t]} places")


_CONSTRUCT_CHECK_EVERY = 64


def construct(model: ObjectiveModel, locks: Dict[int, int], rng: random.Random,
              sizes: Optional[Sequence[int]] = None,
              stopper: Optional["_Stopper"] = None) -> SearchState:
    sizes = list(model.team_sizes if sizes is None else sizes)
    assignment = [-1] * model.n
    for i, t in locks.items():
        assignment[i] = t
    state = SearchState(model, assignment, locks)
    capacity = [sizes[t] - state.sizes[t] for t in range(model.n_teams)]
    order = [i for i in range(model.n) if assignment[i] < 0]
    rng.shuffle(order)
    for pos, i in enumerate(order):
        if stopper is not None and pos % _CONSTRUCT_CHECK_EVERY == 0 and stopper.check():
            for j in order[pos:]:
                t = next((u for u in range(model.n_teams) if capacity[u] > 0), -1)
                state.apply_add(j, t)
                if t >= 0:
                    capacity[t] -= 1
            break
        best_t, best_d = -1, math.inf
        for t in range(model.n_teams):
            if capacity[t] <= 0:
                continue
            d = state.delta_add(i, t)
            if d < best_d - 1e-12:
                best_t, best_d = t, d
        state.apply_add(i, best_t)
        capacity[best_t] -= 1
    return state


def _calibrate(state: SearchState, free: List[int], rng: random.Random, flexible: bool,
               bounds: Tuple[int, int]) -> Tuple[float, int]:
    assign = state.assignment
    nf = len(free)
    T = state.T
    worse: List[float] = []
    evals = 0
    for _ in range(_CALIBRATION_SAMPLES):
        if flexible and rng.random() < _MOVE_SHARE:
            i = free[int(rng.random() * nf)]
            t = int(rng.random() * T)
            A = assign[i]
            if t == A or state.sizes[A] <= bounds[0] or state.sizes[t] >= bounds[1]:
                continue
            d = state.delta_move(i, t)
        else:
            i = free[int(rng.random() * nf)]
            j = free[int(rng.random() * nf)]
            if assign[i] == assign[j]:
                continue
            d = state.delta_swap(i, j)
        evals += 1
        if d > 1e-12:
            worse.append(d)
    soft = [d for d in worse if d < state.model.hard_penalty * 0.5]
    pool = sorted(soft or worse)
    if not pool:
        return 1.0, evals
    return pool[len(pool) // 2], evals


def _anneal(state: SearchState, free: List[int], rng: random.Random, budget: int,
            t_start: float, t_end: float, flexible: bool, bounds: Tuple[int, int],
            stopper: _Stopper, stats: Dict[str, Any], best_global: Optional[List[float]],
            fraction: Callable[[float], float]) -> Tuple[List[int], float]:
    track = best_global if best_global is not None else [math.inf]
    assign = state.assignment
    sizes = state.sizes
    nf = len(free)
    T = state.T
    lo, hi = bounds
    best = list(assign)
    best_obj = state.objective
    temp = t_start
    alpha = (t_end / t_start) ** (1.0 / budget) if budget > 0 and t_start > 0 else 1.0
    rand = rng.random
    exp = math.exp
    delta_swap = state.delta_swap
    apply_swap = state.apply_swap
    move_share = _MOVE_SHARE if flexible else 0.0
    focus_share = _FOCUS_SHARE if state.fw > 0.0 and T > 2 else 0.0
    locked = state.locks
    members = state.members
    hard_keys = _hard_group_keys(state.model)
    pair_share = _PAIR_SWAP_SHARE if hard_keys is not None else 0.0
    by_key: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {}
    if pair_share:
        for x in free:
            by_key.setdefault(hard_keys[x], []).append(x)
    evaluations = accepted = improvements = done = 0
    for k in range(budget):
        if k & (_CHECK_EVERY - 1) == 0 and k:
            if stopper.check():
                break
            if best_global is not None:
                stopper.report(fraction(k / budget), best_global[0], "annealing")
        done = k + 1
        temp *= alpha
        if pair_share and rand() < pair_share:
            if rand() < 0.5:
                pool = members[state.top[0]]
                i = pool[int(rand() * len(pool))] if pool else -1
                if i < 0 or i in locked:
                    continue
            else:
                i = free[int(rand() * nf)]
            j = free[int(rand() * nf)]
            A = assign[i]
            if A == assign[j] or hard_keys[i] == hard_keys[j]:
                continue
            kk = _partner(members[A], i, hard_keys, locked, rand)
            if kk < 0:
                continue
            same = by_key[hard_keys[j]]
            ll = same[int(rand() * len(same))]
            if ll == j or assign[ll] == A:
                continue
            d = delta_swap(i, j)
            apply_swap(i, j)
            d += delta_swap(kk, ll)
            evaluations += 2
            if d > 0.0 and rand() >= exp(-d / temp):
                apply_swap(i, j)
                continue
            apply_swap(kk, ll)
        elif move_share and rand() < move_share:
            i = free[int(rand() * nf)]
            t = int(rand() * T)
            A = assign[i]
            if t == A or sizes[A] <= lo or sizes[t] >= hi:
                continue
            d = state.delta_move(i, t)
            evaluations += 1
            if d > 0.0 and rand() >= exp(-d / temp):
                continue
            state.apply_move(i, t)
        else:
            if focus_share and rand() < focus_share:
                pool = members[state.top[0]]
                if not pool:
                    continue
                i = pool[int(rand() * len(pool))]
                if i in locked:
                    continue
            else:
                i = free[int(rand() * nf)]
            j = free[int(rand() * nf)]
            if assign[i] == assign[j]:
                continue
            d = delta_swap(i, j)
            evaluations += 1
            if d > 0.0 and rand() >= exp(-d / temp):
                continue
            apply_swap(i, j)
        accepted += 1
        obj = state.objective
        if obj < best_obj - _IMPROVE_EPS:
            best_obj = obj
            best = list(assign)
            improvements += 1
            if best_global is not None and obj < track[0] - _IMPROVE_EPS:
                track[0] = obj
                history = stats["best_history"]
                if len(history) < 2000:
                    history.append((round(stopper.elapsed(), 4), obj))
    stats["iterations"] += done
    stats["evaluations"] += evaluations
    stats["accepted"] += accepted
    stats["improvements"] += improvements
    return best, best_obj


def _hard_group_keys(model: ObjectiveModel) -> Optional[List[Tuple[Tuple[int, ...], ...]]]:
    rules = [r for r, g in enumerate(model.group_rules)
             if g.hard and g.family in ("count", "isolation")]
    if not rules:
        return None
    codes = model.group_codes
    return [tuple(codes[i][r] for r in rules) for i in range(model.n)]


def _partner(team: List[int], i: int, keys: List[Tuple[Tuple[int, ...], ...]],
             locked: Dict[int, int], rand: Callable[[], float]) -> int:
    key = keys[i]
    options = [m for m in team if m != i and keys[m] == key and m not in locked]
    if not options:
        return -1
    return options[int(rand() * len(options))]


def _anneal_stages(model: ObjectiveModel, state: SearchState, locks: Dict[int, int],
                   free: List[int], rng: random.Random, budget: int, warm: bool,
                   flexible: bool, bounds: Tuple[int, int], stopper: _Stopper,
                   stats: Dict[str, Any], best_global: List[float],
                   fraction: Callable[[float], float]) -> SearchState:
    median, evals = _calibrate(state, free, rng, flexible, bounds)
    stats["evaluations"] += evals
    p_start = _P_START_WARM if warm else _P_START_COLD
    t_start = median / math.log(1.0 / p_start)
    t_end = t_start * _T_END_FRACTION
    relaxed_penalty = _RELAX_FACTOR * median
    stages: List[Tuple[int, ObjectiveModel]] = []
    if (_RELAX_SHARE > 0 and not warm and model.has_hard_team_rules
            and 0 < relaxed_penalty < model.hard_penalty):
        first = int(budget * _RELAX_SHARE)
        ratio = model.hard_penalty / relaxed_penalty
        per_step = first // _RELAX_STEPS
        for step in range(_RELAX_STEPS):
            steps = per_step if step < _RELAX_STEPS - 1 else first - per_step * step
            penalty = relaxed_penalty * ratio ** (step / _RELAX_STEPS)
            stages.append((steps, model.with_relaxed_team_rules(penalty)))
        stages.append((budget - first, model))
    else:
        stages.append((budget, model))
    done = 0
    best: List[int] = list(state.assignment)
    best_obj = state.objective
    for steps, stage_model in stages:
        if steps <= 0 or stopper.stopped:
            continue
        if stage_model is not state.model:
            state = SearchState(stage_model, state.assignment, locks)
        seg_start = t_start * (t_end / t_start) ** (done / budget)
        seg_end = t_start * (t_end / t_start) ** ((done + steps) / budget)
        offset = done

        def seg_fraction(f: float, offset: int = offset, steps: int = steps) -> float:
            return fraction((offset + f * steps) / budget)

        full = stage_model is model
        run_best, run_obj = _anneal(state, free, rng, steps, seg_start, seg_end, flexible,
                                    bounds, stopper, stats, best_global if full else None,
                                    seg_fraction)
        done += steps
        if full and run_obj < best_obj - _IMPROVE_EPS:
            best, best_obj = run_best, run_obj
    if state.model is not model:
        state = SearchState(model, state.assignment, locks)
    if best_obj < state.objective - _IMPROVE_EPS:
        state = SearchState(model, best, locks)
    return state


def _descend(state: SearchState, free: List[int], rng: random.Random, flexible: bool,
             bounds: Tuple[int, int], stopper: _Stopper, stats: Dict[str, Any],
             max_evaluations: Optional[int] = None) -> None:
    assign = state.assignment
    T = state.T
    lo, hi = bounds
    limit = math.inf if max_evaluations is None else max(0, int(max_evaluations))
    evaluations = 0
    improved = True
    while improved and not stopper.stopped and evaluations < limit:
        improved = False
        order = list(free)
        rng.shuffle(order)
        for a_pos, i in enumerate(order):
            for j in order[a_pos + 1:]:
                if assign[i] == assign[j]:
                    continue
                evaluations += 1
                if state.delta_swap(i, j) < -_IMPROVE_EPS:
                    state.apply_swap(i, j)
                    improved = True
            if evaluations >= limit:
                break
            if evaluations > 4096 and a_pos % 16 == 0 and stopper.check():
                break
        if flexible and not stopper.stopped and evaluations < limit:
            for i in order:
                for t in range(T):
                    A = assign[i]
                    if t == A or state.sizes[A] <= lo or state.sizes[t] >= hi:
                        continue
                    evaluations += 1
                    if state.delta_move(i, t) < -_IMPROVE_EPS:
                        state.apply_move(i, t)
                        improved = True
                if evaluations >= limit:
                    break
    stats["evaluations"] += evaluations


def optimize(model: ObjectiveModel, *, settings: SolverSettings,
             initial: Optional[Sequence[int]] = None,
             locks: Optional[Dict[int, int]] = None,
             size_bounds: Optional[Tuple[int, int]] = None,
             progress: Optional[ProgressFn] = None,
             cancel: Optional[CancelFn] = None,
             time_limit_s: Optional[float] = None,
             extra_start: Optional[Sequence[int]] = None) -> SolveResult:
    n, T = model.n, model.n_teams
    locks = {int(i): int(t) for i, t in (locks or {}).items()}
    sizes = list(model.team_sizes)
    _validate(model, locks, sizes)
    flexible = bool(settings.flexible_sizes)
    if size_bounds is None:
        size_bounds = (min(sizes), max(sizes)) if sizes else (1, 1)
    bounds = (max(1, int(size_bounds[0])), max(1, int(size_bounds[1])))
    initial = _checked_start(initial, n, T, sizes, flexible, locks, "Initial")
    extra = _checked_start(extra_start, n, T, sizes, flexible, locks, "Extra start")

    limit = effective_time_limit(settings, n) if time_limit_s is None else float(time_limit_s)
    stopper = _Stopper(limit, progress, cancel)
    budget = settings.iterations if settings.iterations is not None else auto_iterations(n, T)
    budget = max(0, int(budget))
    restarts = max(1, int(settings.restarts))
    warm: Dict[int, List[int]] = {}
    if initial is not None:
        warm[0] = initial
    if extra is not None:
        warm[restarts] = extra
        restarts += 1
    free = [i for i in range(n) if i not in locks]
    stats: Dict[str, Any] = {
        "restarts": 0, "iterations": 0, "evaluations": 0, "accepted": 0, "improvements": 0,
        "elapsed_s": 0.0, "time_limited": False, "cancelled": False, "best_history": [],
        "per_restart": [], "budget": budget, "polish_budget": polish_evaluations(n),
        "seeds": [], "workers": 1, "parallel": False, "restarts_planned": restarts,
        "restarts_completed": 0, "best_restart": None, "warm_start_ran": False,
        "warm_start_won": False,
        "time_limit_s": round(limit, 3),
        "workers_requested": int(getattr(settings, "workers", 1) or 1), "warnings": [],
    }
    if n == 0 or T == 0:
        return SolveResult(assignment=[], objective=0.0, hard_units=0, stats=stats)

    plan = _RestartPlan(settings=settings, warm=warm, locks=locks, free=free,
                        budget=budget, flexible=flexible, bounds=bounds, restarts=restarts,
                        polish_budget=polish_evaluations(n),
                        extra=restarts - 1 if extra is not None else None)
    seeds = [plan.seed(r) for r in range(restarts)]
    outcomes: Dict[int, Tuple[List[int], float]] = {}
    workers = _worker_count(settings, plan)
    if workers > 1:
        try:
            _parallel_restarts(model, plan, workers, stopper, stats, outcomes)
            stats["workers"] = workers
            stats["parallel"] = True
        except Exception as exc:
            stats["warnings"].append(
                f"Parallel restarts unavailable ({type(exc).__name__}: {exc}); "
                f"ran the remaining restarts in-process")
    best_global = [min((obj for _, obj in outcomes.values()), default=math.inf)]
    for r in plan.order():
        if r in outcomes:
            continue
        if stopper.stopped and outcomes:
            break
        assignment, obj = _restart(model, plan, r, stopper, stats, best_global)
        outcomes[r] = (assignment, obj)
        stopper.report(len(outcomes) / restarts, best_global[0],
                       f"restart {len(outcomes)}/{restarts} done", force=True)
        if stopper.stopped:
            break

    best_assignment: Optional[List[int]] = None
    best_obj = math.inf
    for r in sorted(outcomes):
        assignment, obj = outcomes[r]
        stats["per_restart"].append(obj)
        stats["seeds"].append(seeds[r])
        stats["restarts"] += 1
        if obj < best_obj - _IMPROVE_EPS or best_assignment is None:
            best_obj = obj
            best_assignment = assignment
            stats["best_restart"] = r

    assert best_assignment is not None
    objective = model.evaluate(best_assignment)
    stats["restarts_completed"] = stats["restarts"]
    stats["warm_start_ran"] = extra is not None and restarts - 1 in outcomes
    stats["warm_start_won"] = extra is not None and stats["best_restart"] == restarts - 1
    stats["elapsed_s"] = round(stopper.elapsed(), 4)
    stats["time_limited"] = stopper.time_limited
    stats["cancelled"] = stopper.cancelled
    return SolveResult(
        assignment=best_assignment,
        objective=objective,
        hard_units=model.assignment_hard_units(best_assignment),
        stats=stats,
    )


def _checked_start(start: Optional[Sequence[int]], n: int, T: int, sizes: List[int],
                   flexible: bool, locks: Dict[int, int], what: str) -> Optional[List[int]]:
    if start is None:
        return None
    start = [int(t) for t in start]
    if len(start) != n or any(not 0 <= t < T for t in start):
        raise ValueError(f"{what} allocation does not match the cohort and team count")
    counts = [0] * T
    for t in start:
        counts[t] += 1
    if not flexible and counts != sizes:
        raise ValueError(f"{what} allocation does not have the planned team sizes")
    for i, t in locks.items():
        if start[i] != t:
            raise ValueError(f"{what} allocation moves locked student index {i}")
    return start


@dataclass
class _RestartPlan:
    settings: SolverSettings
    warm: Dict[int, List[int]]
    locks: Dict[int, int]
    free: List[int]
    budget: int
    flexible: bool
    bounds: Tuple[int, int]
    restarts: int
    polish_budget: Optional[int] = None
    extra: Optional[int] = None

    def seed(self, r: int) -> Optional[int]:
        return restart_seed(self.settings.seed, r)

    def order(self) -> List[int]:
        rest = [r for r in range(self.restarts) if r != self.extra]
        return rest if self.extra is None else [self.extra] + rest


def _restart(model: ObjectiveModel, plan: _RestartPlan, r: int, stopper: _Stopper,
             stats: Dict[str, Any], best_global: List[float]) -> Tuple[List[int], float]:
    restarts = plan.restarts
    slot = plan.order().index(r)
    rng = random.Random(plan.seed(r))
    start = plan.warm.get(r)
    warm = start is not None
    locks = plan.locks
    state = (SearchState(model, start, locks) if warm
             else construct(model, locks, rng, stopper=stopper))
    if state.objective < best_global[0]:
        best_global[0] = state.objective
        stats["best_history"].append((round(stopper.elapsed(), 4), state.objective))

    def fraction(f: float) -> float:
        return (slot + f) / restarts

    can_search = len(plan.free) >= 2 and model.n_teams >= 2
    if can_search and plan.budget > 0 and not stopper.stopped:
        state = _anneal_stages(model, state, locks, plan.free, rng, plan.budget, warm,
                               plan.flexible, plan.bounds, stopper, stats, best_global, fraction)
    if plan.settings.polish and can_search and not stopper.stopped:
        stopper.report(fraction(1.0), best_global[0], "polishing", force=True)
        _descend(state, plan.free, rng, plan.flexible, plan.bounds, stopper, stats,
                 plan.polish_budget)
    state.resync()
    obj = state.objective
    if obj < best_global[0] - _IMPROVE_EPS:
        best_global[0] = obj
        stats["best_history"].append((round(stopper.elapsed(), 4), obj))
    return list(state.assignment), obj


def _worker_count(settings: SolverSettings, plan: _RestartPlan) -> int:
    wanted = int(getattr(settings, "workers", 1) or 1)
    if wanted <= 1 or plan.restarts <= 1 or plan.budget <= 0 or len(plan.free) < 2:
        return 1
    return max(1, min(wanted, plan.restarts, os.cpu_count() or 1))


_WORKER_STOP: Any = None
_WORKER_MODEL: Tuple[Optional[str], Optional[ObjectiveModel]] = (None, None)


def _init_worker(stop_event: Any) -> None:
    global _WORKER_STOP
    _WORKER_STOP = stop_event


def _restart_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER_MODEL
    token, model = _WORKER_MODEL
    if token != payload["token"] or model is None:
        model = model_from_args(payload["model"])
        _WORKER_MODEL = (payload["token"], model)
    plan: _RestartPlan = payload["plan"]
    stop = _WORKER_STOP
    stopper = _Stopper(payload["time_limit_s"], None,
                       (lambda: stop.is_set()) if stop is not None else None)
    stats: Dict[str, Any] = {key: 0 for key in _COUNTERS}
    stats["best_history"] = []
    assignment, obj = _restart(model, plan, payload["restart"], stopper, stats, [math.inf])
    return {"assignment": assignment, "objective": obj,
            "counters": {key: stats[key] for key in _COUNTERS},
            "time_limited": stopper.time_limited, "cancelled": stopper.cancelled}


def _parallel_restarts(model: ObjectiveModel, plan: _RestartPlan, workers: int,
                       stopper: _Stopper, stats: Dict[str, Any],
                       outcomes: Dict[int, Tuple[List[int], float]]) -> None:
    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    token = uuid.uuid4().hex
    model_args = model.rebuild_args()
    best = min((obj for _, obj in outcomes.values()), default=math.inf)
    order = plan.order()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(stop_event,)) as pool:
        pending: Dict[Any, int] = {}
        next_r = 0
        try:
            while next_r < len(order) or pending:
                while (next_r < len(order) and len(pending) < workers
                       and not stopper.stopped):
                    payload = {"token": token, "model": model_args, "plan": plan,
                               "restart": order[next_r],
                               "time_limit_s": max(0.0, stopper.deadline - time.perf_counter())}
                    pending[pool.submit(_restart_worker, payload)] = order[next_r]
                    next_r += 1
                if not pending:
                    break
                done, _ = wait(list(pending), timeout=_POOL_POLL_S,
                               return_when=FIRST_COMPLETED)
                for future in done:
                    r = pending.pop(future)
                    result = future.result()
                    outcomes[r] = (list(result["assignment"]), float(result["objective"]))
                    for key in _COUNTERS:
                        stats[key] += result["counters"][key]
                    stopper.time_limited = stopper.time_limited or result["time_limited"]
                    if result["objective"] < best - _IMPROVE_EPS:
                        best = result["objective"]
                        stats["best_history"].append((round(stopper.elapsed(), 4), best))
                    stopper.report(len(outcomes) / plan.restarts, best,
                                   f"restart {len(outcomes)}/{plan.restarts} done", force=True)
                if stopper.check():
                    stop_event.set()
        finally:
            stop_event.set()
