from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .policy import (
    AttributeRule,
    CountRule,
    IsolationRule,
    NumericBoundsRule,
    PairRule,
    Policy,
)
from .roster import EncodedAttribute, EncodedCohort

MAX_PAIR_WEIGHT_ROWS = 80_000

MAX_PREFERENCE_ROWS = 20_000

MAX_WEIGHT_CLIQUES = 2_000

NUMERIC_BREAKPOINTS: Tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_AUX_COUNT_MIN_MEMBERS = 3

MAX_RANGE_EXCLUSION_PAIRS = 3_000

MAX_SHARED_VARS = 20_000

_INF = math.inf


def _segment_points() -> List[float]:
    pos = list(NUMERIC_BREAKPOINTS)
    return [-p for p in reversed(pos)] + [0.0] + pos


def numeric_pwl(u: float) -> float:
    pts = _segment_points()
    return max(-a * b + (a + b) * u for a, b in zip(pts, pts[1:]))


def categorical_balance_expectation(size: int, pool_counts: Sequence[int],
                                    pool_size: int) -> float:
    N = pool_size
    if N <= 1 or size <= 0:
        return 0.0
    return sum(
        size * (k / N) * (1 - k / N) * (N - size) / (N - 1) for k in pool_counts if k > 0
    )


def categorical_diversify_expectation(size: int, pool_counts: Sequence[int],
                                      pool_size: int) -> float:
    N = pool_size
    if N <= 1 or size <= 1:
        return 0.0
    pairs = size * (size - 1) / 2
    return sum(pairs * k * (k - 1) / (N * (N - 1)) for k in pool_counts if k > 1)


@dataclass
class HardSpec:
    sizes: List[int]
    locks: Dict[int, int] = field(default_factory=dict)
    forbidden: Set[Tuple[int, int]] = field(default_factory=set)
    together: List[List[int]] = field(default_factory=list)
    counts: List[Tuple[str, np.ndarray, Optional[int], Optional[int]]] = field(default_factory=list)
    isolation: List[Tuple[str, np.ndarray, int]] = field(default_factory=list)
    bounds: List[Tuple[str, np.ndarray, np.ndarray, Optional[float], Optional[float],
                       Optional[float]]] = field(default_factory=list)
    shared: List[Tuple[str, List[Set[int]], int]] = field(default_factory=list)


@dataclass
class MilpModel:
    n: int
    n_teams: int
    c: np.ndarray
    A: sparse.csr_matrix
    row_lo: np.ndarray
    row_hi: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray
    hard: HardSpec
    notes: List[str] = field(default_factory=list)
    infeasible_reasons: List[str] = field(default_factory=list)
    build_s: float = 0.0
    has_worst_team: bool = False

    @property
    def n_vars(self) -> int:
        return int(self.c.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_integer(self) -> int:
        return int(self.integrality.sum())

    def describe(self) -> str:
        return (f"{self.n_vars} variables ({self.n_integer} integer), "
                f"{self.n_rows} constraints, {self.A.nnz} non-zeros")


class _Builder:
    def __init__(self) -> None:
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.integer: List[int] = []
        self.cost: List[float] = []
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []
        self.lo: List[float] = []
        self.hi: List[float] = []

    def add_vars(self, count: int, lb: float = 0.0, ub: float = _INF, integer: bool = False) -> int:
        first = len(self.lb)
        self.lb.extend([lb] * count)
        self.ub.extend([ub] * count)
        self.integer.extend([1 if integer else 0] * count)
        self.cost.extend([0.0] * count)
        return first

    def add_var(self, lb: float = 0.0, ub: float = _INF, integer: bool = False) -> int:
        return self.add_vars(1, lb, ub, integer)

    def add_row(self, cols: Sequence[int], vals: Sequence[float], lo: float, hi: float) -> None:
        r = len(self.lo)
        self.rows.extend([r] * len(cols))
        self.cols.extend(cols)
        self.vals.extend(vals)
        self.lo.append(lo)
        self.hi.append(hi)

    @property
    def n_vars(self) -> int:
        return len(self.lb)


def validate_inputs(
    cohort: EncodedCohort,
    team_sizes: Sequence[int],
    forbidden_pairs: Iterable[Tuple[int, int]],
    pair_weights: Optional[Dict[Tuple[int, int], float]],
    locks: Optional[Dict[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float], Dict[int, int]]:
    n, T = cohort.n, len(team_sizes)
    sizes = [int(s) for s in team_sizes]
    if any(s < 0 for s in sizes):
        raise ValueError(f"Team sizes must be non-negative, got {sizes}")
    if sum(sizes) != n:
        raise ValueError(f"Team sizes sum to {sum(sizes)} but the cohort has {n} students")

    def check_index(i: int, what: str) -> int:
        i = int(i)
        if not 0 <= i < n:
            raise ValueError(f"{what}: student index {i} is out of range 0..{n - 1}")
        return i

    norm_locks: Dict[int, int] = {}
    for i, t in (locks or {}).items():
        i = check_index(i, "Lock")
        t = int(t)
        if not 0 <= t < T:
            raise ValueError(f"Lock: team index {t} for student {i} is out of range 0..{T - 1}")
        norm_locks[i] = t

    forbidden: List[Tuple[int, int]] = []
    for pair in forbidden_pairs or ():
        a, b = (check_index(v, "Forbidden pair") for v in pair)
        if a != b:
            forbidden.append((min(a, b), max(a, b)))

    weights: Dict[Tuple[int, int], float] = {}
    for pair, w in (pair_weights or {}).items():
        a, b = (check_index(v, "Pair weight") for v in pair)
        if a != b:
            key = (min(a, b), max(a, b))
            weights[key] = weights.get(key, 0.0) + float(w)
    return forbidden, weights, norm_locks


def _category_members(attr: EncodedAttribute) -> List[np.ndarray]:
    k = len(attr.categories)
    if attr.kind == "categorical":
        codes = np.asarray(attr.codes, dtype=np.int64)
        return [np.flatnonzero(codes == c) for c in range(k)]
    buckets: List[List[int]] = [[] for _ in range(k)]
    for i, items in enumerate(attr.multi):
        for c in set(items):
            if 0 <= c < k:
                buckets[c].append(i)
    return [np.asarray(b, dtype=np.int64) for b in buckets]


def _clique_cover(edges: Set[Tuple[int, int]]) -> List[List[int]]:
    adj: Dict[int, Set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    uncovered = set(edges)
    cliques: List[List[int]] = []
    for edge in sorted(edges):
        if edge not in uncovered:
            continue
        a, b = edge
        clique = [a, b]
        candidates = (adj[a] & adj[b]) - {a, b}
        while candidates:
            best = max(sorted(candidates),
                       key=lambda v: sum((min(u, v), max(u, v)) in uncovered for u in clique))
            clique.append(best)
            candidates &= adj[best]
            candidates.discard(best)
        for u, v in combinations(clique, 2):
            uncovered.discard((min(u, v), max(u, v)))
        cliques.append(sorted(clique))
    return cliques


def decompose_pair_weights(
    weights: Dict[Tuple[int, int], float],
    wildcards: Iterable[Tuple[int, int]] = (),
    max_cliques: int = MAX_WEIGHT_CLIQUES,
    tol: float = 1e-9,
) -> Tuple[List[Tuple[List[int], float]], Dict[Tuple[int, int], float]]:
    remaining = {k: float(w) for k, w in weights.items() if w > tol}
    free = {(min(a, b), max(a, b)) for a, b in wildcards} - set(remaining)
    adj: Dict[int, Set[int]] = {}
    for a, b in list(remaining) + sorted(free):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def weight(u: int, v: int) -> float:
        key = (u, v) if u < v else (v, u)
        return math.inf if key in free else remaining[key]

    cliques: List[Tuple[List[int], float]] = []
    for edge in sorted(remaining, key=lambda e: (-remaining[e], e)):
        while edge in remaining and len(cliques) < max_cliques:
            a, b = edge
            clique = [a, b]
            candidates = (adj[a] & adj[b]) - {a, b}
            while candidates:
                best = max(sorted(candidates), key=lambda v: min(weight(u, v) for u in clique))
                clique.append(best)
                candidates &= adj[best]
                candidates.discard(best)
            if len(clique) < 3:
                break
            w = min(weight(u, v) for u, v in combinations(clique, 2))
            for u, v in combinations(clique, 2):
                key = (u, v) if u < v else (v, u)
                if key in free:
                    continue
                remaining[key] -= w
                if remaining[key] <= tol:
                    del remaining[key]
                    # INVARIANT: a pair that is fully paid leaves the adjacency, and
                    # later cliques must not charge it again.
                    adj[u].discard(v)
                    adj[v].discard(u)
            cliques.append((sorted(clique), w))
    return cliques, remaining


class _ModelFactory:
    def __init__(self, cohort: EncodedCohort, policy: Policy, sizes: List[int],
                 forbidden: List[Tuple[int, int]], weights: Dict[Tuple[int, int], float],
                 locks: Dict[int, int], project_id: Optional[str],
                 symmetry_breaking: bool) -> None:
        self.cohort = cohort
        self.policy = policy
        self.sizes = sizes
        self.n = cohort.n
        self.T = len(sizes)
        self.forbidden_in = forbidden
        self.weights_in = weights
        self.locks = locks
        self.project_id = None if project_id is None else str(project_id)
        self.symmetry_breaking = symmetry_breaking
        self.b = _Builder()
        self.notes: List[str] = []
        self.reasons: List[str] = []
        self.hard = HardSpec(sizes=list(sizes), locks=dict(locks))
        self.team_terms: List[List[Tuple[int, float]]] = [[] for _ in sizes]
        self.per_team_pairs = policy.fairness.worst_team_weight > 0
        self.has_worst_team = False
        self._count_cache: Dict[Tuple[int, bytes], int] = {}
        self.range_conflicts: Set[Tuple[int, int]] = set()

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def x_cols(self, members: np.ndarray, t: int) -> List[int]:
        return (members * self.T + t).tolist()

    def count_expr(self, members: np.ndarray, t: int, uses: int) -> Tuple[List[int], List[float]]:
        key = (t, members.tobytes())
        cached = self._count_cache.get(key)
        if cached is not None:
            return [cached], [1.0]
        if uses <= 1 or len(members) < _AUX_COUNT_MIN_MEMBERS:
            return self.x_cols(members, t), [1.0] * len(members)
        col = self.b.add_var(0.0, float(min(self.sizes[t], len(members))))
        cols = self.x_cols(members, t)
        self.b.add_row(cols + [col], [1.0] * len(cols) + [-1.0], 0.0, 0.0)
        self._count_cache[key] = col
        return [col], [1.0]

    def team_cost(self, t: int, col: int, coef: float) -> None:
        if coef != 0.0:
            self.team_terms[t].append((col, coef))

    def convex_count_term(self, members: np.ndarray, t: int, f: Sequence[float],
                          coef: float) -> None:
        lines = []
        for k in range(len(f) - 1):
            slope = f[k + 1] - f[k]
            intercept = f[k] - slope * k
            if slope == 0.0 and intercept <= 0.0:
                continue
            lines.append((slope, intercept))
        if not lines:
            return
        z = self.b.add_var(0.0)
        cols, vals = self.count_expr(members, t, len(lines))
        for slope, intercept in lines:
            self.b.add_row([z] + cols, [1.0] + [-slope * v for v in vals], intercept, _INF)
        self.team_cost(t, z, coef)

    def add_assignment(self) -> None:
        n, T, b = self.n, self.T, self.b
        b.add_vars(n * T, 0.0, 1.0, integer=True)
        for i in range(n):
            b.add_row(list(range(i * T, i * T + T)), [1.0] * T, 1.0, 1.0)
        for t, s in enumerate(self.sizes):
            b.add_row(list(range(t, n * T, T)), [1.0] * n, float(s), float(s))
        locked_per_team = [0] * T
        for i, t in self.locks.items():
            for u in range(T):
                if u == t:
                    b.lb[i * T + u] = 1.0
                else:
                    b.ub[i * T + u] = 0.0
            locked_per_team[t] += 1
        for t, k in enumerate(locked_per_team):
            if k > self.sizes[t]:
                self.reasons.append(f"{k} students are locked to team {t + 1}, which has only "
                                    f"{self.sizes[t]} places")

    def add_symmetry_breaking(self) -> None:
        if not self.symmetry_breaking:
            return
        locked_teams = set(self.locks.values())
        groups: Dict[int, List[int]] = {}
        for t, s in enumerate(self.sizes):
            if t not in locked_teams and s > 0:
                groups.setdefault(s, []).append(t)
        free = [i for i in range(self.n) if i not in self.locks]
        for teams in groups.values():
            if len(teams) < 2:
                continue
            for rank, i in enumerate(free):
                if rank + 1 >= len(teams):
                    break
                for t in teams[rank + 1:]:
                    self.b.ub[i * self.T + t] = 0.0

    def add_attribute_rules(self) -> None:
        for rule in self.policy.active_attributes():
            attr = self.cohort.attributes.get(rule.attribute)
            if attr is None:
                self.note(f"attribute rule '{rule.attribute}' skipped (not an encoded attribute)")
                continue
            if attr.kind in ("categorical", "multi"):
                if rule.goal == "balance":
                    self.categorical_goal(rule, attr, balance=True)
                elif rule.goal == "diversify":
                    self.categorical_goal(rule, attr, balance=False)
                elif rule.goal == "cluster":
                    self.note(f"'{rule.attribute}' cluster goal not modelled (concave); "
                              f"left to local search")
                else:
                    self.note(f"'{rule.attribute}': goal '{rule.goal}' does not fit a "
                              f"{attr.kind} attribute; skipped")
            elif attr.kind == "numeric":
                if rule.goal in ("balance", "spread"):
                    self.numeric_balance(rule, attr)
                    if rule.goal == "spread":
                        self.note(f"'{rule.attribute}' spread: only the mean-balance part is "
                                  f"modelled (within-team dispersion left to local search)")
                elif rule.goal == "cluster":
                    self.note(f"'{rule.attribute}' cluster goal not modelled (non-convex "
                              f"quadratic); left to local search")
                else:
                    self.note(f"'{rule.attribute}': goal '{rule.goal}' does not fit a numeric "
                              f"attribute; skipped")

    def categorical_goal(self, rule: AttributeRule, attr: EncodedAttribute, balance: bool) -> None:
        N = self.n
        members = _category_members(attr)
        counts = [len(m) for m in members]
        for t, s in enumerate(self.sizes):
            if balance:
                expected = categorical_balance_expectation(s, counts, N)
            else:
                expected = categorical_diversify_expectation(s, counts, N)
            if expected <= 1e-12:
                continue
            coef = rule.weight / expected
            for mem, k in zip(members, counts):
                if k == 0 or (not balance and k < 2):
                    continue
                top = min(s, k)
                if balance:
                    share = s * k / N
                    f = [(v - share) ** 2 for v in range(top + 1)]
                else:
                    f = [v * (v - 1) / 2 for v in range(top + 1)]
                self.convex_count_term(mem, t, f, coef)

    def numeric_balance(self, rule: AttributeRule, attr: EncodedAttribute) -> None:
        vals = np.asarray(attr.values, dtype=float)
        present = np.flatnonzero(~np.isnan(vals))
        Np, N = len(present), self.n
        if Np < 2:
            return
        mu = float(vals[present].mean())
        var = float(vals[present].var())
        if var <= 1e-12:
            return
        if Np < N:
            self.note(f"'{rule.attribute}' balance: team present-count approximated by its "
                      f"expectation ({N - Np} missing values)")
        dev = vals[present] - mu
        pts = _segment_points()
        segments = [((a + b), -a * b) for a, b in zip(pts, pts[1:])]
        for t, s in enumerate(self.sizes):
            m = s * Np / N
            if m <= 0 or Np - m <= 1e-9:
                continue
            sd = math.sqrt(m * var * (Np - m) / (Np - 1))
            d = self.b.add_var(-_INF, _INF)
            cols = self.x_cols(present, t)
            self.b.add_row(cols + [d], (dev / sd).tolist() + [-1.0], 0.0, 0.0)
            z = self.b.add_var(0.0)
            for slope, intercept in segments:
                self.b.add_row([z, d], [1.0, -slope], intercept, _INF)
            self.team_cost(t, z, rule.weight)

    def count_groups(self, rule: CountRule) -> Optional[List[np.ndarray]]:
        attr = self.cohort.attributes.get(rule.attribute)
        name = rule.label or rule.attribute
        if attr is None:
            self.note(f"count rule '{name}' skipped (attribute not encoded)")
            return None
        if attr.kind == "numeric":
            vals = np.asarray(attr.values, dtype=float)
            if rule.numeric_range is not None:
                lo, hi = (list(rule.numeric_range) + [None, None])[:2]
                mask = ~np.isnan(vals)
                if lo is not None:
                    mask &= vals >= float(lo)
                if hi is not None:
                    mask &= vals <= float(hi)
                return [np.flatnonzero(mask)]
            targets = []
            for v in rule.values:
                try:
                    targets.append(float(v))
                except (TypeError, ValueError):
                    continue
            if not targets:
                self.note(f"count rule '{name}' skipped (numeric attribute needs a range)")
                return None
            groups = [np.flatnonzero(np.isclose(vals, v)) for v in targets]
            if rule.mode == "any_of":
                return [np.unique(np.concatenate(groups))]
            return groups
        if rule.numeric_range is not None:
            self.note(f"count rule '{name}' skipped (numeric range on a {attr.kind} attribute)")
            return None
        members = _category_members(attr)
        if rule.mode == "each":
            if not rule.values:
                return [m for m in members if len(m)]
            return [members[c] if c >= 0 else np.zeros(0, dtype=np.int64)
                    for c in (attr.code_of(v) for v in rule.values)]
        if not rule.values:
            self.note(f"count rule '{name}' skipped (no values listed)")
            return None
        codes = [c for c in (attr.code_of(v) for v in rule.values) if c >= 0]
        if not codes:
            return [np.zeros(0, dtype=np.int64)]
        return [np.unique(np.concatenate([members[c] for c in codes]))]

    def add_count_rules(self) -> None:
        for rule in self.policy.counts:
            if not rule.enabled or (rule.min_count is None and rule.max_count is None):
                continue
            if not rule.hard and rule.weight <= 0:
                continue
            groups = self.count_groups(rule)
            if groups is None:
                continue
            name = rule.label or rule.attribute
            lo = rule.min_count if rule.min_count is not None and rule.min_count > 0 else None
            hi = rule.max_count
            for g in groups:
                if rule.hard:
                    self.hard.counts.append((name, g, lo, hi))
                    self.check_count_feasible(name, g, lo, hi)
                for t, s in enumerate(self.sizes):
                    cap = min(s, len(g))
                    need_lo = lo is not None
                    need_hi = hi is not None and hi < cap
                    if not (need_lo or need_hi):
                        continue
                    uses = 1 if rule.hard else int(need_lo) + int(need_hi)
                    cols, vals = self.count_expr(g, t, uses)
                    if rule.hard:
                        self.b.add_row(cols, vals, float(lo) if need_lo else -_INF,
                                       float(hi) if need_hi else _INF)
                        continue
                    if need_lo:
                        u = self.b.add_var(0.0)
                        self.b.add_row(cols + [u], vals + [1.0], float(lo), _INF)
                        self.team_cost(t, u, rule.weight)
                    if need_hi:
                        u = self.b.add_var(0.0)
                        self.b.add_row(cols + [u], vals + [-1.0], -_INF, float(hi))
                        self.team_cost(t, u, rule.weight)

    def check_count_feasible(self, name: str, g: np.ndarray, lo: Optional[int],
                             hi: Optional[int]) -> None:
        k = len(g)
        if lo is not None:
            if k < lo * self.T:
                self.reasons.append(f"hard count rule '{name}' needs at least {lo} per team "
                                    f"({lo * self.T} in total) but only {k} students match")
            elif any(s < lo for s in self.sizes):
                self.reasons.append(f"hard count rule '{name}' needs at least {lo} per team "
                                    f"but some teams have only {min(self.sizes)} places")
        if hi is not None:
            room = sum(min(hi, s) for s in self.sizes)
            if k > room:
                self.reasons.append(f"hard count rule '{name}' allows at most {hi} per team "
                                    f"({room} in total) but {k} students match")

    def add_isolation_rules(self) -> None:
        for rule in self.policy.isolation:
            if not rule.enabled or (not rule.hard and rule.weight <= 0):
                continue
            name = rule.label or rule.attribute
            attr = self.cohort.attributes.get(rule.attribute)
            if attr is None or attr.kind not in ("categorical", "multi"):
                self.note(f"isolation rule '{name}' skipped (needs an encoded categorical "
                          f"attribute)")
                continue
            k = int(rule.min_together)
            if k <= 1:
                continue
            members = _category_members(attr)
            if rule.values:
                picked = [members[c] for c in (attr.code_of(v) for v in rule.values) if c >= 0]
            else:
                picked = list(members)
            for g in picked:
                if len(g) == 0:
                    continue
                if rule.hard:
                    self.hard.isolation.append((name, g, k))
                    if len(g) < k or max(self.sizes) < k:
                        self.reasons.append(
                            f"hard isolation rule '{name}': {len(g)} matching student(s) cannot "
                            f"form groups of at least {k} in teams of at most {max(self.sizes)}")
                for t, s in enumerate(self.sizes):
                    if s == 0:
                        continue
                    self.isolation_term(rule, g, t, k)

    def isolation_term(self, rule: IsolationRule, g: np.ndarray, t: int, k: int) -> None:
        cap = min(self.sizes[t], len(g))
        cols, vals = self.count_expr(g, t, 2)
        if rule.hard:
            if k > cap:
                self.b.add_row(cols, vals, 0.0, 0.0)
                return
            y = self.b.add_var(0.0, 1.0, integer=True)
            self.b.add_row(cols + [y], vals + [-float(cap)], -_INF, 0.0)
            self.b.add_row(cols + [y], vals + [-float(k)], 0.0, _INF)
            return
        y = self.b.add_var(0.0, 1.0, integer=True)
        u = self.b.add_var(0.0)
        self.b.add_row([u, y] + cols, [1.0, -float(k)] + vals, 0.0, _INF)
        self.b.add_row([u, y] + cols, [1.0, float(cap)] + [-v for v in vals],
                       0.0, _INF)
        self.team_cost(t, u, rule.weight)

    def add_bounds_rules(self) -> None:
        for rule in self.policy.numeric_bounds:
            if not rule.enabled or (not rule.hard and rule.weight <= 0):
                continue
            if rule.mean_min is None and rule.mean_max is None and rule.max_range is None:
                continue
            name = rule.label or rule.attribute
            attr = self.cohort.attributes.get(rule.attribute)
            if attr is None or attr.kind != "numeric":
                self.note(f"bounds rule '{name}' skipped (needs an encoded numeric attribute)")
                continue
            vals = np.asarray(attr.values, dtype=float)
            present = np.flatnonzero(~np.isnan(vals))
            if len(present) < 2:
                continue
            pv = vals[present]
            sd = float(pv.std())
            if sd <= 1e-12:
                continue
            lo = None if rule.mean_min is None else float(rule.mean_min)
            hi = None if rule.mean_max is None else float(rule.mean_max)
            width = None if rule.max_range is None else float(rule.max_range)
            if rule.hard:
                self.hard.bounds.append((name, present, pv, lo, hi, width))
            missing = self.n - len(present)
            if missing and not rule.hard and (lo is not None or hi is not None):
                self.note(f"bounds rule '{name}': team valued count approximated by its "
                          f"expectation ({missing} missing values)")
            for t, s in enumerate(self.sizes):
                if s == 0:
                    continue
                cols = self.x_cols(present, t)
                expected = s * len(present) / self.n
                for bound, sign in ((lo, 1.0), (hi, -1.0)):
                    if bound is None:
                        continue
                    coefs = (sign * (pv - bound)).tolist()
                    if rule.hard:
                        self.b.add_row(cols, coefs, 0.0, _INF)
                        continue
                    if expected <= 1e-9:
                        continue
                    e = self.b.add_var(0.0)
                    self.b.add_row(cols + [e], coefs + [expected * sd], 0.0, _INF)
                    self.team_cost(t, e, rule.weight)
                if width is not None:
                    self.range_window(rule, name, present, pv, t, width, sd)
            if width is not None and rule.hard:
                self.range_exclusions(name, present, pv, width)

    def range_window(self, rule: NumericBoundsRule, name: str, present: np.ndarray,
                     pv: np.ndarray, t: int, width: float, sd: float) -> None:
        vmin, vmax = float(pv.min()), float(pv.max())
        if vmax - vmin <= width + 1e-12:
            return
        b = self.b
        lo_t = b.add_var(vmin, vmax)
        hi_t = b.add_var(vmin, vmax)
        for i, v in zip(present.tolist(), pv.tolist()):
            col = i * self.T + t
            if vmax - v > 1e-12:
                b.add_row([lo_t, col], [1.0, vmax - v], -_INF, vmax)
            if v - vmin > 1e-12:
                b.add_row([hi_t, col], [1.0, -(v - vmin)], vmin, _INF)
        if rule.hard:
            b.add_row([hi_t, lo_t], [1.0, -1.0], -_INF, width)
            return
        r = b.add_var(0.0)
        b.add_row([hi_t, lo_t, r], [1.0, -1.0, -1.0], -_INF, width)
        self.team_cost(t, r, rule.weight / sd)

    def range_exclusions(self, name: str, present: np.ndarray, pv: np.ndarray,
                         width: float) -> None:
        order = np.argsort(pv, kind="stable")
        sv, si = pv[order], present[order]
        pairs: Set[Tuple[int, int]] = set()
        j = 0
        for k in range(len(sv)):
            while j < len(sv) and sv[j] <= sv[k] + width + 1e-9:
                j += 1
            for m in range(j, len(sv)):
                a, c = int(si[k]), int(si[m])
                pairs.add((min(a, c), max(a, c)))
            if len(pairs) > MAX_RANGE_EXCLUSION_PAIRS:
                self.note(f"bounds rule '{name}': range enforced by window rows only (more "
                          f"than {MAX_RANGE_EXCLUSION_PAIRS} far-apart pairs)")
                return
        self.range_conflicts.update(pairs)

    def add_shared_rules(self) -> None:
        for rule in self.policy.shared:
            if not rule.enabled or (not rule.hard and rule.weight <= 0):
                continue
            name = rule.label or rule.attribute
            attr = self.cohort.attributes.get(rule.attribute)
            if attr is None or attr.kind not in ("categorical", "multi"):
                self.note(f"shared-value rule '{name}' skipped (needs an encoded categorical "
                          f"or multi-value attribute)")
                continue
            k = int(rule.min_shared)
            if k < 1:
                continue
            members = _category_members(attr)
            held: List[Set[int]] = [set() for _ in range(self.n)]
            for c, mem in enumerate(members):
                for i in mem.tolist():
                    held[i].add(c)
            if rule.hard:
                self.hard.shared.append((name, held, k))
                short = [self.cohort.student_ids[i] for i in range(self.n) if len(held[i]) < k]
                if short:
                    self.reasons.append(
                        f"hard shared-value rule '{name}': {len(short)} student(s) hold fewer "
                        f"than {k} value(s) (e.g. {short[0]})")
            n_vars = sum(1 for s in self.sizes for mem in members if s and len(mem) >= s)
            if n_vars > MAX_SHARED_VARS:
                self.note(f"shared-value rule '{name}' left to local search ({n_vars} team/value "
                          f"indicators, more than {MAX_SHARED_VARS})")
                continue
            for t, s in enumerate(self.sizes):
                if s == 0:
                    continue
                zs: List[int] = []
                for mem in members:
                    if len(mem) < s:
                        continue
                    z = self.b.add_var(0.0, 1.0, integer=True)
                    cols = self.x_cols(mem, t)
                    self.b.add_row([z] + cols, [float(s)] + [-1.0] * len(cols), -_INF, 0.0)
                    zs.append(z)
                if rule.hard:
                    self.b.add_row(zs, [1.0] * len(zs), float(k), _INF)
                    continue
                u = self.b.add_var(0.0, float(k))
                self.b.add_row(zs + [u], [1.0] * len(zs) + [1.0], float(k), _INF)
                self.team_cost(t, u, rule.weight)

    def pair_rule_applies(self, rule: PairRule) -> bool:
        if rule.projects is None:
            return True
        return self.project_id is not None and self.project_id in {str(p) for p in rule.projects}

    def add_pairs_and_conflicts(self) -> None:
        conflicts: Set[Tuple[int, int]] = set(self.forbidden_in) | self.range_conflicts
        prefs = self.policy.preferences
        if prefs.use_student_preferences and prefs.avoid_hard:
            for i, listed in self.cohort.avoid.items():
                for j in listed:
                    if i != j and 0 <= j < self.n:
                        conflicts.add((min(i, j), max(i, j)))

        parent = list(range(self.n))

        def find(v: int) -> int:
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for rule in self.policy.pairs:
            if not rule.enabled or not self.pair_rule_applies(rule):
                continue
            idx: List[int] = []
            for sid in rule.students:
                i = self.cohort.lookup(sid)
                if i is not None and i not in idx:
                    idx.append(i)
            if len(idx) < 2:
                continue
            name = rule.label or ", ".join(rule.students)
            if rule.kind == "together":
                if rule.hard:
                    for i in idx[1:]:
                        parent[find(i)] = find(idx[0])
                elif rule.weight > 0:
                    self.soft_together(idx, rule.weight)
            elif rule.kind == "apart":
                if rule.hard:
                    conflicts.update((min(a, b), max(a, b)) for a, b in combinations(idx, 2))
                elif rule.weight > 0:
                    g = np.asarray(sorted(idx), dtype=np.int64)
                    for t, s in enumerate(self.sizes):
                        top = min(s, len(g))
                        self.convex_count_term(g, t, [v * (v - 1) / 2 for v in range(top + 1)],
                                               rule.weight)
            else:
                self.note(f"pair rule '{name}' skipped (unknown kind '{rule.kind}')")

        groups: Dict[int, List[int]] = {}
        for i in range(self.n):
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            if len(members) < 2:
                continue
            self.hard.together.append(members)
            self.add_together(members, conflicts)

        self.hard.forbidden = conflicts
        for a, b in conflicts:
            if a in self.locks and b in self.locks and self.locks[a] == self.locks[b]:
                self.reasons.append(f"students {self.cohort.student_ids[a]} and "
                                    f"{self.cohort.student_ids[b]} must be apart but are locked "
                                    f"to the same team")
        for clique in _clique_cover(conflicts):
            arr = np.asarray(clique, dtype=np.int64)
            for t in range(self.T):
                self.b.add_row(self.x_cols(arr, t), [1.0] * len(clique), -_INF, 1.0)
        self.add_pair_weights(conflicts)
        self.add_preferences(conflicts)

    def add_together(self, members: List[int], conflicts: Set[Tuple[int, int]]) -> None:
        ids = self.cohort.student_ids
        if len(members) > max(self.sizes):
            self.reasons.append(f"{len(members)} students must be together but the largest "
                                f"team has {max(self.sizes)} places")
        clash = [(a, b) for a, b in combinations(members, 2) if (a, b) in conflicts]
        if clash:
            a, b = clash[0]
            self.reasons.append(f"students {ids[a]} and {ids[b]} must be both together and apart")
        locked = {self.locks[i] for i in members if i in self.locks}
        if len(locked) > 1:
            self.reasons.append("students that must be together are locked to different teams")
        first = members[0]
        for other in members[1:]:
            for t in range(self.T):
                self.b.add_row([first * self.T + t, other * self.T + t], [1.0, -1.0], 0.0, 0.0)

    def soft_together(self, idx: List[int], weight: float) -> None:
        T = self.T
        for a, b in combinations(sorted(idx), 2):
            for t in range(T):
                w = self.b.add_var(0.0)
                self.b.add_row([w, a * T + t, b * T + t], [1.0, -1.0, 1.0], 0.0, _INF)
                self.b.add_row([w, a * T + t, b * T + t], [1.0, 1.0, -1.0], 0.0, _INF)
                self.team_cost(t, w, weight / 2.0)

    def add_pair_weights(self, conflicts: Set[Tuple[int, int]]) -> None:
        weights = dict(self.weights_in)
        prefs = self.policy.preferences
        if prefs.use_student_preferences and not prefs.avoid_hard and prefs.avoid_weight > 0:
            for i, listed in self.cohort.avoid.items():
                for j in listed:
                    key = (min(i, j), max(i, j))
                    weights[key] = weights.get(key, 0.0) + prefs.avoid_weight
        positive = {k: w for k, w in weights.items() if w > 0 and k not in conflicts}
        if not positive:
            return
        cliques, residual = decompose_pair_weights(positive, wildcards=conflicts)
        self.note(f"pair weights: {len(positive)} weighted pairs modelled as {len(cliques)} "
                  f"group term(s) and {len(residual)} pair indicator(s)")
        for members, w in cliques:
            g = np.asarray(members, dtype=np.int64)
            for t, s in enumerate(self.sizes):
                top = min(s, len(g))
                self.convex_count_term(g, t, [v * (v - 1) / 2 for v in range(top + 1)], w)
        items = sorted(residual.items())
        if not items:
            return
        budget = MAX_PAIR_WEIGHT_ROWS // self.T
        if len(items) > budget:
            items.sort(key=lambda kw: (-kw[1], kw[0]))
            self.note(f"pairwise history: modelled the {budget} heaviest of {len(items)} weighted "
                      f"pairs outside prior-group cliques (row budget {MAX_PAIR_WEIGHT_ROWS}); "
                      f"the rest are left to local search")
            items = sorted(items[:budget])
        T, b = self.T, self.b
        if self.per_team_pairs:
            for (i, j), w in items:
                first = b.add_vars(T, 0.0, 1.0)
                for t in range(T):
                    b.add_row([first + t, i * T + t, j * T + t], [1.0, -1.0, -1.0], -1.0, _INF)
                    self.team_cost(t, first + t, w)
            return
        first = b.add_vars(len(items), 0.0, 1.0)
        for k, ((i, j), w) in enumerate(items):
            p = first + k
            b.cost[p] += w
            for t in range(T):
                b.add_row([p, i * T + t, j * T + t], [1.0, -1.0, -1.0], -1.0, _INF)

    def add_preferences(self, conflicts: Set[Tuple[int, int]]) -> None:
        prefs = self.policy.preferences
        if not prefs.use_student_preferences or prefs.prefer_weight <= 0 or not self.cohort.prefer:
            return
        bonus: Dict[Tuple[int, int], float] = {}
        for i, listed in self.cohort.prefer.items():
            for j in listed:
                key = (min(i, j), max(i, j))
                if key not in conflicts:
                    bonus[key] = bonus.get(key, 0.0) + prefs.prefer_weight
        if not bonus:
            return
        rows = len(bonus) * self.T * (2 if self.per_team_pairs else 1)
        if rows > MAX_PREFERENCE_ROWS:
            self.note(f"student preferences left out ({len(bonus)} preferred pairs need {rows} "
                      f"rows, more than {MAX_PREFERENCE_ROWS})")
            return
        T, b = self.T, self.b
        items = sorted(bonus.items())
        if self.per_team_pairs:
            for (i, j), w in items:
                first = b.add_vars(T, 0.0, 1.0)
                for t in range(T):
                    b.add_row([first + t, i * T + t], [1.0, -1.0], -_INF, 0.0)
                    b.add_row([first + t, j * T + t], [1.0, -1.0], -_INF, 0.0)
                    self.team_cost(t, first + t, -w)
            return
        first = b.add_vars(len(items), 0.0, 1.0)
        for k, ((i, j), w) in enumerate(items):
            r = first + k
            b.cost[r] -= w
            for t in range(T):
                b.add_row([r, i * T + t, j * T + t], [1.0, 1.0, -1.0], -_INF, 1.0)

    def finish_objective(self) -> None:
        w = min(max(float(self.policy.fairness.worst_team_weight), 0.0), 1.0)
        has_terms = any(self.team_terms)
        for terms in self.team_terms:
            for col, coef in terms:
                self.b.cost[col] += (1.0 - w) * coef
        if w <= 0.0 or not has_terms or self.T == 0:
            return
        self.has_worst_team = True
        z = self.b.add_var(-_INF, _INF)
        self.b.cost[z] += w * self.T
        for terms in self.team_terms:
            cols = [z] + [col for col, _ in terms]
            vals = [1.0] + [-coef for _, coef in terms]
            self.b.add_row(cols, vals, 0.0, _INF)

    def build(self) -> MilpModel:
        start = time.perf_counter()
        self.add_assignment()
        self.add_symmetry_breaking()
        self.add_attribute_rules()
        self.add_count_rules()
        self.add_isolation_rules()
        self.add_bounds_rules()
        self.add_shared_rules()
        self.add_pairs_and_conflicts()
        self.finish_objective()
        b = self.b
        nv = b.n_vars
        A = sparse.csr_matrix(
            (np.asarray(b.vals, dtype=float), (np.asarray(b.rows, dtype=np.int64),
                                               np.asarray(b.cols, dtype=np.int64))),
            shape=(len(b.lo), nv),
        )
        A.sum_duplicates()
        return MilpModel(
            n=self.n, n_teams=self.T,
            c=np.asarray(b.cost, dtype=float), A=A,
            row_lo=np.asarray(b.lo, dtype=float), row_hi=np.asarray(b.hi, dtype=float),
            lb=np.asarray(b.lb, dtype=float), ub=np.asarray(b.ub, dtype=float),
            integrality=np.asarray(b.integer, dtype=np.uint8),
            hard=self.hard, notes=self.notes, infeasible_reasons=self.reasons,
            build_s=time.perf_counter() - start, has_worst_team=self.has_worst_team,
        )


def build_model(
    cohort: EncodedCohort,
    policy: Policy,
    team_sizes: Sequence[int],
    *,
    forbidden_pairs: Iterable[Tuple[int, int]] = (),
    pair_weights: Optional[Dict[Tuple[int, int], float]] = None,
    locks: Optional[Dict[int, int]] = None,
    project_id: Optional[str] = None,
    symmetry_breaking: bool = True,
) -> MilpModel:
    forbidden, weights, norm_locks = validate_inputs(
        cohort, team_sizes, forbidden_pairs, pair_weights, locks)
    factory = _ModelFactory(cohort, policy, [int(s) for s in team_sizes], forbidden, weights,
                            norm_locks, project_id, symmetry_breaking)
    return factory.build()


def hard_violations(model: MilpModel, assignment: Sequence[int]) -> List[str]:
    spec = model.hard
    T = model.n_teams
    team = np.asarray(assignment, dtype=np.int64)
    problems: List[str] = []
    if team.shape != (model.n,) or (len(team) and (team.min() < 0 or team.max() >= T)):
        return ["assignment has the wrong length or team indices out of range"]
    sizes = np.bincount(team, minlength=T)
    if sizes.tolist() != list(spec.sizes):
        problems.append(f"team sizes {sizes.tolist()} differ from the required {spec.sizes}")
    for i, t in spec.locks.items():
        if team[i] != t:
            problems.append(f"student {i} is not in its locked team {t}")
    clashes = [(a, b) for a, b in spec.forbidden if team[a] == team[b]]
    if clashes:
        problems.append(f"{len(clashes)} forbidden pair(s) share a team, e.g. {clashes[0]}")
    for members in spec.together:
        if len(set(team[members].tolist())) > 1:
            problems.append(f"students {members} must be together but are split")
    for name, g, lo, hi in spec.counts:
        n_t = np.bincount(team[g], minlength=T) if len(g) else np.zeros(T, dtype=np.int64)
        bad = [t for t in range(T) if (lo is not None and n_t[t] < lo)
               or (hi is not None and n_t[t] > hi)]
        if bad:
            problems.append(f"hard count rule '{name}' broken in {len(bad)} team(s)")
    for name, g, k in spec.isolation:
        n_t = np.bincount(team[g], minlength=T)
        bad = [t for t in range(T) if 0 < n_t[t] < k]
        if bad:
            problems.append(f"hard isolation rule '{name}' broken in {len(bad)} team(s)")
    tol = 1e-7
    for name, idx, vals, lo, hi, width in spec.bounds:
        bad = 0
        for t in range(T):
            v = vals[team[idx] == t]
            if not len(v):
                continue
            mean = float(v.mean())
            if ((lo is not None and mean < lo - tol) or (hi is not None and mean > hi + tol)
                    or (width is not None and float(v.max() - v.min()) > width + tol)):
                bad += 1
        if bad:
            problems.append(f"hard bounds rule '{name}' broken in {bad} team(s)")
    for name, held, k in spec.shared:
        bad = 0
        for t in range(T):
            members = np.flatnonzero(team == t).tolist()
            if not members:
                continue
            common = set.intersection(*(held[i] for i in members))
            if len(common) < k:
                bad += 1
        if bad:
            problems.append(f"hard shared-value rule '{name}' broken in {bad} team(s)")
    return problems


def linearised_objective(model: MilpModel, assignment: Sequence[int]) -> Optional[float]:
    T = model.n_teams
    team = np.asarray(assignment, dtype=np.int64)
    if team.shape != (model.n,) or (len(team) and (team.min() < 0 or team.max() >= T)):
        return None
    nx = model.n * T
    x = np.zeros(nx)
    x[np.arange(model.n) * T + team] = 1.0
    lb, ub = model.lb.copy(), model.ub.copy()
    lb[:nx] = x
    ub[:nx] = x
    res = milp(model.c, constraints=LinearConstraint(model.A, model.row_lo, model.row_hi),
               integrality=model.integrality, bounds=Bounds(lb, ub),
               options={"disp": False, "mip_rel_gap": 0.0})
    return float(res.fun) if res.status == 0 else None
