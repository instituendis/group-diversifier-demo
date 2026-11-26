from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .history import PairHistoryData
from .policy import (
    HARD_PENALTY,
    AttributeRule,
    CountRule,
    IsolationRule,
    NumericBoundsRule,
    PairRule,
    Policy,
    SharedValueRule,
)
from .roster import EncodedAttribute, EncodedCohort

_EPS = 1e-12
_BOUND_TOL = 1e-9
_GOAL_CODES = {"balance": 0, "diversify": 1, "cluster": 2}
_NUM_GOALS = ("balance", "cluster", "spread")

HISTORY_LABEL = "history"
HISTORY_CAP_LABEL = "history cap"
PREFERENCES_LABEL = "preferences"
PAIR_PENALTY_LABEL = "overlap with alternatives"


@dataclass
class TermInfo:
    label: str
    family: str
    attribute: Optional[str] = None
    goal: Optional[str] = None
    hard: bool = False
    weight: float = 0.0
    rule: Any = None


@dataclass
class _CatRule:
    term: int
    name: str
    goal: int
    weight: float
    offset: int
    ncat: int
    n_pool: int
    sqn: float
    coef: List[float]
    multi: bool = False


@dataclass
class _NumRule:
    term: int
    name: str
    goal: str
    weight: float
    shift: float
    mean_shifted: float
    var: float
    n_present: int
    coef_b: List[float]
    coef_c: List[float]
    coef_d: List[Optional[float]]
    present_values: List[float] = field(default_factory=list)

    def disp_coef(self, m: int) -> float:
        while m >= len(self.coef_d):
            self.coef_d.append(None)
        cached = self.coef_d[m]
        if cached is not None:
            return cached
        expected = _expected_dispersion(self.present_values, self.var, m)
        coef = self.weight / expected if expected > _EPS else 0.0
        self.coef_d[m] = coef
        return coef

    def value(self, m: int, s1: float, sq: float) -> float:
        if m <= 0:
            return 0.0
        if self.goal == "balance":
            dev = s1 - m * self.mean_shifted
            return self.coef_b[m] * (dev * dev / m)
        if self.goal == "cluster":
            return self.coef_c[m] * (sq - s1 * s1 / m)
        dev = s1 - m * self.mean_shifted
        cd = self.coef_d[m] if m < len(self.coef_d) else None
        if cd is None:
            cd = self.disp_coef(m)
        mean = s1 / m
        disp = (sq / m - mean * mean) / self.var - 1.0
        return self.coef_b[m] * (dev * dev / m) + cd * (disp * disp)


@dataclass
class _GroupRule:
    term: int
    label: str
    family: str
    hard: bool
    cost: float
    ngroups: int
    table: List[float]
    group_labels: List[str]
    rule: Any = None
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    min_together: Optional[int] = None
    pair_students: List[int] = field(default_factory=list)


@dataclass
class _BoundsRule:
    term: int
    label: str
    attribute: str
    hard: bool
    weight: float
    cost: float
    sd: float
    shift: float
    mean_min: Optional[float]
    mean_max: Optional[float]
    max_range: Optional[float]
    rule: Any = None

    def units(self, m: int, s1: float, lo: float, hi: float) -> Tuple[float, int]:
        if m <= 0:
            return 0.0, 0
        excess = 0.0
        broken = 0
        mean = s1 / m
        if self.mean_min is not None and mean < self.mean_min - _BOUND_TOL:
            excess += self.mean_min - mean
            broken += 1
        if self.mean_max is not None and mean > self.mean_max + _BOUND_TOL:
            excess += mean - self.mean_max
            broken += 1
        if self.max_range is not None and hi - lo > self.max_range + _BOUND_TOL:
            excess += hi - lo - self.max_range
            broken += 1
        return excess / self.sd, broken

    def value(self, m: int, s1: float, lo: float, hi: float) -> float:
        excess, broken = self.units(m, s1, lo, hi)
        if self.hard:
            return (excess + broken) * self.cost
        return excess * self.cost


@dataclass
class _SharedRule:
    term: int
    label: str
    attribute: str
    hard: bool
    weight: float
    cost: float
    min_shared: int
    n_values: int
    categorical: bool
    rule: Any = None

    def units(self, shared: int, size: int) -> int:
        if size <= 0:
            return 0
        return max(0, self.min_shared - shared)

    def value(self, shared: int, size: int) -> float:
        if size <= 0 or shared >= self.min_shared:
            return 0.0
        return (self.min_shared - shared) * self.cost


def _present_categories(attr: EncodedAttribute) -> set:
    if attr.kind == "multi":
        present: set = set()
        for codes in attr.multi:
            present.update(codes)
        return present
    return {c for c in attr.codes if c >= 0}


def _expected_dispersion(values: List[float], var: float, m: int) -> float:
    n_present = len(values)
    if m <= 0 or m >= n_present or var <= _EPS:
        return 0.0
    if m == 1:
        return 1.0
    N = n_present
    mean = math.fsum(values) / N
    y = [v - mean for v in values]
    P2 = math.fsum(v * v for v in y)
    P4 = math.fsum(v ** 4 for v in y)

    def p(k: int) -> float:
        out = 1.0
        for r in range(k):
            out *= (m - r) / (N - r) if m > r else 0.0
        return out

    p1, p2, p3, p4 = p(1), p(2), p(3), p(4)
    d4 = P4
    d31 = -P4
    d22 = P2 * P2 - P4
    d211 = 2.0 * P4 - P2 * P2
    d1111 = 3.0 * P2 * P2 - 6.0 * P4
    e_s2s2 = p1 * d4 + p2 * d22
    e_s2s1s1 = p1 * d4 + 2.0 * p2 * d31 + p2 * d22 + p3 * d211
    e_s1_4 = p1 * d4 + 4.0 * p2 * d31 + 3.0 * p2 * d22 + 6.0 * p3 * d211 + p4 * d1111
    e_pv2 = e_s2s2 / m ** 2 - 2.0 * e_s2s1s1 / m ** 3 + e_s1_4 / m ** 4
    e_pv = p1 * P2 / m - (p1 * P2 - p2 * P2) / m ** 2
    sigma2 = P2 / N
    value = (e_pv2 - 2.0 * sigma2 * e_pv + sigma2 * sigma2) / (sigma2 * sigma2)
    return max(0.0, value)


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "∞"
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _bounds_text(lo: Optional[int], hi: Optional[int]) -> str:
    if lo is not None and hi is not None:
        return f"{lo}" if lo == hi else f"{lo}–{hi}"
    if hi is not None:
        return f"≤ {hi}"
    return f"≥ {lo}"


class ObjectiveModel:
    def __init__(self, cohort: EncodedCohort, policy: Policy, team_sizes: Sequence[int], *,
                 history: Optional[PairHistoryData] = None,
                 project_id: Optional[str] = None,
                 hard_penalty: float = HARD_PENALTY,
                 pair_hard_penalty: Optional[float] = None,
                 anchor: Optional[Sequence[int]] = None,
                 anchor_weight: float = 0.0,
                 pair_penalties: Optional[Dict[Tuple[int, int], float]] = None) -> None:
        self.cohort = cohort
        self.policy = policy
        self.team_sizes = [int(s) for s in team_sizes]
        self.n = cohort.n
        self.n_teams = len(self.team_sizes)
        self.history = history
        self.project_id = None if project_id is None else str(project_id)
        self.hard_penalty = float(hard_penalty)
        self.pair_hard_penalty = (self.hard_penalty if pair_hard_penalty is None
                                  else float(pair_hard_penalty))
        self.fairness_weight = float(policy.fairness.worst_team_weight)
        self.warnings: List[str] = []
        self.skipped_rules: List[Dict[str, Any]] = []
        self.terms: List[TermInfo] = []
        self.anchor_weight = float(anchor_weight or 0.0)
        if anchor is not None and len(anchor) != self.n:
            raise ValueError(f"Anchor has {len(anchor)} entries for {self.n} students")
        self.anchor: Optional[List[int]] = (
            [(-1 if a is None else int(a)) for a in anchor]
            if anchor is not None and self.anchor_weight > 0 else None)
        self.pair_penalties: Dict[Tuple[int, int], float] = {}
        for (i, j), w in (pair_penalties or {}).items():
            i, j = int(i), int(j)
            if i == j or not w:
                continue
            key = (i, j) if i < j else (j, i)
            self.pair_penalties[key] = self.pair_penalties.get(key, 0.0) + float(w)

        self.cat_rules: List[_CatRule] = []
        self.num_rules: List[_NumRule] = []
        self.group_rules: List[_GroupRule] = []
        self.bound_rules: List[_BoundsRule] = []
        self.shared_rules: List[_SharedRule] = []
        self.flat_k: List[int] = []
        self.flat_names: List[Tuple[str, str]] = []
        self.cat_codes: List[Tuple[int, ...]] = [() for _ in range(self.n)]
        self.multi_codes: List[Tuple[Tuple[int, ...], ...]] = [() for _ in range(self.n)]
        self.num_values: List[Tuple[Optional[float], ...]] = [() for _ in range(self.n)]
        self.group_codes: List[Tuple[Tuple[int, ...], ...]] = [() for _ in range(self.n)]
        self.bound_values: List[Tuple[Optional[float], ...]] = [() for _ in range(self.n)]
        self.shared_codes: List[Tuple[Tuple[int, ...], ...]] = [() for _ in range(self.n)]

        self._build_attribute_rules()
        self.n_cat_single = sum(1 for r in self.cat_rules if not r.multi)
        self.flat_owner: List[int] = [r for r, rule in enumerate(self.cat_rules)
                                      for _ in range(rule.ncat)]
        self._build_group_rules()
        self._build_bounds_rules()
        self._build_shared_rules()
        self._build_pair_costs()
        self.pair_term = (len(self.cat_rules) + len(self.num_rules) + len(self.group_rules)
                          + len(self.bound_rules) + len(self.shared_rules))
        self.n_terms = self.pair_term + 1
        self.has_extra = bool(self.bound_rules or self.shared_rules)
        self._labels_unique()
        self._sync_rule_labels()

    def _add_term(self, info: TermInfo) -> None:
        self.terms.append(info)

    def _labels_unique(self) -> None:
        seen: Dict[str, int] = {}
        for info in self.terms:
            base = info.label
            if base in seen:
                seen[base] += 1
                info.label = f"{base} ({seen[base]})"
            else:
                seen[base] = 1

    def _sync_rule_labels(self) -> None:
        for rule in [*self.group_rules, *self.bound_rules, *self.shared_rules]:
            rule.label = self.terms[rule.term].label

    def _build_attribute_rules(self) -> None:
        cat_rules: List[Tuple[AttributeRule, EncodedAttribute, str]] = []
        multi_rules: List[Tuple[AttributeRule, EncodedAttribute, str]] = []
        num_rules: List[Tuple[AttributeRule, EncodedAttribute, str]] = []
        for rule in self.policy.active_attributes():
            attr = self.cohort.attributes.get(rule.attribute)
            if attr is None:
                self.warnings.append(
                    f"Attribute rule '{rule.attribute}' skipped: attribute is not available "
                    f"as a categorical, numeric or multi-value column"
                )
                continue
            goal = rule.goal
            if attr.kind == "numeric":
                if goal not in _NUM_GOALS:
                    goal = "spread" if goal == "diversify" else "balance"
                    self.warnings.append(
                        f"Attribute rule '{rule.attribute}': goal '{rule.goal}' does not apply to "
                        f"a numeric attribute; using '{goal}'"
                    )
                num_rules.append((rule, attr, goal))
            else:
                if goal not in _GOAL_CODES:
                    goal = "diversify"
                    self.warnings.append(
                        f"Attribute rule '{rule.attribute}': goal '{rule.goal}' does not apply to "
                        f"a categorical attribute; using 'diversify'"
                    )
                present = _present_categories(attr)
                if attr.kind == "multi":
                    has_missing = any(not codes for codes in attr.multi)
                else:
                    has_missing = any(c < 0 for c in attr.codes)
                if not present or (len(present) == 1 and not has_missing):
                    self.warnings.append(
                        f"Attribute rule '{rule.attribute}' skipped: every participant has "
                        f"the same value"
                    )
                    continue
                (multi_rules if attr.kind == "multi" else cat_rules).append((rule, attr, goal))

        n = self.n
        for multi, items in ((False, cat_rules), (True, multi_rules)):
            for rule, attr, goal in items:
                offset = len(self.flat_k)
                ncat = len(attr.categories)
                counts = [0] * ncat
                if multi:
                    for codes in attr.multi:
                        for c in set(codes):
                            counts[c] += 1
                else:
                    for c in attr.codes:
                        if c >= 0:
                            counts[c] += 1
                self.flat_k.extend(counts)
                self.flat_names.extend((attr.name, c) for c in attr.categories)
                present_n = (sum(1 for codes in attr.multi if codes) if multi
                             else sum(counts))
                coef = self._cat_coefficients(goal, counts, n, rule.weight, present_n)
                sk2 = sum(k * k for k in counts)
                term = len(self.terms)
                self.cat_rules.append(_CatRule(
                    term=term, name=attr.name, goal=_GOAL_CODES[goal], weight=float(rule.weight),
                    offset=offset, ncat=ncat, n_pool=n,
                    sqn=(sk2 / (n * n)) if n else 0.0, coef=coef, multi=multi,
                ))
                self._add_term(TermInfo(
                    label=f"{goal}: {attr.name}", family="attribute", attribute=attr.name,
                    goal=goal, weight=float(rule.weight), rule=rule,
                ))
                for i in range(n):
                    if multi:
                        self.multi_codes[i] = self.multi_codes[i] + (
                            tuple(sorted({offset + c for c in attr.multi[i]})),)
                    else:
                        c = attr.codes[i]
                        self.cat_codes[i] = self.cat_codes[i] + ((offset + c) if c >= 0 else -1,)

        for rule, attr, goal in num_rules:
            present = [v for v in attr.values if not math.isnan(v)]
            n_present = len(present)
            if n_present < 2:
                self.warnings.append(f"Attribute rule '{attr.name}' skipped: fewer than two values")
                continue
            mu = sum(present) / n_present
            integral = all(float(v).is_integer() for v in present)
            shift = float(round(mu)) if integral else mu
            shifted = [v - shift for v in present]
            mean_shifted = sum(shifted) / n_present
            var = sum((v - mean_shifted) ** 2 for v in shifted) / n_present
            if var <= _EPS:
                self.warnings.append(f"Attribute rule '{attr.name}' skipped: all values are equal")
                continue
            w = float(rule.weight)
            coef_b = [0.0] * (n_present + 1)
            coef_c = [0.0] * (n_present + 1)
            for m in range(1, n_present + 1):
                eb = (n_present - m) / (n_present - 1)
                ec = (m - 1) * n_present / (n_present - 1)
                coef_b[m] = w / (eb * var) if eb > _EPS else 0.0
                coef_c[m] = w / (ec * var) if ec > _EPS else 0.0
            term = len(self.terms)
            num_rule = _NumRule(
                term=term, name=attr.name, goal=goal, weight=w, shift=shift,
                mean_shifted=mean_shifted, var=var, n_present=n_present,
                coef_b=coef_b, coef_c=coef_c, coef_d=[None] * (n_present + 1),
                present_values=shifted,
            )
            if goal == "spread":
                for m in sorted(set(self.team_sizes) | {s - 1 for s in self.team_sizes}):
                    if 0 < m <= n_present:
                        num_rule.disp_coef(m)
            self.num_rules.append(num_rule)
            self._add_term(TermInfo(
                label=f"{goal}: {attr.name}", family="attribute", attribute=attr.name,
                goal=goal, weight=w, rule=rule,
            ))
            for i in range(n):
                v = attr.values[i]
                self.num_values[i] = self.num_values[i] + (
                    None if math.isnan(v) else v - shift,)

    @staticmethod
    def _cat_coefficients(goal: str, counts: List[int], n: int, weight: float,
                          present: Optional[int] = None) -> List[float]:
        coef = [0.0] * (n + 2)
        if n <= 1 or weight <= 0:
            return coef
        q = [k / n for k in counts]
        kp = sum(counts)
        holders = kp if present is None else present
        same = sum(k * (k - 1) for k in counts) / (n * (n - 1))
        diff = (holders * kp - sum(k * k for k in counts)) / (n * (n - 1))
        var_sum = sum(x * (1.0 - x) for x in q)
        for s in range(1, n + 1):
            pairs = s * (s - 1) / 2.0
            if goal == "balance":
                expected = s * (n - s) / (n - 1) * var_sum
            elif goal == "diversify":
                expected = pairs * same
            else:
                expected = pairs * diff
            coef[s] = weight / expected if expected > _EPS else 0.0
        return coef

    def _build_group_rules(self) -> None:
        n = self.n
        for rule in self.policy.counts:
            if rule.enabled:
                self._add_count_rule(rule)
        for rule in self.policy.isolation:
            if rule.enabled:
                self._add_isolation_rule(rule)
        for rule in self.policy.pairs:
            if not rule.enabled:
                continue
            if rule.projects is not None:
                targets = {str(p) for p in rule.projects}
                if self.project_id is None or self.project_id not in targets:
                    continue
            self._add_pair_rule(rule)
        for i in range(n):
            if len(self.group_codes[i]) < len(self.group_rules):
                self.group_codes[i] = self.group_codes[i] + (
                    (),) * (len(self.group_rules) - len(self.group_codes[i]))

    def _register_group(self, grule: _GroupRule, memberships: List[Tuple[int, ...]],
                        info: TermInfo) -> None:
        r = len(self.group_rules)
        self.group_rules.append(grule)
        self._add_term(info)
        for i in range(self.n):
            codes = self.group_codes[i]
            if len(codes) < r:
                codes = codes + ((),) * (r - len(codes))
            self.group_codes[i] = codes + (memberships[i],)

    def _value_groups(self, attr: EncodedAttribute, values: List[str], all_if_empty: bool,
                      where: str) -> List[int]:
        if not values and all_if_empty:
            return sorted(_present_categories(attr))
        codes = []
        for value in values:
            c = attr.code_of(value)
            if c < 0:
                self.warnings.append(f"{where}: value '{value}' does not occur in '{attr.name}'")
            elif c not in codes:
                codes.append(c)
        return codes

    def _add_count_rule(self, rule: CountRule) -> None:
        where = f"Count rule '{rule.label or rule.attribute}'"
        attr = self.cohort.attributes.get(rule.attribute)
        if attr is None:
            self.warnings.append(f"{where} skipped: attribute '{rule.attribute}' is not available")
            return
        if rule.min_count is None and rule.max_count is None:
            self.warnings.append(f"{where} skipped: no minimum or maximum")
            return
        n = self.n
        memberships: List[Tuple[int, ...]] = [()] * n
        group_labels: List[str] = []
        if attr.kind == "numeric":
            if rule.numeric_range is None:
                self.warnings.append(f"{where} skipped: numeric attributes need a numeric_range")
                return
            lo, hi = (list(rule.numeric_range) + [None, None])[:2]
            lo = None if lo is None else float(lo)
            hi = None if hi is None else float(hi)
            for i, v in enumerate(attr.values):
                if not math.isnan(v) and (lo is None or v >= lo) and (hi is None or v <= hi):
                    memberships[i] = (0,)
            lo_txt = "−∞" if lo is None else _fmt_num(lo)
            group_labels = [f"{attr.name} in [{lo_txt}, {_fmt_num(hi)}]"]
            auto = group_labels[0]
        else:
            if rule.numeric_range is not None:
                self.warnings.append(f"{where}: numeric_range ignored for a non-numeric attribute")
            codes = self._value_groups(attr, list(rule.values), rule.mode == "each", where)
            if not codes:
                self.warnings.append(f"{where} skipped: no matching values")
                return
            names = attr.categories
            if rule.mode == "each":
                index = {c: g for g, c in enumerate(codes)}
                group_labels = [f"{attr.name}={names[c]}" for c in codes]
                for i in range(n):
                    own = attr.multi[i] if attr.kind == "multi" else (
                        [attr.codes[i]] if attr.codes[i] >= 0 else [])
                    memberships[i] = tuple(sorted({index[c] for c in own if c in index}))
                auto = (f"each {attr.name}" if not rule.values
                        else "each of " + "/".join(names[c] for c in codes))
            else:
                wanted = set(codes)
                group_labels = [f"{attr.name}=" + "/".join(names[c] for c in codes)]
                for i in range(n):
                    own = attr.multi[i] if attr.kind == "multi" else (
                        [attr.codes[i]] if attr.codes[i] >= 0 else [])
                    if any(c in wanted for c in own):
                        memberships[i] = (0,)
                auto = "/".join(names[c] for c in codes)
        lo_c, hi_c = rule.min_count, rule.max_count
        table = [float(max(0, lo_c - k) if lo_c is not None else 0)
                 + float(max(0, k - hi_c) if hi_c is not None else 0) for k in range(n + 2)]
        label = f"count: {rule.label or auto + ' ' + _bounds_text(lo_c, hi_c)}"
        cost = self.hard_penalty if rule.hard else float(rule.weight)
        grule = _GroupRule(
            term=len(self.terms), label=label, family="count", hard=bool(rule.hard), cost=cost,
            ngroups=len(group_labels), table=table, group_labels=group_labels, rule=rule,
            min_count=lo_c, max_count=hi_c,
        )
        self._register_group(grule, memberships, TermInfo(
            label=label, family="count", attribute=attr.name, hard=bool(rule.hard),
            weight=float(rule.weight), rule=rule,
        ))

    def _add_isolation_rule(self, rule: IsolationRule) -> None:
        where = f"Isolation rule '{rule.label or rule.attribute}'"
        attr = self.cohort.attributes.get(rule.attribute)
        if attr is None or attr.kind == "numeric":
            self.warnings.append(
                f"{where} skipped: '{rule.attribute}' is not an available categorical attribute")
            return
        codes = self._value_groups(attr, list(rule.values), True, where)
        if not codes:
            self.warnings.append(f"{where} skipped: no matching values")
            return
        n = self.n
        k = max(2, int(rule.min_together))
        index = {c: g for g, c in enumerate(codes)}
        memberships: List[Tuple[int, ...]] = []
        for i in range(n):
            own = attr.multi[i] if attr.kind == "multi" else (
                [attr.codes[i]] if attr.codes[i] >= 0 else [])
            memberships.append(tuple(sorted({index[c] for c in own if c in index})))
        names = attr.categories
        table = [float(min(c, k - c)) if 0 < c < k else 0.0 for c in range(n + 2)]
        if rule.label:
            label = f"isolation: {rule.label}"
        elif rule.values:
            label = f"isolation: {attr.name}=" + "/".join(names[c] for c in codes)
        else:
            label = f"isolation: {attr.name} (any value)"
        cost = self.hard_penalty if rule.hard else float(rule.weight)
        grule = _GroupRule(
            term=len(self.terms), label=label, family="isolation", hard=bool(rule.hard),
            cost=cost, ngroups=len(codes), table=table,
            group_labels=[f"{attr.name}={names[c]}" for c in codes], rule=rule, min_together=k,
        )
        self._register_group(grule, memberships, TermInfo(
            label=label, family="isolation", attribute=attr.name, hard=bool(rule.hard),
            weight=float(rule.weight), rule=rule,
        ))

    def _add_pair_rule(self, rule: PairRule) -> None:
        indices: List[int] = []
        missing: List[str] = []
        for sid in rule.students:
            i = self.cohort.lookup(sid)
            if i is None:
                if str(sid) not in missing:
                    missing.append(str(sid))
            elif i not in indices:
                indices.append(i)
        if len(indices) < 2:
            label = f"pair: {rule.label}" if rule.label else (
                f"pair: {rule.kind} {','.join(str(s) for s in rule.students)}")
            shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
            reason = (f"only {len(indices)} of its students "
                      f"{'is' if len(indices) == 1 else 'are'} in this project"
                      + (f" (not here: {shown})" if missing else ""))
            self.skipped_rules.append({"label": label, "family": "pair", "rule": rule,
                                       "hard": bool(rule.hard), "reason": reason})
            return
        g = len(indices)
        n = self.n
        if rule.kind == "together":
            table = [c * (g - c) / 2.0 if 0 <= c <= g else 0.0 for c in range(n + 2)]
        else:
            table = [c * (c - 1) / 2.0 for c in range(n + 2)]
        ids = [self.cohort.student_ids[i] for i in indices]
        label = f"pair: {rule.label}" if rule.label else f"pair: {rule.kind} {','.join(ids)}"
        memberships: List[Tuple[int, ...]] = [()] * n
        for i in indices:
            memberships[i] = (0,)
        cost = self.hard_penalty if rule.hard else float(rule.weight)
        grule = _GroupRule(
            term=len(self.terms), label=label, family="pair", hard=bool(rule.hard), cost=cost,
            ngroups=1, table=table, group_labels=[",".join(ids)], rule=rule,
            pair_students=indices,
        )
        self._register_group(grule, memberships, TermInfo(
            label=label, family="pair", hard=bool(rule.hard), weight=float(rule.weight), rule=rule,
        ))

    def _build_bounds_rules(self) -> None:
        for rule in self.policy.numeric_bounds:
            if rule.enabled:
                self._add_bounds_rule(rule)

    def _add_bounds_rule(self, rule: NumericBoundsRule) -> None:
        where = f"Numeric bounds rule '{rule.label or rule.attribute}'"
        attr = self.cohort.attributes.get(rule.attribute)
        if attr is None or attr.kind != "numeric":
            self.warnings.append(
                f"{where} skipped: '{rule.attribute}' is not an available numeric attribute")
            return
        if rule.mean_min is None and rule.mean_max is None and rule.max_range is None:
            self.warnings.append(f"{where} skipped: no mean bound or maximum range")
            return
        if not rule.hard and rule.weight <= 0:
            return
        present = [v for v in attr.values if not math.isnan(v)]
        if len(present) < 2:
            self.warnings.append(f"{where} skipped: fewer than two values")
            return
        mu = sum(present) / len(present)
        sd = math.sqrt(sum((v - mu) ** 2 for v in present) / len(present))
        if sd <= _EPS:
            self.warnings.append(f"{where} skipped: all values are equal (no spread to "
                                 f"measure violations in)")
            return
        integral = all(float(v).is_integer() for v in present)
        shift = float(round(mu)) if integral else mu
        lo = None if rule.mean_min is None else float(rule.mean_min) - shift
        hi = None if rule.mean_max is None else float(rule.mean_max) - shift
        parts = []
        if rule.mean_min is not None and rule.mean_max is not None:
            parts.append(f"mean {_fmt_num(rule.mean_min)}–{_fmt_num(rule.mean_max)}")
        elif rule.mean_min is not None:
            parts.append(f"mean ≥ {_fmt_num(rule.mean_min)}")
        elif rule.mean_max is not None:
            parts.append(f"mean ≤ {_fmt_num(rule.mean_max)}")
        if rule.max_range is not None:
            parts.append(f"range ≤ {_fmt_num(rule.max_range)}")
        label = f"bounds: {rule.label}" if rule.label else f"bounds: {attr.name} {', '.join(parts)}"
        brule = _BoundsRule(
            term=len(self.terms), label=label, attribute=attr.name, hard=bool(rule.hard),
            weight=float(rule.weight), cost=self.hard_penalty if rule.hard else float(rule.weight),
            sd=sd, shift=shift, mean_min=lo, mean_max=hi,
            max_range=None if rule.max_range is None else float(rule.max_range), rule=rule,
        )
        self.bound_rules.append(brule)
        self._add_term(TermInfo(label=label, family="bounds", attribute=attr.name,
                                hard=bool(rule.hard), weight=float(rule.weight), rule=rule))
        for i in range(self.n):
            v = attr.values[i]
            self.bound_values[i] = self.bound_values[i] + (None if math.isnan(v) else v - shift,)

    def _build_shared_rules(self) -> None:
        for rule in self.policy.shared:
            if rule.enabled:
                self._add_shared_rule(rule)

    def _add_shared_rule(self, rule: SharedValueRule) -> None:
        where = f"Shared-value rule '{rule.label or rule.attribute}'"
        attr = self.cohort.attributes.get(rule.attribute)
        if attr is None or attr.kind not in ("categorical", "multi"):
            self.warnings.append(f"{where} skipped: '{rule.attribute}' is not an available "
                                 f"categorical or multi-value attribute")
            return
        k = int(rule.min_shared)
        if k < 1:
            self.warnings.append(f"{where} skipped: min_shared must be at least 1")
            return
        if not rule.hard and rule.weight <= 0:
            return
        categorical = attr.kind == "categorical"
        if rule.label:
            label = f"shared: {rule.label}"
        elif categorical:
            label = f"shared: same {attr.name}"
        else:
            label = f"shared: {attr.name} ≥ {k} in common"
        srule = _SharedRule(
            term=len(self.terms), label=label, attribute=attr.name, hard=bool(rule.hard),
            weight=float(rule.weight), cost=self.hard_penalty if rule.hard else float(rule.weight),
            min_shared=k, n_values=len(attr.categories), categorical=categorical, rule=rule,
        )
        self.shared_rules.append(srule)
        self._add_term(TermInfo(label=label, family="shared", attribute=attr.name,
                                hard=bool(rule.hard), weight=float(rule.weight), rule=rule))
        for i in range(self.n):
            if categorical:
                c = attr.codes[i]
                codes: Tuple[int, ...] = (c,) if c >= 0 else ()
            else:
                codes = tuple(sorted(set(attr.multi[i])))
            self.shared_codes[i] = self.shared_codes[i] + (codes,)

    def _build_pair_costs(self) -> None:
        n = self.n
        self.hist_soft: Dict[Tuple[int, int], float] = {}
        self.hist_cap: Dict[Tuple[int, int], float] = {}
        self.pref_cost: Dict[Tuple[int, int], float] = {}
        self.pair_hard: Dict[Tuple[int, int], int] = {}
        hp = self.policy.history
        self.has_history_terms = False
        self.has_cap_term = False
        self.has_pref_term = False
        if self.history is not None and hp.enabled:
            if hp.repeat_weight:
                self.has_history_terms = True
                for pair, w in self.history.weights.items():
                    if w:
                        self.hist_soft[pair] = hp.repeat_weight * w
            if hp.max_times_together is not None:
                self.has_cap_term = True
                cap_cost = self.pair_hard_penalty if hp.hard_cap else float(hp.cap_weight)
                for pair, c in self.history.cap_counts.items():
                    if c + 1 > hp.max_times_together:
                        self.hist_cap[pair] = cap_cost
                        if hp.hard_cap:
                            self.pair_hard[pair] = self.pair_hard.get(pair, 0) + 1
        pp = self.policy.preferences
        if pp.use_student_preferences and (self.cohort.prefer or self.cohort.avoid):
            self.has_pref_term = True
            for i, listed in self.cohort.prefer.items():
                for j in listed:
                    if 0 <= j < n and j != i:
                        pair = (i, j) if i < j else (j, i)
                        self.pref_cost[pair] = self.pref_cost.get(pair, 0.0) - pp.prefer_weight
            avoid_cost = self.pair_hard_penalty if pp.avoid_hard else float(pp.avoid_weight)
            for i, listed in self.cohort.avoid.items():
                for j in listed:
                    if 0 <= j < n and j != i:
                        pair = (i, j) if i < j else (j, i)
                        self.pref_cost[pair] = self.pref_cost.get(pair, 0.0) + avoid_cost
                        if pp.avoid_hard:
                            self.pair_hard[pair] = self.pair_hard.get(pair, 0) + 1
        if self.has_history_terms:
            self._add_term(TermInfo(label=HISTORY_LABEL, family="history",
                                    weight=float(hp.repeat_weight)))
        if self.has_cap_term:
            self._add_term(TermInfo(label=HISTORY_CAP_LABEL, family="history_cap",
                                    hard=bool(hp.hard_cap), weight=float(hp.cap_weight)))
        if self.has_pref_term:
            self._add_term(TermInfo(label=PREFERENCES_LABEL, family="preferences",
                                    hard=bool(pp.avoid_hard), weight=float(pp.prefer_weight)))
        self.pair_penalties = {(i, j): w for (i, j), w in self.pair_penalties.items()
                               if 0 <= i < n and 0 <= j < n}
        self.has_pair_penalty_term = bool(self.pair_penalties)
        if self.has_pair_penalty_term:
            self._add_term(TermInfo(label=PAIR_PENALTY_LABEL, family="pair_penalty"))
        combined: Dict[Tuple[int, int], float] = {}
        for source in (self.hist_soft, self.hist_cap, self.pref_cost, self.pair_penalties):
            for pair, w in source.items():
                combined[pair] = combined.get(pair, 0.0) + w
        self.pair_cost: List[Dict[int, float]] = [dict() for _ in range(n)]
        self.neighbours: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for (i, j), w in sorted(combined.items()):
            if w == 0.0:
                continue
            self.pair_cost[i][j] = w
            self.pair_cost[j][i] = w
            self.neighbours[i].append((j, w))
            self.neighbours[j].append((i, w))
        self.has_pairs = any(self.neighbours)

    def _aggregate(self, members: Sequence[int]) -> Tuple:
        n_flat = len(self.flat_k)
        cnt = [0] * n_flat
        ncat = len(self.cat_rules)
        s2 = [0] * ncat
        pn = [0] * ncat
        lsum = [0] * ncat
        nsingle = self.n_cat_single
        flat_k = self.flat_k
        mp = [0] * (ncat - nsingle)
        for i in members:
            for r, c in enumerate(self.cat_codes[i]):
                if c >= 0:
                    s2[r] += 2 * cnt[c] + 1
                    cnt[c] += 1
                    pn[r] += 1
                    lsum[r] += flat_k[c]
            for rr, codes in enumerate(self.multi_codes[i]):
                r = nsingle + rr
                if codes:
                    mp[rr] += 1
                for c in codes:
                    s2[r] += 2 * cnt[c] + 1
                    cnt[c] += 1
                    pn[r] += 1
                    lsum[r] += flat_k[c]
        nnum = len(self.num_rules)
        mm = [0] * nnum
        s1 = [0.0] * nnum
        sq = [0.0] * nnum
        for i in members:
            for r, v in enumerate(self.num_values[i]):
                if v is not None:
                    mm[r] += 1
                    s1[r] += v
                    sq[r] += v * v
        gc = [[0] * g.ngroups for g in self.group_rules]
        for i in members:
            for r, groups in enumerate(self.group_codes[i]):
                for g in groups:
                    gc[r][g] += 1
        gu = []
        for r, grule in enumerate(self.group_rules):
            table = grule.table
            gu.append(float(sum(table[c] for c in gc[r])))
        ptot = 0.0
        if self.has_pairs:
            pc = self.pair_cost
            ms = list(members)
            for a_pos, a in enumerate(ms):
                row = pc[a]
                if row:
                    for b in ms[a_pos + 1:]:
                        w = row.get(b)
                        if w is not None:
                            ptot += w
        ext = self._extra_aggregate(members) if self.has_extra else None
        return cnt, s2, pn, lsum, mm, s1, sq, gc, gu, ptot, ext, mp

    def _extra_aggregate(self, members: Sequence[int]) -> Tuple:
        nb = len(self.bound_rules)
        bm = [0] * nb
        bs = [0.0] * nb
        blo = [math.inf] * nb
        bhi = [-math.inf] * nb
        if nb:
            for i in members:
                for r, v in enumerate(self.bound_values[i]):
                    if v is not None:
                        bm[r] += 1
                        bs[r] += v
                        if v < blo[r]:
                            blo[r] = v
                        if v > bhi[r]:
                            bhi[r] = v
        sc = [[0] * rule.n_values for rule in self.shared_rules]
        if sc:
            for i in members:
                for r, codes in enumerate(self.shared_codes[i]):
                    held = sc[r]
                    for c in codes:
                        held[c] += 1
        members = list(members)
        size = len(members)
        sh = [self._shared_count(r, sc[r], members, size) for r in range(len(sc))]
        return bm, bs, blo, bhi, sc, sh

    def _shared_count(self, r: int, held: Sequence[int], members: Sequence[int],
                      size: int) -> int:
        if size <= 0 or not members:
            return 0
        return sum(1 for c in self.shared_codes[members[0]][r] if held[c] == size)

    def anchor_cost(self, assignment: Sequence[int]) -> float:
        anchor = self.anchor
        if anchor is None:
            return 0.0
        away = sum(1 for i, t in enumerate(assignment) if 0 <= anchor[i] != t)
        return away * self.anchor_weight

    def _values(self, size: int, s2: Sequence[int], pn: Sequence[int], lsum: Sequence[int],
                mm: Sequence[int], s1: Sequence[float], sq: Sequence[float],
                gu: Sequence[float], ptot: float, ext: Optional[Tuple] = None,
                mp: Optional[Sequence[int]] = None) -> List[float]:
        out: List[float] = []
        nsingle = self.n_cat_single
        for r, rule in enumerate(self.cat_rules):
            c = rule.coef[size] if size < len(rule.coef) else 0.0
            if c == 0.0:
                out.append(0.0)
                continue
            if rule.multi and rule.goal == 2:
                out.append(_multi_cluster_value(c, mp[r - nsingle], pn[r], s2[r]))
                continue
            out.append(_cat_value(rule.goal, c, size, s2[r], pn[r], lsum[r], rule.n_pool, rule.sqn))
        for r, rule in enumerate(self.num_rules):
            out.append(rule.value(mm[r], s1[r], sq[r]))
        for r, grule in enumerate(self.group_rules):
            out.append(gu[r] * grule.cost)
        if self.has_extra:
            bm, bs, blo, bhi, _, sh = ext if ext is not None else _EMPTY_EXT
            for r, brule in enumerate(self.bound_rules):
                out.append(brule.value(bm[r], bs[r], blo[r], bhi[r]) if bm else 0.0)
            for r, srule in enumerate(self.shared_rules):
                out.append(srule.value(sh[r], size) if sh else 0.0)
        out.append(ptot)
        return out

    def _team_values(self, members: Sequence[int]) -> List[float]:
        cnt, s2, pn, lsum, mm, s1, sq, gc, gu, ptot, ext, mp = self._aggregate(members)
        return self._values(len(members), s2, pn, lsum, mm, s1, sq, gu, ptot, ext, mp)

    def team_penalty(self, members: Sequence[int]) -> float:
        return float(sum(self._team_values(list(members))))

    def team_breakdown(self, members: Sequence[int]) -> Dict[str, float]:
        members = list(members)
        values = self._team_values(members)
        out: Dict[str, float] = {}
        for k in range(self.pair_term):
            out[self.terms[k].label] = values[k]
        if (self.has_history_terms or self.has_cap_term or self.has_pref_term
                or self.has_pair_penalty_term):
            parts = self._pair_parts(members)
            if self.has_history_terms:
                out[HISTORY_LABEL] = parts[0]
            if self.has_cap_term:
                out[HISTORY_CAP_LABEL] = parts[1]
            if self.has_pref_term:
                out[PREFERENCES_LABEL] = parts[2]
            if self.has_pair_penalty_term:
                out[PAIR_PENALTY_LABEL] = parts[3]
        return out

    def _pair_parts(self, members: Sequence[int]) -> Tuple[float, float, float, float]:
        soft = cap = pref = extra = 0.0
        ms = sorted(members)
        for a_pos, a in enumerate(ms):
            for b in ms[a_pos + 1:]:
                pair = (a, b)
                soft += self.hist_soft.get(pair, 0.0)
                cap += self.hist_cap.get(pair, 0.0)
                pref += self.pref_cost.get(pair, 0.0)
                extra += self.pair_penalties.get(pair, 0.0)
        return soft, cap, pref, extra

    def bounds_stats(self, r: int, members: Sequence[int]) -> Dict[str, Any]:
        brule = self.bound_rules[r]
        vals = [self.bound_values[i][r] for i in members]
        vals = [v for v in vals if v is not None]
        m = len(vals)
        if not m:
            return {"count": 0, "mean": None, "min": None, "max": None, "range": None,
                    "excess_sd": 0.0, "broken": 0}
        s1 = sum(vals)
        lo, hi = min(vals), max(vals)
        excess, broken = brule.units(m, s1, lo, hi)
        return {"count": m, "mean": s1 / m + brule.shift, "min": lo + brule.shift,
                "max": hi + brule.shift, "range": hi - lo, "excess_sd": excess,
                "broken": broken}

    def shared_stats(self, r: int, members: Sequence[int]) -> Dict[str, Any]:
        srule = self.shared_rules[r]
        members = list(members)
        held = [0] * srule.n_values
        for i in members:
            for c in self.shared_codes[i][r]:
                held[c] += 1
        size = len(members)
        common = ([c for c in self.shared_codes[members[0]][r] if held[c] == size]
                  if members else [])
        return {"shared": len(common), "common": common, "units": srule.units(len(common), size),
                "held": held}

    def team_hard_units_exact(self, members: Sequence[int]) -> float:
        members = list(members)
        _, _, _, _, _, _, _, _, gu, _, ext, _ = self._aggregate(members)
        units = sum(gu[r] for r, g in enumerate(self.group_rules) if g.hard)
        if ext is not None:
            bm, bs, blo, bhi, _, sh = ext
            for r, brule in enumerate(self.bound_rules):
                if brule.hard:
                    units += brule.units(bm[r], bs[r], blo[r], bhi[r])[1]
            for r, srule in enumerate(self.shared_rules):
                if srule.hard:
                    units += srule.units(sh[r], len(members))
        if self.pair_hard:
            ms = sorted(members)
            for a_pos, a in enumerate(ms):
                for b in ms[a_pos + 1:]:
                    units += self.pair_hard.get((a, b), 0)
        return float(units)

    def hard_units(self, members: Sequence[int]) -> int:
        return int(math.ceil(self.team_hard_units_exact(members) - 1e-9))

    def assignment_hard_units(self, assignment: Sequence[int]) -> int:
        teams = self.teams_of(assignment)
        return int(round(sum(self.team_hard_units_exact(m) for m in teams)))

    def teams_of(self, assignment: Sequence[int]) -> List[List[int]]:
        if len(assignment) != self.n:
            raise ValueError(f"Assignment has {len(assignment)} entries for {self.n} students")
        teams: List[List[int]] = [[] for _ in range(self.n_teams)]
        for i, t in enumerate(assignment):
            if t is None or not 0 <= int(t) < self.n_teams:
                raise ValueError(f"Student index {i} has invalid team {t!r}")
            teams[int(t)].append(i)
        return teams

    def team_penalties(self, assignment: Sequence[int]) -> List[float]:
        return [self.team_penalty(m) for m in self.teams_of(assignment)]

    def aggregate(self, penalties: Sequence[float]) -> float:
        if not penalties:
            return 0.0
        w = self.fairness_weight
        total = math.fsum(penalties)
        if w == 0.0:
            return total
        return (1.0 - w) * total + w * len(penalties) * max(penalties)

    def evaluate(self, assignment: Sequence[int]) -> float:
        value = self.aggregate(self.team_penalties(assignment))
        if self.anchor is not None:
            value += self.anchor_cost(assignment)
        return value

    @property
    def term_labels(self) -> List[str]:
        return [t.label for t in self.terms]

    @property
    def has_hard_terms(self) -> bool:
        return bool(self.pair_hard) or self.has_hard_team_rules

    @property
    def has_hard_team_rules(self) -> bool:
        return (any(g.hard for g in self.group_rules) or any(b.hard for b in self.bound_rules)
                or any(s.hard for s in self.shared_rules))

    def with_hard_penalty(self, hard_penalty: float) -> "ObjectiveModel":
        return model_from_args(self.rebuild_args(policy=self.policy, hard_penalty=hard_penalty,
                                                 pair_hard_penalty=None))

    def with_relaxed_team_rules(self, hard_penalty: float) -> "ObjectiveModel":
        return model_from_args(self.rebuild_args(policy=self.policy, hard_penalty=hard_penalty,
                                                 pair_hard_penalty=self.pair_hard_penalty))

    def without_extras(self) -> "ObjectiveModel":
        return model_from_args(self.rebuild_args(policy=self.policy, anchor=None,
                                                 anchor_weight=0.0, pair_penalties=None))

    def rebuild_args(self, **overrides: Any) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "cohort": self.cohort, "policy": self.policy.to_dict(),
            "team_sizes": list(self.team_sizes), "history": self.history,
            "project_id": self.project_id, "hard_penalty": self.hard_penalty,
            "pair_hard_penalty": (None if self.pair_hard_penalty == self.hard_penalty
                                  else self.pair_hard_penalty),
            "anchor": None if self.anchor is None else list(self.anchor),
            "anchor_weight": self.anchor_weight,
            "pair_penalties": dict(self.pair_penalties) or None,
        }
        args.update(overrides)
        return args


def model_from_args(args: Dict[str, Any]) -> ObjectiveModel:
    args = dict(args)
    policy = args.pop("policy")
    if not isinstance(policy, Policy):
        policy = Policy.from_dict(policy)
    return ObjectiveModel(args.pop("cohort"), policy, args.pop("team_sizes"), **args)


_EMPTY_EXT: Tuple = ((), (), (), (), (), ())


def _multi_cluster_value(c: float, mp: int, pn: int, s2: int) -> float:
    return (mp * pn - s2) * 0.5 * c


def _cat_value(goal: int, c: float, s: int, s2: int, pn: int, lsum: int, n_pool: int,
               sqn: float) -> float:
    if goal == 0:
        return (s2 - lsum * (2.0 * s / n_pool) + s * s * sqn) * c
    if goal == 1:
        return (s2 - pn) * 0.5 * c
    return (pn * pn - s2) * 0.5 * c

