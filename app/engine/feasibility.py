from __future__ import annotations

import difflib
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .attributes import (
    _fmt_num,
    _range_label,
    loosest_feasible_min,
    redundant_pairs,
    tightest_feasible_max,
    unavoidable_count_violations,
)
from .policy import CountRule, IsolationRule, NumericBoundsRule, PairRule, Policy, SharedValueRule
from .policy_text import rule_headline
from .roster import AttributeInfo, EncodedAttribute, EncodedCohort, infer_attributes
from .sizing import balanced_sizes, describe_sizes, plan_team_sizes, size_problems

LEVELS = ("error", "warning", "info")
LOW_COVERAGE = 0.6


@dataclass
class Finding:
    level: str
    area: str
    rule_label: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeasibilityReport:
    findings: List[Finding] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def infos(self) -> List[Finding]:
        return [f for f in self.findings if f.level == "info"]

    def by_area(self, area: str) -> List[Finding]:
        return [f for f in self.findings if f.area == area]

    def summary(self) -> str:
        def plural(count: int, word: str) -> str:
            return f"{count} {word}{'' if count == 1 else 's'}"
        if not self.findings:
            return "No problems found"
        return ", ".join([plural(len(self.errors), "error"), plural(len(self.warnings), "warning"),
                          plural(len(self.infos), "note")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "elapsed_s": round(self.elapsed_s, 4),
            "findings": [f.to_dict() for f in self.findings],
        }


class ProjectLike(Protocol):
    project_id: Any
    student_ids: Optional[Sequence[str]]


class PriorProjectLike(Protocol):
    project_id: Any
    teams: Sequence[Sequence[str]]


_ALIASES = {
    "project_id": ("project_id", "id", "name"),
    "student_ids": ("student_ids", "participants", "students", "participant_ids"),
    "team_sizes": ("team_sizes", "sizes"),
    "size_min": ("size_min", "min_size", "team_size_min", "min_team_size"),
    "size_max": ("size_max", "max_size", "team_size_max", "max_team_size"),
    "size_target": ("size_target", "target_size", "target", "team_size"),
    "num_teams": ("num_teams", "n_teams", "team_count"),
    "teams": ("teams", "groups"),
}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    for key in _ALIASES.get(name, (name,)):
        if isinstance(obj, Mapping):
            if key in obj:
                return obj[key]
        elif hasattr(obj, key):
            return getattr(obj, key)
    return default


@dataclass
class _Project:
    project_id: str
    pool: List[int]
    sizes: List[int]


def _prior_teams(item: Any) -> Tuple[str, List[List[str]]]:
    if isinstance(item, (tuple, list)) and len(item) == 2 and not isinstance(item[0], (list, tuple)):
        pid, teams = item
    else:
        pid, teams = _get(item, "project_id", "?"), _get(item, "teams", [])
    if isinstance(teams, Mapping):
        teams = list(teams.values())
    return str(pid), [[str(s) for s in team] for team in (teams or [])]


def count_rule_groups(rule: CountRule, value_counts: Mapping[str, int],
                      numeric_values: Optional[Sequence[float]] = None) -> Dict[str, int]:
    if rule.numeric_range is not None:
        lo, hi = (list(rule.numeric_range) + [None, None])[:2]
        matching = sum(1 for x in (numeric_values or [])
                       if (lo is None or x >= lo) and (hi is None or x <= hi))
        return {_range_label(rule.attribute, lo, hi): matching}
    if rule.mode == "each":
        values = list(rule.values) or sorted(value_counts, key=lambda v: (-value_counts[v], v))
        return {str(v): int(value_counts.get(str(v), 0)) for v in values}
    label = ", ".join(map(str, rule.values))
    return {label: int(sum(value_counts.get(str(v), 0) for v in rule.values))}


def count_rule_floor(rule: CountRule, groups: Mapping[str, int],
                     sizes: Sequence[int]) -> Tuple[int, List[Dict[str, Any]]]:
    breakdown = []
    total_max = 0
    total_min = 0
    for group, k in groups.items():
        over = unavoidable_count_violations(k, sizes, max_count=rule.max_count)
        under = unavoidable_count_violations(k, sizes, min_count=rule.min_count)
        if over:
            breakdown.append(dict(group=group, students=k, units=over, side="max",
                                  suggestion=tightest_feasible_max(k, sizes)))
        if under:
            breakdown.append(dict(group=group, students=k, units=under, side="min",
                                  suggestion=loosest_feasible_min(k, sizes)))
        total_max += over
        total_min += under
    if rule.mode == "each" and rule.min_count and len(groups) > 1:
        seats = sum(max(0, len(groups) * rule.min_count - s) for s in sizes)
        if seats > total_min:
            breakdown.append(dict(group="all values", students=None, units=seats - total_min,
                                  side="seats", suggestion=None))
            total_min = seats
    breakdown.sort(key=lambda b: -b["units"])
    return int(total_max + total_min), breakdown


def isolation_floor(matching: int, min_together: int, caps: Sequence[int]) -> int:
    if matching <= 0 or min_together <= 1:
        return 0
    if matching < min_together:
        return min(matching, min_together - matching)
    ordered = sorted((int(c) for c in caps), reverse=True)
    reach = 0
    for g, cap in enumerate(ordered, start=1):
        if cap < min_together:
            break
        reach += cap
        if g * min_together <= matching <= reach:
            return 0
        if g * min_together > matching:
            break
    return 1


@dataclass
class ValueBounds:
    cap: Optional[int] = None
    cap_hard: bool = False
    cap_index: Optional[int] = None
    floor: int = 0
    floor_hard: bool = False
    floor_index: Optional[int] = None


def value_bounds(policy: Policy, attribute: str, value: str) -> ValueBounds:
    bounds = ValueBounds()
    for index, rule in enumerate(policy.counts):
        if not rule.enabled or rule.attribute != attribute or rule.numeric_range is not None:
            continue
        applies = (value in rule.values) if rule.values else rule.mode == "each"
        if not applies:
            continue
        if rule.max_count is not None and (
                bounds.cap is None or rule.max_count < bounds.cap
                or (rule.max_count == bounds.cap and rule.hard and not bounds.cap_hard)):
            bounds.cap, bounds.cap_hard, bounds.cap_index = rule.max_count, rule.hard, index
        single = rule.mode == "each" or list(rule.values) == [value]
        if single and rule.min_count and (
                rule.min_count > bounds.floor
                or (rule.min_count == bounds.floor and rule.hard and not bounds.floor_hard)):
            bounds.floor, bounds.floor_hard, bounds.floor_index = rule.min_count, rule.hard, index
    return bounds


def _plural(count: int, word: str, plural: Optional[str] = None) -> str:
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def _close_matches(value: str, options: Sequence[str]) -> List[str]:
    lowered = {o.lower(): o for o in options}
    if value.lower() in lowered:
        return [lowered[value.lower()]]
    hits = difflib.get_close_matches(value.lower(), list(lowered), n=3, cutoff=0.75)
    return [lowered[h] for h in hits]


def _infos_from_cohort(cohort: EncodedCohort) -> Dict[str, AttributeInfo]:
    infos: Dict[str, AttributeInfo] = {}
    n = cohort.n or 1
    for name, attr in cohort.attributes.items():
        present = sum(1 for i in range(cohort.n) if attr.present(i))
        counts: Dict[str, int] = {}
        stats: Dict[str, float] = {}
        if attr.kind == "categorical":
            for code in attr.codes:
                if code >= 0:
                    counts[attr.categories[code]] = counts.get(attr.categories[code], 0) + 1
        elif attr.kind == "multi":
            for codes in attr.multi:
                for code in set(codes):
                    counts[attr.categories[code]] = counts.get(attr.categories[code], 0) + 1
        else:
            nums = [v for v in attr.values if not math.isnan(v)]
            for v in nums:
                key = _fmt_num(v)
                counts[key] = counts.get(key, 0) + 1
            if nums:
                mean = sum(nums) / len(nums)
                stats = {"min": min(nums), "max": max(nums), "mean": mean,
                         "std": math.sqrt(sum((x - mean) ** 2 for x in nums) / len(nums))}
        values = sorted(counts, key=lambda k: (-counts[k], k))
        kind = attr.kind
        if len(values) == 1 and kind != "multi":
            kind = "constant"
        elif not values:
            kind = "empty"
        infos[name] = AttributeInfo(name=name, kind=kind, coverage=present / n,
                                    n_unique=len(values), values=values, counts=counts,
                                    stats=stats)
    return infos


def _frame_from_cohort(cohort: EncodedCohort) -> pd.DataFrame:
    data: Dict[str, List[Any]] = {"student_id": list(cohort.student_ids)}
    for name, attr in cohort.attributes.items():
        if attr.kind == "categorical":
            data[name] = [attr.categories[c] if c >= 0 else None for c in attr.codes]
        elif attr.kind == "numeric":
            data[name] = [None if math.isnan(v) else v for v in attr.values]
        else:
            data[name] = ["|".join(attr.categories[c] for c in codes) or None
                          for codes in attr.multi]
    return pd.DataFrame(data)


def _encoding_overrides(policy: Policy, infos: Dict[str, AttributeInfo]) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for rule in policy.attributes:
        if rule.kind:
            kinds[rule.attribute] = rule.kind
    for rule in [*policy.counts, *policy.isolation]:
        info = infos.get(rule.attribute)
        if info is None or rule.attribute in kinds:
            continue
        if isinstance(rule, CountRule) and rule.numeric_range is not None:
            if info.kind != "numeric":
                kinds[rule.attribute] = "numeric"
        elif info.kind in ("text", "constant"):
            kinds[rule.attribute] = "categorical"
    return kinds


def _construct(forb: np.ndarray, weights: np.ndarray, sizes: Sequence[int],
               rng: np.random.Generator) -> np.ndarray:
    n, teams = len(forb), len(sizes)
    capacity = np.array(sizes, dtype=np.int64)
    conflicts = np.zeros((n, teams), dtype=np.int64)
    assign = -np.ones(n, dtype=np.int64)
    degree = forb.sum(axis=1)
    noise = rng.random(n)
    for _ in range(n):
        open_teams = capacity > 0
        free = (conflicts == 0) & open_teams
        options = free.sum(axis=1).astype(float)
        options[assign >= 0] = np.inf
        key = options * 1e6 - degree * 10.0 - noise
        i = int(np.argmin(key))
        choices = np.flatnonzero(free[i])
        if len(choices) == 0:
            choices = np.flatnonzero(open_teams)
            best = conflicts[i, choices].min()
            choices = choices[conflicts[i, choices] == best]
        else:
            room = capacity[choices]
            choices = choices[room == room.max()]
        t = int(rng.choice(choices))
        assign[i] = t
        capacity[t] -= 1
        conflicts[:, t] += weights[i]
    return assign


def _repair(forb: np.ndarray, weights: np.ndarray, assign: np.ndarray,
            rng: np.random.Generator, deadline: float, max_steps: int = 4000) -> int:
    n = len(assign)
    teams = int(assign.max()) + 1
    member = np.zeros((n, teams), dtype=np.int64)
    member[np.arange(n), assign] = 1
    conflicts = weights @ member
    idx = np.arange(n)
    for step in range(max_steps):
        own = conflicts[idx, assign]
        total = int(own.sum()) // 2
        if total == 0:
            return 0
        if step % 64 == 0 and time.perf_counter() > deadline:
            return total
        i = int(rng.choice(np.flatnonzero(own > 0)))
        a = assign[i]
        delta = (conflicts[i, assign] + conflicts[:, a] - 2 * weights[i]
                 - conflicts[i, a] - own)
        delta[assign == a] = np.iinfo(np.int64).max // 4
        best = delta.min()
        if best > 0 and rng.random() > 0.1:
            continue
        candidates = np.flatnonzero(delta == best) if best <= 0 else np.flatnonzero(delta <= best + 1)
        j = int(rng.choice(candidates))
        b = assign[j]
        conflicts[:, a] += weights[j] - weights[i]
        conflicts[:, b] += weights[i] - weights[j]
        assign[i], assign[j] = b, a
    return int(conflicts[idx, assign].sum()) // 2


def probe_forbidden_partition(forbidden: np.ndarray, sizes: Sequence[int], *,
                              time_budget_s: float = 0.3,
                              seed: Optional[int] = 0) -> Dict[str, Any]:
    start = time.perf_counter()
    deadline = start + max(0.0, time_budget_s)
    forb = np.asarray(forbidden, dtype=bool)
    weights = forb.astype(np.int64)
    rng = np.random.default_rng(seed)
    best, best_assign, attempts = None, None, 0
    while True:
        attempts += 1
        assign = _construct(forb, weights, sizes, rng)
        conflicts = _repair(forb, weights, assign, rng, deadline)
        if best is None or conflicts < best:
            best, best_assign = conflicts, assign.copy()
        if best == 0 or time.perf_counter() >= deadline:
            break
    return {
        "found": best == 0,
        "best_conflicts": int(best or 0),
        "attempts": attempts,
        "elapsed_s": time.perf_counter() - start,
        "assignment": best_assign.tolist() if best_assign is not None else None,
    }


class _Checker:
    def __init__(self, source: Union[pd.DataFrame, EncodedCohort], policy: Policy,
                 probe_time_s: float, seed: Optional[int]) -> None:
        self.policy = policy
        self.report = FeasibilityReport()
        self.probe_time_s = probe_time_s
        self.seed = seed if seed is not None else (policy.solver.seed or 0)
        if isinstance(source, EncodedCohort):
            self.cohort = source
            self.infos = _infos_from_cohort(source)
            self.frame: Optional[pd.DataFrame] = None
            self.raw_infos = self.infos
        else:
            self.frame = source
            self.raw_infos = infer_attributes(source)
            overrides = _encoding_overrides(policy, self.raw_infos)
            orders = {r.attribute: r.order for r in policy.attributes if r.order}
            missing = [r.attribute for r in policy.attributes if r.missing_as_category]
            self.cohort = EncodedCohort.from_frame(source, kinds=overrides, orders=orders,
                                                   missing_as_category=missing)
            self.infos = dict(self.raw_infos)
            self.infos.update({k: v for k, v in _infos_from_cohort(self.cohort).items()
                               if k in overrides})
        self.multi_project = False
        self._pair_counts: Optional[Dict[Tuple[int, int], int]] = None
        self._hard_together: List[Tuple[int, List[int]]] = []

    def add(self, level: str, area: str, message: str, rule_label: str = "",
            project: Optional[_Project] = None, **detail: Any) -> None:
        if project is not None:
            detail["project_id"] = project.project_id
            if self.multi_project:
                message = f"Project {project.project_id}: {message}"
        self.report.findings.append(Finding(level, area, rule_label, message, detail))

    def attr(self, name: str) -> Optional[EncodedAttribute]:
        return self.cohort.attributes.get(name)

    def pool_counts(self, attr: EncodedAttribute, pool: Sequence[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if attr.kind == "categorical":
            for i in pool:
                code = attr.codes[i]
                if code >= 0:
                    counts[attr.categories[code]] = counts.get(attr.categories[code], 0) + 1
        elif attr.kind == "multi":
            for i in pool:
                for code in set(attr.multi[i]):
                    counts[attr.categories[code]] = counts.get(attr.categories[code], 0) + 1
        return counts

    def any_of_count(self, attr: EncodedAttribute, pool: Sequence[int],
                     values: Iterable[str]) -> int:
        wanted = {attr.code_of(v) for v in values} - {-1}
        if attr.kind == "categorical":
            return sum(1 for i in pool if attr.codes[i] in wanted)
        if attr.kind == "multi":
            return sum(1 for i in pool if wanted.intersection(attr.multi[i]))
        return 0

    def resolve_projects(self, projects: Optional[Sequence[Any]],
                         default_team_size: int) -> List[_Project]:
        everyone = list(range(self.cohort.n))
        if not projects:
            sizes = plan_team_sizes(len(everyone), default_team_size, default_team_size,
                                    default_team_size)
            self.add("info", "sizes",
                     f"No projects given: checked one project with all {len(everyone)} "
                     f"students ({describe_sizes(sizes)}).")
            return [_Project("all", everyone, sizes)] if sizes else []
        self.multi_project = len(projects) > 1
        resolved = []
        for k, raw in enumerate(projects):
            pid = str(_get(raw, "project_id", k + 1))
            ids = _get(raw, "student_ids", None)
            project = _Project(pid, [], [])
            if ids is None:
                project.pool = list(everyone)
            else:
                unknown, seen, duplicates = [], set(), []
                for sid in ids:
                    i = self.cohort.lookup(sid)
                    if i is None:
                        unknown.append(str(sid))
                    elif i in seen:
                        duplicates.append(str(sid))
                    else:
                        seen.add(i)
                        project.pool.append(i)
                if unknown:
                    self.add("error", "pool",
                             f"{_plural(len(unknown), 'participant id')} not in the roster "
                             f"({', '.join(unknown[:5])}{' …' if len(unknown) > 5 else ''}). "
                             "Remove them from the project or upload a roster that includes them.",
                             project=project, unknown_ids=unknown)
                if duplicates:
                    self.add("warning", "pool",
                             f"{_plural(len(duplicates), 'participant')} listed twice "
                             f"({', '.join(duplicates[:5])}); each is placed once.",
                             project=project, duplicate_ids=duplicates)
            n = len(project.pool)
            if n == 0:
                self.add("error", "pool", "The project has no participants. Add students "
                         "to it or remove the project.", project=project)
                continue
            project.sizes = self.resolve_sizes(raw, project, n)
            if project.sizes:
                resolved.append(project)
        return resolved

    def resolve_sizes(self, raw: Any, project: _Project, n: int) -> List[int]:
        explicit = _get(raw, "team_sizes", None)
        size_min = _get(raw, "size_min", None)
        size_max = _get(raw, "size_max", None)
        target = _get(raw, "size_target", None)
        num_teams = _get(raw, "num_teams", 0) or 0
        if explicit:
            sizes = [int(s) for s in explicit]
            if any(s <= 0 for s in sizes):
                self.add("error", "sizes", "Team sizes must all be positive.", project=project,
                         sizes=sizes)
                return []
            if sum(sizes) != n:
                self.add("error", "sizes",
                         f"The team sizes add up to {sum(sizes)} but the project has {n} "
                         "participants. Adjust the sizes or the participant list.",
                         project=project, sizes=sizes, participants=n)
                return []
            if size_min is not None and size_max is not None:
                for problem in size_problems(sizes, int(size_min), int(size_max)):
                    self.add("warning", "sizes", f"{problem}.", project=project, sizes=sizes)
            return sizes
        if size_min is None and size_max is None and target is None and not num_teams:
            self.add("error", "sizes", "No team sizes given: set a minimum, maximum and target "
                     "size or the number of teams.", project=project)
            return []
        bounded = size_min is not None or size_max is not None
        lo = int(size_min) if size_min is not None else 1
        hi = int(size_max) if size_max is not None else n
        if lo > hi:
            self.add("error", "sizes",
                     f"The minimum team size ({lo}) exceeds the maximum ({hi}).", project=project)
            return []
        if target is None and size_min is None and size_max is None:
            sizes = balanced_sizes(n, min(int(num_teams), n))
        else:
            target = int(target if target is not None else (size_max or size_min))
            sizes = plan_team_sizes(n, lo, hi, target, int(num_teams))
        problems = size_problems(sizes, lo, hi) if bounded else []
        if bounded and n < lo:
            self.add("error", "sizes",
                     f"Only {n} participants, fewer than the minimum team size of {lo}. "
                     "Lower the minimum or add participants.", project=project)
        elif problems:
            hint = ("change the number of teams" if num_teams
                    else "widen the size range or change the target size")
            self.add("error", "sizes",
                     f"{n} participants cannot be split within {lo}–{hi} per team: "
                     f"{'; '.join(problems)}. To fix, {hint}.",
                     project=project, sizes=sizes)
        else:
            self.add("info", "sizes", f"{n} participants → {describe_sizes(sizes)}.",
                     project=project, sizes=sizes)
        return sizes

    def check_policy_structure(self) -> None:
        kinds = {name: info.kind for name, info in self.infos.items()}
        for rule in self.policy.attributes:
            if rule.kind:
                kinds[rule.attribute] = rule.kind
        for problem in self.policy.validate(kinds):
            if "is not in the roster" in problem:
                continue
            self.add("error", "policy", f"{problem}.")

    def check_attributes(self) -> None:
        roles: Dict[str, List[Tuple[str, Any, int, bool]]] = defaultdict(list)
        for i, rule in enumerate(self.policy.attributes):
            if rule.enabled and rule.goal != "ignore" and rule.weight > 0:
                roles[rule.attribute].append(("attribute", rule, i, False))
        for i, rule in enumerate(self.policy.counts):
            if rule.enabled:
                roles[rule.attribute].append(("count", rule, i, rule.hard))
        for i, rule in enumerate(self.policy.isolation):
            if rule.enabled:
                roles[rule.attribute].append(("isolation", rule, i, rule.hard))
        for i, rule in enumerate(self.policy.numeric_bounds):
            if rule.enabled:
                roles[rule.attribute].append(("bounds", rule, i, rule.hard))
        for i, rule in enumerate(self.policy.shared):
            if rule.enabled:
                roles[rule.attribute].append(("shared", rule, i, rule.hard))

        columns = list(self.raw_infos)
        for name, uses in roles.items():
            label = _rule_label(uses[0][1])
            info = self.infos.get(name)
            if info is None:
                hard = any(h for _, _, _, h in uses)
                matches = _close_matches(name, columns)
                hint = f" Did you mean '{matches[0]}'?" if matches else ""
                self.add("error" if hard else "warning", "attributes",
                         f"'{name}' is not a column of the roster, so "
                         f"{_plural(len(uses), 'rule')} on it cannot be applied.{hint}",
                         rule_label=label, attribute=name, suggestions=matches,
                         rule_type=uses[0][0], rule_index=uses[0][2])
                continue
            if info.kind == "empty":
                self.add("warning", "attributes",
                         f"'{name}' has no values; rules on it have no effect.",
                         rule_label=label, attribute=name)
                continue
            if info.kind == "constant":
                self.add("info", "attributes",
                         f"Everyone has the same {name} ('{info.values[0]}'); rules on it "
                         "have no effect and can be disabled.",
                         rule_label=label, attribute=name)
            if info.kind == "text" and any(t == "attribute" for t, _, _, _ in uses):
                self.add("warning", "attributes",
                         f"'{name}' looks like free text ({info.n_unique} distinct values for "
                         f"{round(info.coverage * self.cohort.n)} students), so it is only used "
                         "if its kind is set to categorical — and then every spelling is its "
                         "own category. Prefer a coarser column or clean the values.",
                         rule_label=label, attribute=name)
            if info.coverage < LOW_COVERAGE:
                self.add("warning", "attributes",
                         f"Only {info.coverage:.0%} of students have a value for '{name}'. "
                         "Students without one are ignored by its rules; fill in the data or "
                         "treat missing as its own category.",
                         rule_label=label, attribute=name, coverage=info.coverage)
            self.check_rule_values(name, uses)

        self.check_double_counting()

    def check_rule_values(self, name: str, uses: List[Tuple[str, Any, int, bool]]) -> None:
        attr = self.attr(name)
        if attr is None or attr.kind not in ("categorical", "multi"):
            return
        for rule_type, rule, index, hard in uses:
            if rule_type == "attribute":
                continue
            for value in getattr(rule, "values", []) or []:
                if attr.code_of(value) >= 0:
                    continue
                matches = _close_matches(str(value), attr.categories)
                hint = (f" Did you mean {', '.join(repr(m) for m in matches)}?" if matches
                        else f" Known values: {', '.join(attr.categories[:8])}"
                             f"{' …' if len(attr.categories) > 8 else ''}.")
                self.add("warning", "attributes",
                         f"No student has {name} = '{value}', so that part of the rule never "
                         f"applies.{hint}",
                         rule_label=_rule_label(rule), attribute=name, value=value,
                         suggestions=matches, rule_type=rule_type, rule_index=index)

    def check_double_counting(self) -> None:
        weighted = {r.attribute: r for r in self.policy.active_attributes()
                    if r.attribute in self.infos}
        names = [n for n in weighted if self.infos[n].kind in ("categorical", "numeric", "text")]
        if len(names) < 2:
            return
        frame = self.frame if self.frame is not None else _frame_from_cohort(self.cohort)
        for pair in redundant_pairs(frame, attributes=names, infos=self.raw_infos):
            a, b = weighted[pair["a"]], weighted[pair["b"]]
            self.add("info", "attributes",
                     f"Double counting: {pair['explanation']}. Both {a.attribute} (weight "
                     f"{_fmt_num(a.weight)}) and {b.attribute} (weight {_fmt_num(b.weight)}) are "
                     "used, so what they share counts twice — fine if intended, otherwise "
                     "ignore one or lower its weight.",
                     rule_label=f"{pair['a']} + {pair['b']}",
                     attributes=[pair["a"], pair["b"]], relation=pair["relation"],
                     strength=pair["strength"])

    def check_counts(self, project: _Project) -> None:
        sizes = project.sizes
        for index, rule in enumerate(self.policy.counts):
            if not rule.enabled or (rule.min_count is None and rule.max_count is None):
                continue
            attr = self.attr(rule.attribute)
            if attr is None:
                continue
            label = _rule_label(rule)
            if rule.numeric_range is not None:
                if attr.kind != "numeric":
                    continue
                numbers = [attr.values[i] for i in project.pool if not math.isnan(attr.values[i])]
                groups = count_rule_groups(rule, {}, numbers)
            elif attr.kind == "numeric":
                continue
            elif rule.mode == "any_of":
                groups = {", ".join(rule.values): self.any_of_count(attr, project.pool, rule.values)}
            else:
                groups = count_rule_groups(rule, self.pool_counts(attr, project.pool))
            units, breakdown = count_rule_floor(rule, groups, sizes)
            if units == 0:
                continue
            self.add("error" if rule.hard else "warning", "counts",
                     _count_message(rule, breakdown, sizes, units),
                     rule_label=label, project=project, rule_type="count", rule_index=index,
                     units=units, breakdown=breakdown)

    def check_isolation(self, project: _Project) -> None:
        sizes = project.sizes
        teams = len(sizes)
        for index, rule in enumerate(self.policy.isolation):
            if not rule.enabled:
                continue
            attr = self.attr(rule.attribute)
            if attr is None or attr.kind not in ("categorical", "multi"):
                continue
            label = _rule_label(rule)
            counts = self.pool_counts(attr, project.pool)
            values = list(rule.values) or sorted(counts, key=lambda v: -counts[v])
            k = rule.min_together
            level = "error" if rule.hard else "warning"
            for value in values:
                matching = counts.get(value, 0)
                if matching == 0:
                    continue
                bounds = value_bounds(self.policy, rule.attribute, value)
                cap, cap_hard, cap_index = bounds.cap, bounds.cap_hard, bounds.cap_index
                floor, floor_hard, floor_index = (bounds.floor, bounds.floor_hard,
                                                  bounds.floor_index)
                detail = dict(rule_type="isolation", rule_index=index, value=value,
                              students=matching)
                if matching == 1:
                    self.add(level, "isolation",
                             f"Only one student has {rule.attribute} = {value}, so they are "
                             "always the only one in their team and 'no lone' can never hold "
                             f"for them. Remove '{value}' from the rule"
                             f"{'' if rule.hard else ' (costs ' + _fmt_num(rule.weight) + ' points otherwise)'}.",
                             rule_label=label, project=project, **detail)
                    continue
                if cap is not None and cap < k:
                    both_hard = rule.hard and cap_hard
                    self.add("error" if both_hard else "warning", "isolation",
                             f"Contradiction: at most {_plural(cap, 'student')} with "
                             f"{rule.attribute} = {value} per team, but any team with one needs "
                             f"at least {k}. Every team that gets one of these students violates "
                             f"one of the two rules; raise the maximum to {k} or lower "
                             "min_together.",
                             rule_label=label, project=project, count_rule_index=cap_index,
                             **detail)
                    continue
                caps = [s if cap is None else min(s, cap) for s in sizes]
                units = isolation_floor(matching, k, caps)
                if units:
                    self.add(level, "isolation",
                             f"{matching} students have {rule.attribute} = {value}; they cannot "
                             f"be grouped at least {k} at a time within the team sizes"
                             f"{' and the head-count cap' if cap is not None else ''}, so at "
                             f"least {_plural(units, 'student')} will be isolated. Lower "
                             "min_together or remove the value from the rule.",
                             rule_label=label, project=project, units=units, **detail)
                    continue
                if floor >= 1:
                    per_team = max(floor, k)
                    need = sum(min(per_team, s) for s in sizes)
                    if need > matching:
                        both_hard = rule.hard and floor_hard
                        self.add(
                            "error" if both_hard else "warning", "isolation",
                            f"'At least {floor} {rule.attribute} = {value}' together with 'no "
                            f"lone {value}' needs {per_team} such students in each of {teams} "
                            f"teams ({need} in total) but there are only {matching}. Drop one "
                            "of the two rules for this value.",
                            rule_label=label, project=project, count_rule_index=floor_index,
                            units=need - matching, **detail)

    def check_bounds(self, project: _Project) -> None:
        sizes = project.sizes
        teams = len(sizes)
        for index, rule in enumerate(self.policy.numeric_bounds):
            if not rule.enabled:
                continue
            attr = self.attr(rule.attribute)
            if attr is None or attr.kind != "numeric":
                continue
            label = _rule_label(rule)
            level = "error" if rule.hard else "warning"
            detail = dict(rule_type="bounds", rule_index=index)
            values = sorted(attr.values[i] for i in project.pool if not math.isnan(attr.values[i]))
            if len(values) < 2 or values[-1] - values[0] <= 1e-12:
                self.add("warning", "bounds",
                         f"'{label}' is ignored: fewer than two different {rule.attribute} "
                         "values in this project.", rule_label=label, project=project, **detail)
                continue
            fillers = len(project.pool) - len(values)
            mean = sum(values) / len(values)
            lo, hi = rule.mean_min, rule.mean_max
            sides: List[str] = []
            if lo is not None and lo > values[-1] + 1e-9:
                self.add(level, "bounds",
                         f"'{label}': no student has {rule.attribute} above "
                         f"{_fmt_num(values[-1])}, so no team can average at least {_fmt_num(lo)}. "
                         "Lower the minimum.", rule_label=label, project=project,
                         pool_mean=mean, **detail)
            elif lo is not None and mean < lo - 1e-9:
                self.add(level, "bounds",
                         f"'{label}': the project's average {rule.attribute} is {_fmt_num(mean)}, "
                         f"below the minimum team average of {_fmt_num(lo)}, so at least one team "
                         f"must fall short. Lower the minimum to {_fmt_num(math.floor(mean * 100) / 100)} "
                         "or less.", rule_label=label, project=project, pool_mean=mean, **detail)
            elif lo is not None:
                sides.append("min")
            if hi is not None and hi < values[0] - 1e-9:
                self.add(level, "bounds",
                         f"'{label}': no student has {rule.attribute} below "
                         f"{_fmt_num(values[0])}, so no team can average at most {_fmt_num(hi)}. "
                         "Raise the maximum.", rule_label=label, project=project,
                         pool_mean=mean, **detail)
            elif hi is not None and mean > hi + 1e-9:
                self.add(level, "bounds",
                         f"'{label}': the project's average {rule.attribute} is {_fmt_num(mean)}, "
                         f"above the maximum team average of {_fmt_num(hi)}, so at least one team "
                         f"must exceed it. Raise the maximum to {_fmt_num(math.ceil(mean * 100) / 100)} "
                         "or more.", rule_label=label, project=project, pool_mean=mean, **detail)
            elif hi is not None:
                sides.append("max")
            if sides:
                self._check_extreme_means(project, rule, label, level, detail, values, sides)
            if rule.max_range is None:
                continue
            width = float(rule.max_range)
            self._check_range(project, rule, label, level, detail, values, fillers, width, teams)

    def _check_extreme_means(self, project: _Project, rule: NumericBoundsRule, label: str,
                             level: str, detail: Dict[str, Any], values: List[float],
                             sides: Sequence[str] = ("max", "min")) -> None:
        largest = max(project.sizes)
        lo, hi = rule.mean_min, rule.mean_max
        for side, bound in (("max", hi), ("min", lo)):
            if bound is None or side not in sides:
                continue
            ordered = values if side == "max" else values[::-1]
            extreme: List[float] = []
            best_mean = None
            for pos in range(len(values) - 1, -1, -1):
                v = ordered[pos]
                if (side == "max" and v <= bound) or (side == "min" and v >= bound):
                    break
                total, count = v, 1
                for other in (ordered[:pos] + ordered[pos + 1:])[:largest - 1]:
                    if (side == "max" and other >= total / count) or (
                            side == "min" and other <= total / count):
                        break
                    total += other
                    count += 1
                mean = total / count
                if (side == "max" and mean > bound + 1e-9) or (
                        side == "min" and mean < bound - 1e-9):
                    extreme.append(v)
                    best_mean = mean if best_mean is None else best_mean
                else:
                    break
            if not extreme:
                continue
            shown = ", ".join(_fmt_num(v) for v in sorted(set(extreme))[:5])
            word = "maximum" if side == "max" else "minimum"
            fix = "Raise the maximum" if side == "max" else "Lower the minimum"
            self.add(level, "bounds",
                     f"'{label}': {_plural(len(extreme), 'student')} with {rule.attribute} "
                     f"{shown} cannot be in any team that meets the {word} average of "
                     f"{_fmt_num(bound)}: even with the most favourable classmates a team of "
                     f"{largest} averages {_fmt_num(round(best_mean, 2))}. {fix}"
                     f"{'' if rule.hard else ' (or accept the penalty)'}.",
                     rule_label=label, project=project, students=len(extreme),
                     best_mean=best_mean, side=side, **detail)

    def _check_range(self, project: _Project, rule: NumericBoundsRule, label: str, level: str,
                     detail: Dict[str, Any], values: List[float], fillers: int, width: float,
                     teams: int) -> None:
        sizes = project.sizes
        smallest = min(sizes)
        reach = []
        end = 0
        for start, v in enumerate(values):
            end = max(end, start)
            while end + 1 < len(values) and values[end + 1] <= v + width + 1e-9:
                end += 1
            reach.append(end - start + 1)
        lonely = []
        left = 0
        for k, v in enumerate(values):
            while values[left] < v - width - 1e-9:
                left += 1
            if max(reach[left:k + 1]) + fillers < smallest:
                lonely.append(v)
        windows = 0
        k = 0
        while k < len(values):
            windows += 1
            reach = values[k] + width + 1e-9
            while k < len(values) and values[k] <= reach:
                k += 1
        extreme = ", ".join(_fmt_num(v) for v in sorted(set(lonely))[:5])
        if lonely:
            self.add(level, "bounds",
                     f"'{label}': {_plural(len(lonely), 'student')} ({rule.attribute} {extreme}) "
                     f"{'has' if len(lonely) == 1 else 'have'} too few classmates within "
                     f"{_fmt_num(width)} to fill a team of {smallest}, so their team"
                     f"{'' if len(lonely) == 1 else 's'} will exceed the range. Raise the maximum "
                     "range.", rule_label=label, project=project, students=len(lonely),
                     **detail)
        elif windows > teams:
            self.add(level, "bounds",
                     f"'{label}': the {rule.attribute} values span "
                     f"{_fmt_num(values[-1] - values[0])} and need at least {windows} groups of "
                     f"width {_fmt_num(width)}, but there are only {teams} teams, so at least one "
                     "team will exceed the range. Raise the maximum range.",
                     rule_label=label, project=project, groups_needed=windows, **detail)
        elif not _range_witness(values, sizes, fillers, width):
            self.add("warning" if rule.hard else "info", "bounds",
                     f"'{label}': a simple arrangement by {rule.attribute} could not keep every "
                     f"team within a range of {_fmt_num(width)}; the optimiser may still manage, "
                     "otherwise raise the maximum range.",
                     rule_label=label, project=project, **detail)

    def check_shared(self, project: _Project) -> None:
        sizes = project.sizes
        teams = len(sizes)
        for index, rule in enumerate(self.policy.shared):
            if not rule.enabled:
                continue
            attr = self.attr(rule.attribute)
            if attr is None or attr.kind not in ("categorical", "multi"):
                continue
            label = _rule_label(rule)
            level = "error" if rule.hard else "warning"
            detail = dict(rule_type="shared", rule_index=index)
            k = int(rule.min_shared)
            if attr.kind == "categorical" and k > 1:
                self.add(level, "shared",
                         f"'{label}' asks teammates to share {k} values of {rule.attribute}, but "
                         "each student has only one. Set min_shared to 1.",
                         rule_label=label, project=project, **detail)
                continue
            held: Dict[int, List[int]] = defaultdict(list)
            short = []
            for i in project.pool:
                codes = set(attr.multi[i]) if attr.kind == "multi" else (
                    {attr.codes[i]} if attr.codes[i] >= 0 else set())
                if len(codes) < k:
                    short.append(i)
                for c in codes:
                    held[c].append(i)
            if short:
                ids = [self.cohort.student_ids[i] for i in short]
                self.add(level, "shared",
                         f"'{label}': {_plural(len(short), 'student')} ({', '.join(ids[:5])}"
                         f"{' …' if len(ids) > 5 else ''}) {'has' if len(ids) == 1 else 'have'} "
                         f"fewer than {k} {rule.attribute} value{'s' if k != 1 else ''}, so "
                         f"{'their teams' if len(ids) > 1 else 'their team'} cannot comply. Fill "
                         "in their data or lower min_shared.",
                         rule_label=label, project=project, students=ids, **detail)
            if k != 1 or not sizes:
                continue
            smallest = min(sizes)
            capacity = sum(len(m) // smallest for m in held.values())
            if capacity < teams:
                self.add(level, "shared",
                         f"'{label}': at most {capacity} of {teams} teams can have a "
                         f"{rule.attribute} value that all members share (no value has enough "
                         f"holders), so at least {_plural(teams - capacity, 'team')} will not. "
                         "Relax the rule or make it soft.",
                         rule_label=label, project=project, units=teams - capacity, **detail)

    def check_pairs(self, project: _Project) -> None:
        sizes = project.sizes
        teams = len(sizes)
        in_pool = set(project.pool)
        hard_together: List[Tuple[int, List[int]]] = []
        groups: List[Tuple[int, PairRule, List[int]]] = []
        for index, rule in enumerate(self.policy.pairs):
            if not rule.enabled or len(set(rule.students)) < 2:
                continue
            if rule.projects is not None and project.project_id not in {str(p) for p in rule.projects}:
                continue
            label = _rule_label(rule)
            members, missing, unknown = [], [], []
            for sid in dict.fromkeys(rule.students):
                i = self.cohort.lookup(sid)
                if i is None:
                    unknown.append(str(sid))
                elif i not in in_pool:
                    missing.append(str(sid))
                elif i not in members:
                    members.append(i)
            if unknown:
                self.add("warning", "pairs",
                         f"{_plural(len(unknown), 'student id')} of '{label}' "
                         f"({', '.join(unknown[:5])}{' …' if len(unknown) > 5 else ''}) "
                         f"{'is' if len(unknown) == 1 else 'are'} not in the roster"
                         f"{', so the rule cannot apply' if len(members) + len(missing) < 2 else ''}"
                         ". Check the ids.",
                         rule_label=label, rule_type="pair", rule_index=index, unknown=unknown)
            if missing:
                self.add("info", "pairs",
                         f"{_plural(len(missing), 'student')} of '{label}' "
                         f"({', '.join(missing[:5])}) {'is' if len(missing) == 1 else 'are'} not "
                         "in this project and will be ignored.",
                         rule_label=label, project=project, rule_type="pair", rule_index=index,
                         missing=missing)
            if len(members) < 2:
                continue
            groups.append((index, rule, members))
            level = "error" if rule.hard else "warning"
            if rule.kind == "together":
                if rule.hard:
                    hard_together.append((index, members))
                if len(members) > max(sizes):
                    self.add(level, "pairs",
                             f"'{label}' keeps {len(members)} students together but the largest "
                             f"team has {max(sizes)} places. Split the group or allow larger "
                             "teams.",
                             rule_label=label, project=project, rule_type="pair",
                             rule_index=index, group_size=len(members))
            elif rule.kind == "apart" and len(members) > teams:
                q, r = divmod(len(members), teams)
                units = r * math.comb(q + 1, 2) + (teams - r) * math.comb(q, 2)
                self.add(level, "pairs",
                         f"'{label}' keeps {len(members)} students apart but there are only "
                         f"{teams} teams, so at least {_plural(units, 'pair')} must share a team. "
                         "Shorten the list or form more teams.",
                         rule_label=label, project=project, rule_type="pair", rule_index=index,
                         units=units)

        if len(hard_together) > 1:
            parent = {}

            def find(x: int) -> int:
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for _, members in hard_together:
                for other in members[1:]:
                    parent[find(other)] = find(members[0])
            components: Dict[int, List[int]] = defaultdict(list)
            for _, members in hard_together:
                for m in members:
                    if m not in components[find(m)]:
                        components[find(m)].append(m)
            for comp in components.values():
                if len(comp) > max(sizes) and not any(set(comp) == set(m) for _, m in hard_together):
                    ids = [self.cohort.student_ids[i] for i in comp]
                    self.add("error", "pairs",
                             f"Overlapping keep-together rules chain {len(comp)} students "
                             f"({', '.join(ids[:6])}{' …' if len(ids) > 6 else ''}) into one "
                             f"team, more than the largest team ({max(sizes)}).",
                             project=project, students=ids)

        together = [(i, r, set(m)) for i, r, m in groups if r.kind == "together"]
        apart = [(i, r, set(m)) for i, r, m in groups if r.kind == "apart"]
        for ti, trule, tset in together:
            for ai, arule, aset in apart:
                shared = tset & aset
                if len(shared) >= 2:
                    ids = [self.cohort.student_ids[i] for i in sorted(shared)]
                    both_hard = trule.hard and arule.hard
                    self.add("error" if both_hard else "warning", "pairs",
                             f"'{_rule_label(trule)}' keeps {', '.join(ids[:4])} together but "
                             f"'{_rule_label(arule)}' keeps them apart. Remove them from one of "
                             "the rules.",
                             rule_label=_rule_label(trule), project=project, rule_type="pair",
                             rule_index=ti, conflicts_with=ai, students=ids)

        self._hard_together = hard_together
        self._check_together_composition(project, hard_together)

    def _value_counts_of(self, attr: EncodedAttribute, members: Sequence[int]) -> Counter:
        counts: Counter = Counter()
        for i in members:
            if attr.kind == "multi":
                counts.update(attr.categories[c] for c in set(attr.multi[i]))
            elif attr.kind == "categorical" and attr.codes[i] >= 0:
                counts[attr.categories[attr.codes[i]]] += 1
        return counts

    def _check_together_composition(self, project: _Project,
                                    hard_together: List[Tuple[int, List[int]]]) -> None:
        if not hard_together:
            return
        seats_max = max(project.sizes)
        for index, members in hard_together:
            free = seats_max - len(members)
            if free < 0:
                continue
            ids = [self.cohort.student_ids[i] for i in members]
            group = f"'{_rule_label(self.policy.pairs[index])}' ({', '.join(ids[:5])}" \
                    f"{' …' if len(ids) > 5 else ''})"
            problems: List[str] = []
            for c_index, rule in enumerate(self.policy.counts):
                if not (rule.enabled and rule.hard):
                    continue
                attr = self.attr(rule.attribute)
                if attr is None:
                    continue
                if rule.numeric_range is not None:
                    if attr.kind != "numeric":
                        continue
                    lo, hi = (list(rule.numeric_range) + [None, None])[:2]
                    inside = sum(1 for i in members if not math.isnan(attr.values[i])
                                 and (lo is None or attr.values[i] >= lo)
                                 and (hi is None or attr.values[i] <= hi))
                    counts = {_range_label(rule.attribute, lo, hi): inside}
                elif attr.kind == "numeric":
                    continue
                elif rule.mode == "any_of":
                    counts = {"/".join(rule.values): self.any_of_count(attr, members, rule.values)}
                else:
                    pool_counts = self.pool_counts(attr, project.pool)
                    own = self._value_counts_of(attr, members)
                    values = list(rule.values) or sorted(pool_counts)
                    counts = {v: own.get(v, 0) for v in values}
                for value, k in counts.items():
                    if rule.max_count is not None and k > rule.max_count:
                        problems.append(f"it holds {k} × {rule.attribute}={value} but "
                                        f"'{_rule_label(rule)}' allows at most {rule.max_count}")
                    elif rule.min_count is not None and k + free < rule.min_count:
                        problems.append(f"it holds {k} × {rule.attribute}={value} and its team "
                                        f"has only {free} other seat(s), but "
                                        f"'{_rule_label(rule)}' needs {rule.min_count}")
            for rule in self.policy.isolation:
                if not (rule.enabled and rule.hard):
                    continue
                attr = self.attr(rule.attribute)
                if attr is None or attr.kind not in ("categorical", "multi"):
                    continue
                own = self._value_counts_of(attr, members)
                values = list(rule.values) or sorted(own)
                for value in values:
                    k = own.get(value, 0)
                    if 0 < k < rule.min_together and k + free < rule.min_together:
                        problems.append(f"it holds {k} × {rule.attribute}={value} and its team "
                                        f"has only {free} other seat(s), but "
                                        f"'{_rule_label(rule)}' needs at least "
                                        f"{rule.min_together} together")
            if problems:
                self.add("error", "pairs",
                         f"Hard keep-together group {group} breaks a hard team rule whatever "
                         f"the allocation: {'; '.join(problems[:3])}. Change the group or make "
                         "one of the rules soft.",
                         rule_label=_rule_label(self.policy.pairs[index]), project=project,
                         rule_type="pair", rule_index=index, students=ids)

    def pair_counts(self, prior_projects: Sequence[Any]) -> Dict[Tuple[int, int], int]:
        if self._pair_counts is not None:
            return self._pair_counts
        h = self.policy.history
        counts: Dict[Tuple[int, int], int] = Counter()
        unknown: Set[str] = set()
        for item in prior_projects:
            _, teams = _prior_teams(item)
            for team in teams:
                members = []
                for sid in team:
                    i = self.cohort.lookup(sid)
                    if i is None:
                        unknown.add(sid)
                    else:
                        members.append(i)
                if h.ignore_groups_larger_than is not None and len(team) > h.ignore_groups_larger_than:
                    continue
                members = sorted(set(members))
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        counts[(members[x], members[y])] += 1
        if unknown:
            self.add("info", "history",
                     f"{_plural(len(unknown), 'student')} in earlier projects "
                     f"({', '.join(sorted(unknown)[:5])}{' …' if len(unknown) > 5 else ''}) "
                     f"{'is' if len(unknown) == 1 else 'are'} not in the roster and "
                     "are ignored for repeat checks.", unknown_ids=sorted(unknown))
        self._pair_counts = dict(counts)
        return self._pair_counts

    def check_history(self, project: _Project, prior_projects: Sequence[Any]) -> None:
        h = self.policy.history
        prefs = self.policy.preferences
        pool = project.pool
        n = len(pool)
        position = {i: k for k, i in enumerate(pool)}
        cap_forb = np.zeros((n, n), dtype=bool)
        avoid_forb = np.zeros((n, n), dtype=bool)
        apart_forb = np.zeros((n, n), dtype=bool)
        cap_active = bool(h.enabled and h.max_times_together is not None and prior_projects)
        if cap_active:
            for (i, j), count in self.pair_counts(prior_projects).items():
                if count + 1 > h.max_times_together and i in position and j in position:
                    cap_forb[position[i], position[j]] = cap_forb[position[j], position[i]] = True
        if prefs.use_student_preferences and prefs.avoid_hard:
            for i, listed in self.cohort.avoid.items():
                for j in listed:
                    if i in position and j in position:
                        avoid_forb[position[i], position[j]] = True
                        avoid_forb[position[j], position[i]] = True
        for rule in self.policy.pairs:
            if not (rule.enabled and rule.hard and rule.kind == "apart"):
                continue
            if rule.projects is not None and project.project_id not in {str(p) for p in rule.projects}:
                continue
            members = [position[i] for i in (self.cohort.lookup(s) for s in rule.students)
                       if i is not None and i in position]
            for x in members:
                for y in members:
                    if x != y:
                        apart_forb[x, y] = True

        self._check_together_conflicts(project, position, cap_forb, avoid_forb)

        forb = cap_forb | avoid_forb | apart_forb
        cap_pairs = int(cap_forb.sum()) // 2
        total_forbidden = int(forb.sum()) // 2
        if total_forbidden == 0:
            if cap_active:
                self.add("info", "history",
                         "No two participants have reached the repeat cap yet.", project=project)
            return

        area = "history" if cap_pairs else "pairs"
        other_pairs = total_forbidden - int((cap_forb & ~(avoid_forb | apart_forb)).sum()) // 2
        advice = []
        if cap_pairs:
            advice.append("raise max_times_together, make the cap soft or ignore larger "
                          "earlier groups")
        if other_pairs:
            advice.append("shorten keep-apart lists or relax avoid requests")
        advice_text = "; or ".join(advice)
        all_hard = h.hard_cap or not cap_pairs

        degree = forb.sum(axis=1)
        smallest = min(project.sizes)
        allowed = (n - 1) - degree
        stuck = [k for k in range(n) if allowed[k] < smallest - 1]
        if stuck:
            ids = [self.cohort.student_ids[pool[k]] for k in stuck]
            worst = stuck[int(np.argmin(allowed[stuck]))]
            self.add("error" if all_hard else "warning", area,
                     f"{_plural(len(stuck), 'student')} ({', '.join(ids[:5])}"
                     f"{' …' if len(ids) > 5 else ''}) cannot get a full team: e.g. "
                     f"{self.cohort.student_ids[pool[worst]]} may not share a team with "
                     f"{int(degree[worst])} of the {n - 1} other participants, leaving "
                     f"{int(allowed[worst])} possible teammates for a team of {smallest}. "
                     f"To fix, {advice_text}.",
                     project=project, students=ids)

        density = total_forbidden / math.comb(n, 2) if n > 1 else 0.0
        sources = []
        if cap_pairs:
            sources.append(f"{cap_pairs} already reached the repeat cap")
        if other_pairs:
            sources.append(f"{other_pairs} from avoid requests or keep-apart rules")
        summary = (f"{total_forbidden} of {math.comb(n, 2)} participant pairs ({density:.1%}) may "
                   f"not share a team ({'; '.join(sources)}); on average a student has "
                   f"{degree.mean():.1f} such partners (max {int(degree.max())}).")
        detail = dict(forbidden_pairs=total_forbidden, cap_pairs=cap_pairs, density=density,
                      mean_forbidden_partners=float(degree.mean()),
                      max_forbidden_partners=int(degree.max()))
        dense = density >= 0.25
        if self.probe_time_s <= 0:
            self.add("warning" if dense else "info", area, summary, project=project, **detail)
            return
        probe = probe_forbidden_partition(forb, project.sizes, time_budget_s=self.probe_time_s,
                                          seed=self.seed)
        detail.update(probe_found=probe["found"], probe_best_conflicts=probe["best_conflicts"],
                      probe_attempts=probe["attempts"], probe_elapsed_s=probe["elapsed_s"])
        if probe["found"]:
            self.add("warning" if dense else "info", area,
                     f"{summary} A quick construction found teams with none of these pairs "
                     f"in {probe['elapsed_s']:.2f} s, so the restriction is satisfiable.",
                     project=project, **detail)
        else:
            self.add("warning", area,
                     f"{summary} A quick construction could not avoid all of them (best: "
                     f"{_plural(probe['best_conflicts'], 'pair')} together after "
                     f"{_plural(probe['attempts'], 'attempt')}). The optimiser may still "
                     f"succeed; if it does not, {advice_text}.",
                     project=project, **detail)

    def _check_together_conflicts(self, project: _Project, position: Dict[int, int],
                                  cap_forb: np.ndarray, avoid_forb: np.ndarray) -> None:
        h = self.policy.history
        for index, members in self._hard_together:
            capped, avoided = [], []
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    x, y = position[members[a]], position[members[b]]
                    pair = (self.cohort.student_ids[members[a]], self.cohort.student_ids[members[b]])
                    if cap_forb[x, y]:
                        capped.append(pair)
                    elif avoid_forb[x, y]:
                        avoided.append(pair)
            if not capped and not avoided:
                continue
            label = _rule_label(self.policy.pairs[index])
            reasons = []
            if capped:
                reasons.append(f"{_plural(len(capped), 'pair')} already reached the repeat cap "
                               f"(e.g. {capped[0][0]} & {capped[0][1]})")
            if avoided:
                reasons.append(f"{_plural(len(avoided), 'pair')} asked to avoid each other "
                               f"(e.g. {avoided[0][0]} & {avoided[0][1]})")
            hard = bool(avoided) or h.hard_cap
            self.add("error" if hard else "warning", "pairs",
                     f"'{label}' keeps {len(members)} students together, but "
                     f"{' and '.join(reasons)}. Remove those students from the group or relax "
                     "the conflicting rule.",
                     rule_label=label, project=project, rule_type="pair", rule_index=index,
                     capped_pairs=capped, avoid_pairs=avoided)

    def check_repeat_capacity(self, projects: List[_Project],
                              prior_projects: Sequence[Any]) -> None:
        h = self.policy.history
        if not (h.enabled and h.max_times_together is not None) or not projects:
            return
        cap = int(h.max_times_together)
        level = "error" if h.hard_cap else "warning"
        advice = ("use more, smaller teams, allow pairs to share more teams "
                  "(max_times_together), make the cap soft or ignore larger earlier groups")
        imported: List[Tuple[str, List[List[int]]]] = []
        for item in prior_projects or []:
            pid, teams = _prior_teams(item)
            groups = []
            for team in teams:
                if (h.ignore_groups_larger_than is not None
                        and len(team) > h.ignore_groups_larger_than):
                    continue
                members = sorted({i for i in (self.cohort.lookup(sid) for sid in team)
                                  if i is not None})
                if len(members) >= 2:
                    groups.append(members)
            if groups:
                imported.append((pid, groups))
        if cap == 1:
            for k, project in enumerate(projects):
                pool = set(project.pool)
                T = len(project.sizes)
                worst, source = 0, ""
                for pid, groups in imported:
                    inside = [len([m for m in g if m in pool]) for g in groups]
                    covered = sum(inside)
                    n_groups = sum(1 for c in inside if c) + (len(pool) - covered)
                    forced = max(sum(_collisions(c, T) for c in inside),
                                 sum(_collisions(size, n_groups) for size in project.sizes))
                    if forced > worst:
                        worst, source = forced, f"earlier project {pid}"
                for earlier in projects[:k]:
                    if set(earlier.pool) != pool:
                        continue
                    forced = max(sum(_collisions(e, T) for e in earlier.sizes),
                                 sum(_collisions(size, len(earlier.sizes))
                                     for size in project.sizes))
                    if forced > worst:
                        worst, source = forced, f"project {earlier.project_id}"
                if worst:
                    self.add(level, "history",
                             f"At least {_plural(worst, 'pair')} must meet again: never repeating "
                             f"a teammate from {source} needs each of its teams spread over "
                             f"different teams and each new team drawn from different earlier "
                             f"teams, which {T} teams of up to {max(project.sizes)} cannot do. "
                             f"To fix, {advice}.",
                             project=project, units=worst, source=source)
        generated = list(projects)
        if len(generated) < 2:
            return
        n = self.cohort.n
        prior = np.zeros((n, n), dtype=np.int64)
        for (i, j), count in (self.pair_counts(prior_projects) if prior_projects else {}).items():
            prior[i, j] = prior[j, i] = count
        common = np.zeros((n, n), dtype=np.int64)
        need = np.zeros(n, dtype=np.int64)
        for project in generated:
            member = np.zeros(n, dtype=np.int64)
            member[project.pool] = 1
            common += np.outer(member, member)
            need += member * (min(project.sizes) - 1)
        np.fill_diagonal(common, 0)
        supply = np.minimum(np.maximum(cap - prior, 0), common).sum(axis=1)
        short = np.flatnonzero(need > supply)
        if len(short):
            ids = [self.cohort.student_ids[i] for i in short]
            worst = int(short[int(np.argmax((need - supply)[short]))])
            units = int(math.ceil(int((need - supply)[short].sum()) / 2))
            self.add(level, "history",
                     f"Over the {len(generated)} generated projects, "
                     f"{_plural(len(short), 'student')} ({', '.join(ids[:5])}"
                     f"{' …' if len(ids) > 5 else ''}) need more new teammates than the repeat "
                     f"cap leaves: e.g. {self.cohort.student_ids[worst]} needs {int(need[worst])} "
                     f"teammates but may still meet only {int(supply[worst])} classmates, so at "
                     f"least {_plural(units, 'pair')} must exceed the cap. To fix, {advice}.",
                     units=units, students=ids)

    def check_preferences(self) -> None:
        prefs = self.policy.preferences
        if not prefs.use_student_preferences:
            return
        n_prefer = sum(len(v) for v in self.cohort.prefer.values())
        n_avoid = sum(len(v) for v in self.cohort.avoid.values())
        if n_prefer or n_avoid:
            self.add("info", "preferences",
                     f"{_plural(n_prefer, 'teammate request')} and "
                     f"{_plural(n_avoid, 'avoid request')} from "
                     f"{_plural(len(set(self.cohort.prefer) | set(self.cohort.avoid)), 'student')} "
                     f"will be used (avoid requests are "
                     f"{'hard' if prefs.avoid_hard else 'soft'}).",
                     prefer_requests=n_prefer, avoid_requests=n_avoid)


def _collisions(members: int, groups: int) -> int:
    if members < 2 or groups <= 0:
        return 0
    q, r = divmod(members, groups)
    return r * math.comb(q + 1, 2) + (groups - r) * math.comb(q, 2)


def _range_witness(values: Sequence[float], sizes: Sequence[int], fillers: int,
                   width: float) -> bool:
    teams = len(sizes)
    if not teams:
        return True
    m = len(values)
    base, extra = divmod(m, teams)
    quota = sorted(sizes, reverse=True)
    counts = []
    for t, s in enumerate(quota):
        counts.append(min(s, base + (1 if t < extra else 0)))
    short = m - sum(counts)
    for t in range(teams):
        if short <= 0:
            break
        add = min(short, quota[t] - counts[t])
        counts[t] += add
        short -= add
    if short > 0:
        return False
    pos = 0
    for c in counts:
        if c and values[pos + c - 1] - values[pos] > width + 1e-9:
            return False
        pos += c
    return True


def _group_list(items: Sequence[Dict[str, Any]], limit: int = 3) -> str:
    text = ", ".join(f"{b['group']} {b['students']}" for b in items[:limit])
    return text + (", …" if len(items) > limit else "")


def _count_message(rule: CountRule, breakdown: List[Dict[str, Any]], sizes: Sequence[int],
                   units: int) -> str:
    teams = len(sizes)
    title = rule_headline(rule)
    reasons: List[str] = []
    fixes: List[str] = []
    over = [b for b in breakdown if b["side"] == "max"]
    under = [b for b in breakdown if b["side"] == "min"]
    seats = [b for b in breakdown if b["side"] == "seats"]
    if over:
        if len(over) == 1:
            slots = sum(min(rule.max_count, s) for s in sizes)
            reasons.append(f"{over[0]['students']} students have {over[0]['group']} but "
                           f"{teams} teams × {rule.max_count} leave room for only {slots}")
        else:
            reasons.append(f"{len(over)} values have more students than {teams} teams × "
                           f"{rule.max_count} can hold ({_group_list(over)})")
        fixes.append(f"raise the maximum to {max(b['suggestion'] for b in over)}")
    if under:
        if rule.min_count > min(sizes):
            reasons.append(f"a minimum of {rule.min_count} exceeds the smallest team "
                           f"({min(sizes)} members)")
        elif len(under) == 1:
            need = sum(min(rule.min_count, s) for s in sizes)
            reasons.append(f"only {under[0]['students']} students have {under[0]['group']} but "
                           f"{rule.min_count} in each of {teams} teams needs {need}")
        else:
            reasons.append(f"{len(under)} values have too few students for {rule.min_count} in "
                           f"each of {teams} teams ({_group_list(under)})")
        lowest = min(b["suggestion"] for b in under)
        fixes.append(f"lower the minimum to {lowest}" if lowest > 0
                     else "remove the minimum (or list only values with enough students)")
    if seats:
        reasons.append(f"teams of {min(sizes)} cannot hold {rule.min_count} of each of the "
                       "listed values")
        fixes.append("list fewer values")
    reason = "; ".join(reasons)
    fix = " or ".join(fixes)
    if rule.hard:
        return (f"'{title}' cannot hold in every team: {reason}, so this hard rule cannot be "
                f"satisfied. To fix, {fix} or make the rule soft.")
    cost = _fmt_num(units * rule.weight)
    return (f"'{title}' cannot hold in every team: {reason}, so at least "
            f"{_plural(units, 'violation')} are unavoidable (≈ {cost} penalty points). "
            f"To make it achievable, {fix}.")


def _rule_label(rule: Any) -> str:
    label = getattr(rule, "label", "") or ""
    if label:
        return label
    if isinstance(rule, (CountRule, IsolationRule, PairRule, NumericBoundsRule,
                         SharedValueRule)):
        return rule_headline(rule)
    return f"{getattr(rule, 'attribute', '')} rule"


def _merge_duplicates(findings: List[Finding]) -> List[Finding]:
    merged: Dict[Tuple, Finding] = {}
    order: List[Tuple] = []
    for f in findings:
        pid = f.detail.get("project_id")
        message = f.message
        prefix = f"Project {pid}: "
        if pid is not None and message.startswith(prefix):
            message = message[len(prefix):]
        key = (f.level, f.area, f.rule_label, message, pid is None)
        if pid is not None and key in merged:
            existing = merged[key]
            ids = existing.detail.setdefault("project_ids", [existing.detail.get("project_id")])
            ids.append(pid)
            existing.message = f"Projects {', '.join(map(str, ids))}: {message}"
            continue
        if key in merged:
            continue
        merged[key] = f
        order.append(key)
    return [merged[k] for k in order]


def check_feasibility(frame_or_cohort: Union[pd.DataFrame, EncodedCohort],
                      policy: Union[Policy, Dict[str, Any]],
                      projects: Optional[Sequence[Any]] = None,
                      prior_projects: Sequence[Any] = (),
                      *, default_team_size: int = 5,
                      probe_time_s: float = 0.3,
                      seed: Optional[int] = None) -> FeasibilityReport:
    start = time.perf_counter()
    if not isinstance(policy, Policy):
        policy = Policy.from_dict(policy)
    if isinstance(frame_or_cohort, pd.DataFrame) and "student_id" not in frame_or_cohort.columns:
        report = FeasibilityReport([Finding(
            "error", "pool", "", "The roster has no student_id column; build it with "
            "roster.build_roster_frame().")])
        report.elapsed_s = time.perf_counter() - start
        return report
    checker = _Checker(frame_or_cohort, policy, probe_time_s, seed)
    if checker.cohort.n == 0:
        checker.add("error", "pool", "The roster is empty. Upload students first.")
        checker.report.elapsed_s = time.perf_counter() - start
        return checker.report

    checker.check_policy_structure()
    checker.check_attributes()
    resolved = checker.resolve_projects(list(projects or []), default_team_size)
    if resolved:
        checker.probe_time_s = probe_time_s / len(resolved)
    prior = list(prior_projects or [])
    for project in resolved:
        checker.check_counts(project)
        checker.check_isolation(project)
        checker.check_bounds(project)
        checker.check_shared(project)
        checker.check_pairs(project)
        checker.check_history(project, prior)
    checker.check_repeat_capacity(resolved, prior)
    checker.check_preferences()

    findings = _merge_duplicates(checker.report.findings)
    rank = {level: k for k, level in enumerate(LEVELS)}
    findings.sort(key=lambda f: rank.get(f.level, len(LEVELS)))
    checker.report.findings = findings
    checker.report.elapsed_s = time.perf_counter() - start
    return checker.report


__all__ = [
    "Finding", "FeasibilityReport", "ProjectLike", "PriorProjectLike", "check_feasibility",
    "count_rule_groups", "count_rule_floor", "isolation_floor", "probe_forbidden_partition",
    "ValueBounds", "value_bounds",
]
