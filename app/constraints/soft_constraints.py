from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING

from .base import Constraint, ConstraintType, ConstraintResult

if TYPE_CHECKING:
    from ..core.models import Team, Student


ValueGetter = Callable[["Student"], Any]


def _plain(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, list):
        return tuple(value)
    return value


class AttributeGetter:
    def __init__(self, attribute: str):
        self.attribute = attribute

    def __call__(self, student: "Student") -> Any:
        try:
            return student.get_attribute_value(self.attribute)
        except AttributeError:
            return None


class FieldGetter:
    def __init__(self, field: str):
        self.field = field

    def __call__(self, student: "Student") -> Any:
        return getattr(student, self.field, None)


def _attribute_getter(attribute: str) -> ValueGetter:
    return AttributeGetter(attribute)


def max_gini_simpson(n: int, n_categories: Optional[int] = None) -> float:
    if n < 2:
        return 0.0
    m = n if not n_categories else min(n, int(n_categories))
    if m <= 1:
        return 0.0
    q, r = divmod(n, m)
    sum_sq = r * (q + 1) ** 2 + (m - r) * q ** 2
    return 1.0 - sum_sq / (n * n)


def normalized_gini_simpson(counts: Iterable[int], n_categories: Optional[int] = None) -> float:
    counts = [c for c in counts if c > 0]
    n = sum(counts)
    if n < 2 or len(counts) < 2:
        return 0.0
    k = max(len(counts), int(n_categories)) if n_categories else None
    best = max_gini_simpson(n, k)
    if best <= 0:
        return 0.0
    gs = 1.0 - sum(c * c for c in counts) / (n * n)
    return min(1.0, gs / best)


class PreferConstraint(Constraint):
    def __init__(self, bonus_per_match: float = 10.0):
        super().__init__("prefer_teammates", ConstraintType.SOFT)
        self.bonus_per_match = bonus_per_match

    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        matches = []
        member_ids = team.member_ids

        for student in team.members:
            for preferred_id in student.prefer_teammates:
                if preferred_id in member_ids:
                    matches.append(f"{student.student_id} with {preferred_id}")

        bonus = len(matches) * self.bonus_per_match

        return ConstraintResult(
            satisfied=True,
            bonus=bonus,
            message=f"Preference matches: {len(matches)}" if matches else "",
            details={"matches": matches}
        )


class DiversityConstraint(Constraint):
    def __init__(
        self,
        attribute: str,
        weight: float = 1.0,
        diversity_bonus: float = 50.0,
        n_categories: Optional[int] = None,
        name: Optional[str] = None,
        value_getter: Optional[ValueGetter] = None,
    ):
        super().__init__(name or f"diversity_{attribute}", ConstraintType.SOFT, weight=weight)
        self.attribute = attribute
        self.diversity_bonus = diversity_bonus
        self.n_categories = n_categories
        self._get = value_getter or _attribute_getter(attribute)

    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        values = []
        for student in team.members:
            val = self._get(student)
            if val is None or val == "" or val == []:
                continue
            values.append(_plain(val))

        if len(values) < 2:
            return ConstraintResult(satisfied=True, bonus=0)

        counts = Counter(values)
        evenness = normalized_gini_simpson(counts.values(), self.n_categories)
        bonus = evenness * self.diversity_bonus

        return ConstraintResult(
            satisfied=True,
            bonus=bonus,
            message=f"{self.attribute} diversity: {evenness:.2f}",
            details={
                "attribute": self.attribute,
                "unique_count": len(counts),
                "total_count": len(values),
                "evenness": evenness,
                "diversity_ratio": evenness,
            }
        )


class MultiSelectDiversityConstraint(Constraint):
    def __init__(
        self,
        attribute: str,
        weight: float = 1.0,
        diversity_bonus: float = 50.0,
        name: Optional[str] = None,
        value_getter: Optional[ValueGetter] = None,
    ):
        super().__init__(name or f"diversity_{attribute}", ConstraintType.SOFT, weight=weight)
        self.attribute = attribute
        self.diversity_bonus = diversity_bonus
        self._get = value_getter or _attribute_getter(attribute)

    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        selections: List[Any] = []
        members_with_data = 0
        for student in team.members:
            raw = self._get(student)
            if not raw:
                continue
            items = raw if isinstance(raw, (list, tuple, set)) else [raw]
            items = [_plain(v) for v in items if v not in (None, "")]
            if items:
                members_with_data += 1
                selections.extend(items)
        if members_with_data < 2:
            return ConstraintResult(satisfied=True, bonus=0)
        ratio = len(set(selections)) / len(selections)
        return ConstraintResult(
            satisfied=True,
            bonus=ratio * self.diversity_bonus,
            message=f"{self.attribute} variety: {ratio:.2f}",
            details={"attribute": self.attribute, "unique_ratio": ratio,
                     "selections": len(selections)},
        )


class NumericSpreadConstraint(Constraint):
    def __init__(
        self,
        attribute: str,
        pool_mean: float,
        pool_sd: float,
        weight: float = 1.0,
        diversity_bonus: float = 50.0,
        name: Optional[str] = None,
        value_getter: Optional[ValueGetter] = None,
    ):
        super().__init__(name or f"spread_{attribute}", ConstraintType.SOFT, weight=weight)
        self.attribute = attribute
        self.pool_mean = float(pool_mean)
        self.pool_sd = float(pool_sd)
        self.diversity_bonus = diversity_bonus
        self._get = value_getter or _attribute_getter(attribute)

    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        values: List[float] = []
        for student in team.members:
            raw = self._get(student)
            raw = _plain(raw)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isnan(value):
                continue
            values.append(value)
        if len(values) < 2 or self.pool_sd <= 0:
            return ConstraintResult(satisfied=True, bonus=0)
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        spread = min(1.0, sd / self.pool_sd)
        centred = 1.0 - min(1.0, abs(mean - self.pool_mean) / self.pool_sd)
        score = 0.5 * spread + 0.5 * centred
        return ConstraintResult(
            satisfied=True,
            bonus=score * self.diversity_bonus,
            message=f"{self.attribute} spread: {score:.2f}",
            details={"attribute": self.attribute, "team_mean": mean, "team_sd": sd,
                     "spread": spread, "centred": centred},
        )


class ProportionalityConstraint(Constraint):
    def __init__(
        self,
        attribute: str,
        cohort_distribution: Dict[str, float],
        weight: float = 1.0,
        max_bonus: float = 50.0,
    ):
        super().__init__(f"proportionality_{attribute}", ConstraintType.SOFT, weight=weight)
        self.attribute = attribute
        self.cohort_distribution = cohort_distribution
        self.max_bonus = max_bonus

    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        values = []
        for student in team.members:
            try:
                val = student.get_attribute_value(self.attribute)
            except AttributeError:
                continue
            if val is None:
                continue
            values.append(str(_plain(val)))

        if len(values) < 2:
            return ConstraintResult(satisfied=True, bonus=0)

        target = self._target_distribution(kwargs.get("remaining_students"))
        team_dist = {c: values.count(c) / len(values) for c in set(values)}
        tv_distance = 0.5 * sum(
            abs(team_dist.get(c, 0.0) - target.get(c, 0.0))
            for c in set(team_dist) | set(target)
        )
        bonus = (1.0 - tv_distance) * self.max_bonus

        return ConstraintResult(
            satisfied=True,
            bonus=bonus,
            message=f"{self.attribute} proportionality: {1.0 - tv_distance:.2f}",
            details={
                "attribute": self.attribute,
                "team_distribution": team_dist,
                "cohort_distribution": self.cohort_distribution,
                "total_variation": tv_distance,
            }
        )

    def _target_distribution(self, remaining) -> Dict[str, float]:
        if remaining is None:
            return self.cohort_distribution
        counts: Dict[str, int] = {}
        total = 0
        for s in remaining:
            v = s.get_attribute_value(self.attribute)
            if v is None:
                continue
            v = str(_plain(v))
            counts[v] = counts.get(v, 0) + 1
            total += 1
        if total == 0:
            return self.cohort_distribution
        return {c: n / total for c, n in counts.items()}


class OverlapPenaltyConstraint(Constraint):
    def __init__(self, penalty_per_repeat: float = 50.0, max_allowed: Optional[int] = None):
        super().__init__("overlap_penalty", ConstraintType.SOFT)
        self.penalty_per_repeat = penalty_per_repeat
        self.max_allowed = max_allowed

    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        pair_counts: Dict[tuple, int] = kwargs.get("pair_counts") or {}
        if not pair_counts:
            return ConstraintResult(satisfied=True)

        total_repeats = 0
        repeats = []
        for pair in team.get_pairs():
            sorted_pair = tuple(sorted(pair))
            count = pair_counts.get(sorted_pair, 0)
            if count >= 1:
                total_repeats += count
                repeats.append(f"{sorted_pair[0]}-{sorted_pair[1]}: {count} time"
                               + ("s" if count != 1 else "") + " before")

        return ConstraintResult(
            satisfied=True,
            penalty=total_repeats * self.penalty_per_repeat,
            message=f"Repeated pairs: {len(repeats)}" if repeats else "",
            details={"repeats": repeats, "total_repeats": total_repeats,
                     "violations": repeats, "total_excess": total_repeats},
        )
