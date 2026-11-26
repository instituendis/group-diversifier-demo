from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .. import extensions

from .models import Student, Team, Project, Configuration

@dataclass
class _Attribute:
    key: str
    kind: str
    getter: Callable[[Student], Any]
    weight: float
    value_range: Optional[float] = None
    declared_categories: int = 0
    team_score: Optional[Callable[[List[Student]], float]] = None


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, (str, list, tuple, set)) and len(value) == 0:
        return False
    return True


class DiversityMetrics:
    def __init__(self, config: Configuration, cohort_students: Optional[List[Student]] = None):
        self.config = config
        self._cohort: Optional[List[Student]] = list(cohort_students) if cohort_students else None
        self._dimension_config: Dict[str, Any] = dict(getattr(config, "dimensions_config", None) or {})
        self._attributes: Dict[str, _Attribute] = self._build_attributes()
        self._cohort_categories: Dict[str, int] = {}
        self._live: Optional[set] = None
        if self._cohort:
            self._live = set()
            for key, attr in self._attributes.items():
                values = [attr.getter(s) for s in self._cohort]
                present = [v for v in values if _present(v)]
                if present and attr.weight > 0:
                    self._live.add(key)
                if attr.kind == "categorical":
                    self._cohort_categories[key] = len({_enum_value(v) for v in present})
                elif attr.kind == "multi":
                    self._cohort_categories[key] = len({
                        item for v in present for item in (v if isinstance(v, (list, tuple)) else [v])
                    })
                elif attr.kind == "ordinal" and attr.value_range is None:
                    numbers = self._numbers(present)
                    if len(numbers) >= 2:
                        attr.value_range = max(numbers) - min(numbers)

    def _build_attributes(self) -> Dict[str, _Attribute]:
        builtins = extensions.hook("metric_attributes", None)
        attrs = builtins(self.config, _Attribute, _enum_value) if builtins is not None else []
        catalogue = {a.key: a for a in attrs}
        builtin_names = frozenset(extensions.hook("builtin_attributes", frozenset()))

        for dim_name, dim in self._dimension_config.items():
            if not isinstance(dim, dict) or not dim.get("enabled", True):
                continue
            if dim_name in builtin_names or dim_name in catalogue:
                continue
            getter = (lambda name: (lambda s: s.get_attribute_value(name)))(dim_name)
            weight = float(dim.get("weight", 50.0) or 0.0)
            dim_type = dim.get("type", "categorical")
            if dim_type == "categorical":
                catalogue[dim_name] = _Attribute(
                    dim_name, "categorical", getter, weight,
                    declared_categories=len(dim.get("categories") or []))
            elif dim_type == "ordinal":
                levels = [lvl.get("weight") for lvl in dim.get("levels") or []
                          if isinstance(lvl, dict) and lvl.get("weight") is not None]
                value_range = (max(levels) - min(levels)) if len(levels) >= 2 else None
                catalogue[dim_name] = _Attribute(dim_name, "ordinal", getter, weight,
                                                 value_range=value_range)
            elif dim_type == "numeric":
                lo, hi = dim.get("min_value"), dim.get("max_value")
                value_range = (hi - lo) if lo is not None and hi is not None and hi > lo else None
                catalogue[dim_name] = _Attribute(dim_name, "ordinal", getter, weight,
                                                 value_range=value_range)
            elif dim_type == "multi_select":
                catalogue[dim_name] = _Attribute(
                    dim_name, "multi", getter, weight,
                    declared_categories=len(dim.get("values") or []))
        return catalogue

    def calculate_team_diversity(self, team: Team) -> Dict[str, float]:
        if team.size < 2:
            return {key: 0.0 for key in self._attributes}
        return {key: self._score(attr, team.members) for key, attr in self._attributes.items()}

    def _score(self, attr: _Attribute, members: List[Student]) -> float:
        values = [attr.getter(s) for s in members]
        present = [v for v in values if _present(v)]
        if attr.kind == "categorical":
            return self._entropy_diversity([_enum_value(v) for v in present],
                                           self._category_count(attr))
        if attr.kind == "ordinal":
            numbers = self._numbers(present)
            if attr.value_range:
                return self._variance_diversity(numbers, attr.value_range)
            return 0.0
        if attr.kind == "multi":
            pooled = [item for v in present for item in (v if isinstance(v, (list, tuple)) else [v])]
            return self._entropy_diversity(pooled, self._category_count(attr))
        if attr.kind == "custom" and attr.team_score is not None:
            return attr.team_score(members)
        if attr.kind == "overlap":
            return self._overlap_ratio([_enum_value(v) for v in present])
        return 0.0

    def _category_count(self, attr: _Attribute) -> Optional[int]:
        count = self._cohort_categories.get(attr.key)
        if count:
            return count
        return attr.declared_categories or None

    def live_attributes(self, team: Optional[Team] = None) -> List[str]:
        if self._live is not None:
            return [k for k in self._attributes if k in self._live]
        if team is None:
            return [k for k, a in self._attributes.items() if a.weight > 0]
        live = []
        for key, attr in self._attributes.items():
            if attr.weight <= 0:
                continue
            if sum(1 for s in team.members if _present(attr.getter(s))) >= 2:
                live.append(key)
        return live

    def attribute_weights(self) -> Dict[str, float]:
        return {key: attr.weight for key, attr in self._attributes.items()}

    def calculate_weighted_score(self, diversity_scores: Dict[str, float],
                                 team: Optional[Team] = None) -> float:
        live = self.live_attributes(team)
        total_weight = sum(self._attributes[k].weight for k in live)
        if total_weight <= 0:
            return 0.0
        return sum(diversity_scores.get(k, 0.0) * self._attributes[k].weight
                   for k in live) / total_weight

    def calculate_project_diversity(self, project: Project) -> Dict[str, Any]:
        team_scores = []
        team_weighted = []
        for team in project.teams:
            scores = self.calculate_team_diversity(team)
            team_scores.append(scores)
            team_weighted.append(self.calculate_weighted_score(scores, team))

        if not team_scores:
            return {"teams": [], "average": 0.0, "variance": 0.0}

        avg_scores = {}
        for attr in team_scores[0].keys():
            values = [ts.get(attr, 0.0) for ts in team_scores]
            avg_scores[attr] = sum(values) / len(values)

        return {
            "teams": team_scores,
            "team_weighted_scores": team_weighted,
            "average_scores": avg_scores,
            "average_weighted": sum(team_weighted) / len(team_weighted),
            "variance": self._calculate_variance(team_weighted),
            "min_score": min(team_weighted),
            "max_score": max(team_weighted),
            "live_attributes": self.live_attributes(),
        }

    def calculate_overlap_penalty(
        self,
        projects: List[Project],
        max_allowed: int = 1
    ) -> Dict[str, Any]:
        pair_counts: Dict[tuple, int] = {}
        for project in projects:
            for pair in project.get_all_pairs():
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        violations = {pair: count for pair, count in pair_counts.items() if count > max_allowed}
        total_penalty = sum(
            (count - max_allowed) * self.config.overlap_penalty
            for count in violations.values()
        )
        return {
            "pair_counts": pair_counts,
            "violations": violations,
            "total_penalty": total_penalty,
            "num_violations": len(violations),
        }

    def calculate_preference_score(self, team: Team) -> Dict[str, float]:
        prefer_bonus = 0.0
        avoid_penalty = 0.0
        member_ids = team.member_ids
        for student in team.members:
            for preferred_id in student.prefer_teammates:
                if preferred_id in member_ids:
                    prefer_bonus += self.config.prefer_bonus
            if not self.config.avoid_is_hard_constraint:
                for avoided_id in student.avoid_teammates:
                    if avoided_id in member_ids:
                        avoid_penalty += self.config.avoid_soft_penalty
        return {
            "prefer_bonus": prefer_bonus,
            "avoid_penalty": avoid_penalty,
            "net_score": prefer_bonus - avoid_penalty,
        }

    def _entropy_diversity(self, values: List[Any], n_categories: Optional[int] = None) -> float:
        if not values:
            return 0.0
        n = len(values)
        counts = Counter(values)
        if len(counts) <= 1:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / n
            entropy -= p * math.log2(p)
        k = max(len(counts), int(n_categories)) if n_categories else n
        max_entropy = math.log2(min(n, k))
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def _overlap_ratio(values: List[Any]) -> float:
        if len(values) < 2:
            return 0.0
        return Counter(values).most_common(1)[0][1] / len(values)

    @staticmethod
    def _attribute_shares(students: Optional[List[Student]], attribute: str) -> Optional[Dict[str, float]]:
        if not students:
            return None
        counts: Dict[str, int] = {}
        total = 0
        for s in students:
            v = s.get_attribute_value(attribute)
            if v is None:
                continue
            v = str(_enum_value(v))
            counts[v] = counts.get(v, 0) + 1
            total += 1
        if not total:
            return None
        return {c: n / total for c, n in counts.items()}

    @staticmethod
    def _numbers(values: List[Any]) -> List[float]:
        numbers = []
        for v in values:
            try:
                x = float(_enum_value(v))
            except (TypeError, ValueError):
                continue
            if not math.isnan(x):
                numbers.append(x)
        return numbers

    def _variance_diversity(self, values: List[float], max_val: float) -> float:
        if len(values) < 2 or not max_val or max_val <= 0:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return min(variance / (max_val / 2) ** 2, 1.0)

    def _calculate_variance(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _get_attribute_names(self) -> List[str]:
        return list(self._attributes)

    def get_fairness_metrics(self, projects: List[Project]) -> Dict[str, Any]:
        student_scores: Dict[str, List[float]] = {}
        for project in projects:
            for team in project.teams:
                team_score = self.calculate_weighted_score(self.calculate_team_diversity(team), team)
                for student in team.members:
                    student_scores.setdefault(student.student_id, []).append(team_score)

        student_stats = {}
        for sid, scores in student_scores.items():
            student_stats[sid] = {
                "average": sum(scores) / len(scores),
                "variance": self._calculate_variance(scores),
                "min": min(scores),
                "max": max(scores),
            }

        averages = [s["average"] for s in student_stats.values()]
        if not averages:
            return {"student_stats": {}, "overall_variance": 0.0, "min_average": 0,
                    "max_average": 0, "fairness_score": 1.0, "gini": 0.0,
                    "min_max_ratio": 1.0}
        lo, hi = min(averages), max(averages)
        mean = sum(averages) / len(averages)
        fairness = 1.0 - (hi - lo) / mean if mean > 0 else (1.0 if hi == lo else 0.0)
        return {
            "student_stats": student_stats,
            "overall_variance": self._calculate_variance(averages),
            "min_average": lo,
            "max_average": hi,
            "fairness_score": max(0.0, min(1.0, fairness)),
            "gini": self._gini(averages),
            "min_max_ratio": (lo / hi) if hi > 0 else 1.0,
        }

    @staticmethod
    def _gini(values: List[float]) -> float:
        n = len(values)
        total = sum(values)
        if n < 2 or total <= 0:
            return 0.0
        ordered = sorted(values)
        weighted = sum((i + 1) * v for i, v in enumerate(ordered))
        return max(0.0, (2 * weighted) / (n * total) - (n + 1) / n)
