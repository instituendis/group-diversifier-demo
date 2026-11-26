from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .. import extensions

from .base import Constraint, ConstraintEvaluator
from .hard_constraints import (
    AvoidConstraint,
    TeamSizeConstraint,
    UniqueAssignmentConstraint,
    MaxRepeatPairingConstraint,
)
from .soft_constraints import (
    PreferConstraint,
    DiversityConstraint,
    MultiSelectDiversityConstraint,
    NumericSpreadConstraint,
    ProportionalityConstraint,
    OverlapPenaltyConstraint,
    FieldGetter,
)

if TYPE_CHECKING:
    from ..core.models import Configuration, Student


def _typed_field(name: str) -> Callable[["Student"], Any]:
    return FieldGetter(name)


class ConstraintFactory:
    BUILTIN_ATTRIBUTES = frozenset(extensions.hook("builtin_attributes", frozenset()))

    @classmethod
    def create_from_config(cls, config: "Configuration", students: List["Student"] = None) -> ConstraintEvaluator:
        hard = cls._create_hard_constraints(config)
        soft = cls._create_soft_constraints(config, students=students)
        return ConstraintEvaluator(hard + soft)

    @classmethod
    def _create_hard_constraints(cls, config: "Configuration") -> List[Constraint]:
        constraints: List[Constraint] = [
            TeamSizeConstraint(
                min_size=config.team_size_min,
                max_size=config.team_size_max,
                target_size=config.team_size_target,
            ),
            UniqueAssignmentConstraint(),
            AvoidConstraint(
                is_hard=config.avoid_is_hard_constraint,
                soft_penalty=config.avoid_soft_penalty,
            ),
        ]
        if config.num_projects > 1:
            constraints.append(MaxRepeatPairingConstraint(
                max_repeats=max(1, int(config.max_repeat_pairings or 1)),
                is_hard=bool(getattr(config, "max_repeat_is_hard", True)),
                soft_penalty=config.overlap_penalty,
            ))
        return constraints

    @classmethod
    def _create_soft_constraints(cls, config: "Configuration", students: List["Student"] = None) -> List[Constraint]:
        constraints: List[Constraint] = []
        pool = list(students or [])

        def has_data(getter: Callable[["Student"], Any]) -> bool:
            if not pool:
                return True
            return any(cls._present(getter(s)) for s in pool)

        def n_categories(getter: Callable[["Student"], Any]) -> Optional[int]:
            if not pool:
                return None
            values = {cls._plain(getter(s)) for s in pool if cls._present(getter(s))}
            return len(values) or None

        extension_terms = extensions.hook("classic_soft_terms", None)
        context = SimpleNamespace(students=students, has_data=has_data,
                                  n_categories=n_categories, cohort_shares=cls._cohort_shares)

        def slot(name: str) -> None:
            if extension_terms is not None:
                constraints.extend(extension_terms(config, context, name))

        constraints.append(PreferConstraint(bonus_per_match=config.prefer_bonus))
        slot("head")

        if config.num_projects > 1:
            constraints.append(OverlapPenaltyConstraint(
                penalty_per_repeat=config.overlap_penalty,
            ))
        slot("after_overlap")

        constraints.extend(cls._dimension_constraints(config, pool))

        if pool:
            cls._add_proportionality(config, pool, constraints)

        slot("end")

        return constraints

    @classmethod
    def _dimension_constraints(cls, config: "Configuration", pool: List["Student"]) -> List[Constraint]:
        constraints: List[Constraint] = []
        for dim_name, dim_config in (getattr(config, "dimensions_config", None) or {}).items():
            if dim_name in cls.BUILTIN_ATTRIBUTES or not isinstance(dim_config, dict):
                continue
            if not dim_config.get("enabled", True):
                continue
            weight = float(dim_config.get("weight", 50.0) or 0.0)
            if weight <= 0:
                continue
            dim_type = dim_config.get("type")
            values = [s.get_attribute_value(dim_name) for s in pool]
            present = [v for v in values if cls._present(v)]
            if pool and not present:
                continue

            if dim_type == "categorical":
                n_cat = len({cls._plain(v) for v in present}) if pool else (
                    len(dim_config.get("categories") or []) or None)
                constraints.append(DiversityConstraint(
                    attribute=dim_name, weight=weight / 100,
                    diversity_bonus=50.0, n_categories=n_cat,
                ))
            elif dim_type in ("ordinal", "numeric"):
                stats = cls._numeric_stats(present) if pool else None
                if stats is None:
                    stats = cls._declared_numeric_stats(dim_config)
                if stats is None or stats[1] <= 0:
                    continue
                constraints.append(NumericSpreadConstraint(
                    attribute=dim_name, pool_mean=stats[0], pool_sd=stats[1],
                    weight=weight / 100, diversity_bonus=50.0,
                ))
            elif dim_type == "multi_select":
                constraints.append(MultiSelectDiversityConstraint(
                    attribute=dim_name, weight=weight / 100, diversity_bonus=50.0,
                ))
        return constraints

    @classmethod
    def _add_proportionality(cls, config: "Configuration", pool: List["Student"],
                             constraints: List[Constraint]) -> None:
        existing = {
            c.attribute: c for c in constraints
            if isinstance(c, ProportionalityConstraint)
        }
        dims = getattr(config, "dimensions_config", None) or {}
        for attr, p_cfg in (getattr(config, "proportionality_config", None) or {}).items():
            if not isinstance(p_cfg, dict) or not p_cfg.get("enabled", False):
                continue
            dim_type = (dims.get(attr) or {}).get("type") if isinstance(dims.get(attr), dict) else None
            if dim_type and dim_type != "categorical":
                continue
            p_weight = float(p_cfg.get("weight", 50.0) or 0.0)
            if p_weight <= 0:
                continue
            if attr in existing:
                existing[attr].weight += p_weight / 100.0
                continue
            shares = cls._cohort_shares(pool, attr)
            if shares and len(shares) > 1:
                constraint = ProportionalityConstraint(
                    attribute=attr,
                    cohort_distribution=shares,
                    weight=p_weight / 100.0,
                    max_bonus=50.0,
                )
                constraints.append(constraint)
                existing[attr] = constraint

    @staticmethod
    def _plain(value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, list):
            return tuple(value)
        return value

    @staticmethod
    def _present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        if isinstance(value, (str, list, tuple, set, dict)) and len(value) == 0:
            return False
        return True

    @classmethod
    def _numeric_stats(cls, values: List[Any]):
        numbers = []
        for v in values:
            try:
                x = float(cls._plain(v))
            except (TypeError, ValueError):
                continue
            if not math.isnan(x):
                numbers.append(x)
        if len(numbers) < 2:
            return None
        mean = sum(numbers) / len(numbers)
        sd = math.sqrt(sum((x - mean) ** 2 for x in numbers) / len(numbers))
        return mean, sd

    @staticmethod
    def _declared_numeric_stats(dim_config: Dict[str, Any]):
        levels = [lvl.get("weight") for lvl in dim_config.get("levels") or []
                  if isinstance(lvl, dict) and lvl.get("weight") is not None]
        if len(levels) >= 2:
            mean = sum(levels) / len(levels)
            sd = math.sqrt(sum((x - mean) ** 2 for x in levels) / len(levels))
            return mean, sd
        lo, hi = dim_config.get("min_value"), dim_config.get("max_value")
        if lo is not None and hi is not None and hi > lo:
            return (lo + hi) / 2.0, (hi - lo) / math.sqrt(12.0)
        return None

    @staticmethod
    def _cohort_shares(students: List["Student"], attribute: str) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        total = 0
        for s in students or []:
            v = s.get_attribute_value(attribute)
            if v is None or v == "":
                continue
            if hasattr(v, "value"):
                v = v.value
            v = str(v)
            counts[v] = counts.get(v, 0) + 1
            total += 1
        if not total:
            return {}
        return {c: n / total for c, n in counts.items()}

    @classmethod
    def create_minimal(cls) -> ConstraintEvaluator:
        return ConstraintEvaluator([
            TeamSizeConstraint(min_size=2, max_size=10),
            UniqueAssignmentConstraint(),
        ])

    @classmethod
    def create_for_validation(cls, config: "Configuration") -> ConstraintEvaluator:
        return ConstraintEvaluator(cls._create_hard_constraints(config))
