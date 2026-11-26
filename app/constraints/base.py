from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Iterable, Sequence, Tuple, TYPE_CHECKING

from app import extensions
from app.core.ids import pair_key as _ids_pair_key

if TYPE_CHECKING:
    from ..core.models import Team, Project

PairCounts = Dict[Tuple[str, str], int]


class ConstraintType(Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class ConstraintResult:
    satisfied: bool
    penalty: float = 0.0
    bonus: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    units: float = 0.0

    @property
    def net_score(self) -> float:
        return self.bonus - self.penalty

    @property
    def violation_units(self) -> float:
        if self.satisfied:
            return 0.0
        return self.units if self.units > 0 else 1.0


class Constraint(ABC):
    def __init__(
        self, 
        name: str, 
        constraint_type: ConstraintType,
        weight: float = 1.0,
        enabled: bool = True
    ):
        self.name = name
        self.constraint_type = constraint_type
        self.weight = weight
        self.enabled = enabled
    
    @abstractmethod
    def evaluate(self, team: Team, **kwargs) -> ConstraintResult:
        pass
    
    def is_hard(self) -> bool:
        return self.constraint_type == ConstraintType.HARD
    
    def is_soft(self) -> bool:
        return self.constraint_type == ConstraintType.SOFT
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.constraint_type.value})"


class CompositeConstraint(Constraint):
    def __init__(
        self, 
        name: str,
        constraints: List[Constraint] = None,
        require_all: bool = True
    ):
        has_hard = any(c.is_hard() for c in (constraints or []))
        constraint_type = ConstraintType.HARD if has_hard else ConstraintType.SOFT
        
        super().__init__(name, constraint_type)
        self.constraints = constraints or []
        self.require_all = require_all
    
    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)
        if constraint.is_hard():
            self.constraint_type = ConstraintType.HARD
    
    def remove_constraint(self, name: str) -> Optional[Constraint]:
        for i, c in enumerate(self.constraints):
            if c.name == name:
                return self.constraints.pop(i)
        return None
    
    def evaluate(self, team: Team, **kwargs) -> ConstraintResult:
        results = []
        for constraint in self.constraints:
            if constraint.enabled:
                results.append(constraint.evaluate(team, **kwargs))
        
        if not results:
            return ConstraintResult(satisfied=True)
        
        if self.require_all:
            satisfied = all(r.satisfied for r in results)
        else:
            satisfied = any(r.satisfied for r in results)
        
        total_penalty = sum(r.penalty * c.weight for r, c in zip(results, self.constraints) if c.enabled)
        total_bonus = sum(r.bonus * c.weight for r, c in zip(results, self.constraints) if c.enabled)
        
        messages = [r.message for r in results if r.message]
        
        return ConstraintResult(
            satisfied=satisfied,
            penalty=total_penalty,
            bonus=total_bonus,
            message="; ".join(messages),
            details={"sub_results": results}
        )


class ConstraintEvaluator:
    _EXCLUDED_HARD: frozenset = frozenset()

    def __init__(self, constraints: List[Constraint] = None):
        self.constraints = constraints or []

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def get_constraint(self, name: str) -> Optional[Constraint]:
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None

    def get_hard_constraints(self) -> List[Constraint]:
        return [
            c for c in self.constraints
            if c.is_hard() and c.enabled and c.name not in self._EXCLUDED_HARD
        ]

    def get_soft_constraints(self) -> List[Constraint]:
        return [c for c in self.constraints if c.is_soft() and c.enabled]

    def evaluate_team(self, team: Team, **kwargs) -> Dict[str, ConstraintResult]:
        results = {}
        for constraint in self.constraints:
            if constraint.enabled:
                results[constraint.name] = constraint.evaluate(team, **kwargs)
        return results

    def check_hard_constraints(self, team: Team, **kwargs) -> bool:
        for constraint in self.get_hard_constraints():
            if not constraint.evaluate(team, **kwargs).satisfied:
                return False
        return True

    def hard_violation_count(self, team: Team, **kwargs) -> int:
        return sum(
            1 for constraint in self.get_hard_constraints()
            if not constraint.evaluate(team, **kwargs).satisfied
        )

    def hard_violation_units(self, team: Team, **kwargs) -> float:
        return sum(
            constraint.evaluate(team, **kwargs).violation_units
            for constraint in self.get_hard_constraints()
        )

    def _accumulate(self, team: Team, **kwargs) -> Tuple[float, float]:
        units = 0.0
        soft = 0.0
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
            result = constraint.evaluate(team, **kwargs)
            if (constraint.is_soft() or result.satisfied
                    or constraint.name in self._EXCLUDED_HARD):
                soft += result.net_score * constraint.weight
            else:
                units += result.violation_units
        return units, soft

    def calculate_soft_score(self, team: Team, **kwargs) -> float:
        return self._accumulate(team, **kwargs)[1]

    def team_key(self, team: Team, **kwargs) -> Tuple[float, float]:
        units, soft = self._accumulate(team, **kwargs)
        return units, -soft

    def get_violations(self, team: Team, pair_counts: Optional[PairCounts] = None,
                       **kwargs) -> List[str]:
        kwargs["pair_counts"] = pair_counts or {}
        violations = []
        for constraint in self.constraints:
            if constraint.enabled:
                result = constraint.evaluate(team, **kwargs)
                if not result.satisfied and result.message:
                    text = f"[{constraint.name}] {result.message}"
                    details = result.details.get("violations", []) if result.details else []
                    if details:
                        shown = "; ".join(str(d) for d in details[:3])
                        extra = len(details) - 3
                        text += f" — {shown}" + (f" (+{extra} more)" if extra > 0 else "")
                    violations.append(text)
        return violations

    def violations_by_project(
        self,
        projects: Sequence["Project"],
        base_pair_counts: Optional[PairCounts] = None,
        skip_imported: bool = True,
    ) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        counts: PairCounts = dict(base_pair_counts or {})
        for project in projects:
            if not (skip_imported and getattr(project, "imported", False)):
                messages: List[str] = []
                for team in project.teams:
                    messages.extend(self.get_violations(team, pair_counts=counts))
                result[str(project.project_id)] = messages
            add_team_pairs(counts, project.teams)
        return result


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return _ids_pair_key(a, b)


def add_team_pairs(counts: PairCounts, teams: Iterable["Team"]) -> PairCounts:
    for team in teams:
        ids = sorted(team.member_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = (ids[i], ids[j])
                counts[key] = counts.get(key, 0) + 1
    return counts


def prior_pair_counts(
    projects: Sequence["Project"],
    upto: int,
    base_pair_counts: Optional[PairCounts] = None,
) -> PairCounts:
    counts: PairCounts = dict(base_pair_counts or {})
    for project in projects[:upto]:
        add_team_pairs(counts, project.teams)
    return counts


class ConstructionEvaluator(ConstraintEvaluator):
    _EXCLUDED_HARD = frozenset({"team_size"}) | frozenset(
        extensions.hook("excluded_hard_constraints", ()))

    def __init__(self, evaluator: "ConstraintEvaluator"):
        super().__init__(list(evaluator.constraints))
