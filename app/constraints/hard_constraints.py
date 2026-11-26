from __future__ import annotations
from typing import Set, Dict, TYPE_CHECKING

from .base import Constraint, ConstraintType, ConstraintResult

if TYPE_CHECKING:
    from ..core.models import Team


class AvoidConstraint(Constraint):
    def __init__(
        self, 
        is_hard: bool = True,
        soft_penalty: float = 100.0
    ):
        constraint_type = ConstraintType.HARD if is_hard else ConstraintType.SOFT
        super().__init__("avoid_teammates", constraint_type)
        self.soft_penalty = soft_penalty
    
    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        violations = []
        member_ids = team.member_ids
        
        for student in team.members:
            for avoided_id in student.avoid_teammates:
                if avoided_id in member_ids:
                    violations.append(
                        f"{student.student_id} wants to avoid {avoided_id}"
                    )
        
        if violations:
            return ConstraintResult(
                satisfied=False,
                penalty=len(violations) * self.soft_penalty if self.is_soft() else 0,
                message=f"Avoid violations: {', '.join(violations)}",
                details={"violations": violations},
                units=len(violations),
            )
        
        return ConstraintResult(satisfied=True)


class TeamSizeConstraint(Constraint):
    def __init__(
        self, 
        min_size: int = 3, 
        max_size: int = 8,
        target_size: int = None
    ):
        super().__init__("team_size", ConstraintType.HARD)
        self.min_size = min_size
        self.max_size = max_size
        self.target_size = target_size or (min_size + max_size) // 2
    
    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        size = team.size
        
        if size < self.min_size:
            return ConstraintResult(
                satisfied=False,
                message=f"Team too small: {size} < {self.min_size}",
                details={"size": size, "min": self.min_size},
                units=self.min_size - size,
            )
        
        if size > self.max_size:
            return ConstraintResult(
                satisfied=False,
                message=f"Team too large: {size} > {self.max_size}",
                details={"size": size, "max": self.max_size},
                units=size - self.max_size,
            )

        distance = abs(size - self.target_size)
        bonus = max(0, 10 - distance * 2)
        
        return ConstraintResult(
            satisfied=True,
            bonus=bonus,
            details={"size": size, "target": self.target_size}
        )
    
    def update_bounds(self, min_size: int, max_size: int, target_size: int = None) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.target_size = target_size or (min_size + max_size) // 2


class UniqueAssignmentConstraint(Constraint):
    def __init__(self):
        super().__init__("unique_assignment", ConstraintType.HARD)
    
    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        member_ids = [s.student_id for s in team.members]
        unique_ids = set(member_ids)
        
        if len(member_ids) != len(unique_ids):
            duplicates = [
                sid for sid in unique_ids 
                if member_ids.count(sid) > 1
            ]
            return ConstraintResult(
                satisfied=False,
                message=f"Duplicate students in team: {duplicates}",
                details={"duplicates": duplicates},
                units=len(member_ids) - len(unique_ids),
            )
        
        return ConstraintResult(satisfied=True)


class CrossTeamUniqueConstraint(Constraint):
    def __init__(self):
        super().__init__("cross_team_unique", ConstraintType.HARD)
    
    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        assigned_students: Set[str] = kwargs.get("assigned_students", set())
        
        conflicts = team.member_ids & assigned_students
        
        if conflicts:
            return ConstraintResult(
                satisfied=False,
                message=f"Students already assigned: {conflicts}",
                details={"conflicts": list(conflicts)},
                units=len(conflicts),
            )
        
        return ConstraintResult(satisfied=True)


class MaxRepeatPairingConstraint(Constraint):
    def __init__(self, max_repeats: int = 1, is_hard: bool = True,
                 soft_penalty: float = 50.0):
        constraint_type = ConstraintType.HARD if is_hard else ConstraintType.SOFT
        super().__init__("max_repeat_pairing", constraint_type)
        self.max_repeats = max(1, int(max_repeats))
        self.soft_penalty = soft_penalty

    def evaluate(self, team: "Team", **kwargs) -> ConstraintResult:
        pair_counts: Dict[tuple, int] = kwargs.get("pair_counts") or {}
        if not pair_counts:
            return ConstraintResult(satisfied=True)

        violations = []
        for pair in team.get_pairs():
            sorted_pair = tuple(sorted(pair))
            prior = pair_counts.get(sorted_pair, 0)
            if prior + 1 > self.max_repeats:
                violations.append(
                    f"{sorted_pair[0]}-{sorted_pair[1]} already paired {prior} time"
                    + ("s" if prior != 1 else "")
                )

        if violations:
            return ConstraintResult(
                satisfied=False,
                penalty=len(violations) * self.soft_penalty if self.is_soft() else 0.0,
                message=f"Repeat pairing violations: {len(violations)}",
                details={"violations": violations},
                units=len(violations),
            )

        return ConstraintResult(satisfied=True)

    def update_max_repeats(self, max_repeats: int) -> None:
        self.max_repeats = max(1, int(max_repeats))
