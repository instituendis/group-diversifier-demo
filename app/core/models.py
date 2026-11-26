from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any

from .. import extensions
from .model_base import (ALGORITHMS, CLASSIC_ALGORITHMS, DEFAULT_ALGORITHM,
                         BaseConfiguration, BaseStudent)

Student = extensions.hook("student_class", BaseStudent)
Configuration = extensions.hook("configuration_class", BaseConfiguration)

__all__ = [
    "ALGORITHMS", "CLASSIC_ALGORITHMS", "DEFAULT_ALGORITHM",
    "Student", "Team", "Project", "Configuration", "FormationResult",
]


@dataclass
class Team:
    team_id: str
    members: List[Student] = field(default_factory=list)
    project_id: Optional[str] = None
    _cached_member_ids: Optional[Set[str]] = field(default=None, repr=False, compare=False)
    _cached_pairs: Optional[List[tuple]] = field(default=None, repr=False, compare=False)

    def _invalidate_cache(self) -> None:
        self._cached_member_ids = None
        self._cached_pairs = None

    @property
    def member_ids(self) -> Set[str]:
        if self._cached_member_ids is None:
            self._cached_member_ids = {s.student_id for s in self.members}
        return self._cached_member_ids

    @property
    def size(self) -> int:
        return len(self.members)

    def add_member(self, student: Student) -> None:
        self.members.append(student)
        self._invalidate_cache()

    def remove_member(self, student_id: str) -> Optional[Student]:
        for i, student in enumerate(self.members):
            if student.student_id == student_id:
                self._invalidate_cache()
                return self.members.pop(i)
        return None

    def has_member(self, student_id: str) -> bool:
        return student_id in self.member_ids

    def get_pairs(self) -> List[tuple]:
        if self._cached_pairs is None:
            pairs = []
            ids = list(self.member_ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.append((ids[i], ids[j]))
            self._cached_pairs = pairs
        return self._cached_pairs


@dataclass
class Project:
    project_id: str
    name: str
    teams: List[Team] = field(default_factory=list)
    imported: bool = False

    @property
    def all_student_ids(self) -> Set[str]:
        ids = set()
        for team in self.teams:
            ids.update(team.member_ids)
        return ids

    def get_all_pairs(self) -> Set[tuple]:
        pairs = set()
        for team in self.teams:
            for pair in team.get_pairs():
                pairs.add(tuple(sorted(pair)))
        return pairs


@dataclass
class FormationResult:
    projects: List[Project]
    config: Configuration
    total_score: float
    diversity_scores: Dict[str, float]
    constraint_violations: List[str]
    overlap_matrix: Dict[tuple, int]
    execution_time: float
    algorithm_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "diversity_scores": self.diversity_scores,
            "constraint_violations": self.constraint_violations,
            "execution_time": self.execution_time,
            "algorithm_stats": self.algorithm_stats,
            "num_projects": len(self.projects),
            "num_teams": sum(len(p.teams) for p in self.projects),
        }
