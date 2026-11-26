from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from app.core import ids
from app.core.models import Project, Student, Team

from .history import PriorProject
from .runner import EngineResult


def _student_index(students: Sequence[Student]) -> Tuple[ids.IdMatcher, Dict[str, Student]]:
    by_id: Dict[str, Student] = {}
    for s in students:
        by_id.setdefault(str(s.student_id).strip(), s)
    return ids.IdMatcher(by_id), by_id


def _find(sid: str, matcher: ids.IdMatcher, by_id: Dict[str, Student]) -> Optional[Student]:
    resolved = matcher.resolve(sid)
    return None if resolved is None else by_id.get(resolved)


def to_legacy_projects(result: EngineResult, students: Sequence[Student]) -> List[Project]:
    matcher, by_id = _student_index(students)
    projects: List[Project] = []
    for pr in result.projects:
        project_id = str(pr.project_id)
        teams = []
        for t, member_ids in enumerate(pr.teams):
            members = [s for s in (_find(sid, matcher, by_id) for sid in member_ids)
                       if s is not None]
            teams.append(Team(team_id=f"team_{t + 1}", members=members, project_id=project_id))
        projects.append(Project(project_id=project_id, name=f"Project {project_id}", teams=teams))
    return projects


def teams_from_legacy(project: Project) -> List[List[str]]:
    return [[str(s.student_id) for s in team.members] for team in project.teams]


def prior_from_legacy(projects: Sequence[Project]) -> List[PriorProject]:
    return [PriorProject(project_id=str(p.project_id), teams=teams_from_legacy(p))
            for p in projects]


def pair_counts(projects: Sequence[Project]) -> Dict[Tuple[str, str], int]:
    return ids.pair_counts(
        [str(s.student_id) for s in team.members]
        for project in projects
        for team in project.teams
    )
