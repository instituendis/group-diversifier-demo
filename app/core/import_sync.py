from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Set

from .assignment_import import build_imported_projects
from .engine_session.plain_form import projects_to_ids
from .ids import IdMatcher, canonical_id
from .models import Project, Student, Team


def is_imported(project: Any) -> bool:
    if isinstance(project, Project):
        return bool(project.imported)
    return bool(project.get("imported")) if isinstance(project, dict) else False


def team_sets(entry: Dict[str, Any], keep: Optional[set] = None) -> Set[frozenset]:
    teams = (frozenset(cid for cid in map(canonical_id, team) if keep is None or cid in keep)
             for team in entry.get("teams", []))
    return {team for team in teams if team}


def refreshed_imports(projects: Sequence[Dict[str, Any]], imported_assignments: Any,
                      students: Sequence[Student]) -> Dict[str, Dict[str, Any]]:
    if imported_assignments is None:
        return {}
    by_id = {s.student_id: s for s in students}
    try:
        fresh = {p["project_id"]: p for p in _imported_plain(imported_assignments, by_id)}
    except KeyError:
        return {}
    current = {str(p["project_id"]): p for p in projects if p.get("imported")}
    if not current or set(fresh) != set(current):
        return {}
    roster = {canonical_id(sid) for sid in by_id}
    changed: Dict[str, Dict[str, Any]] = {}
    for pid, entry in current.items():
        members = {cid for team in team_sets(entry) for cid in team}
        if team_sets(fresh[pid]) != team_sets(entry) \
                and team_sets(entry, roster) == team_sets(fresh[pid], members):
            changed[pid] = {**fresh[pid], "name": entry.get("name") or fresh[pid]["name"]}
    return changed


def _imported_plain(imported_assignments: Any,
                    by_id: Dict[str, Student]) -> List[Dict[str, Any]]:
    return projects_to_ids(build_imported_projects(imported_assignments, by_id))


def refresh_plain_projects(projects: List[Dict[str, Any]],
                           fresh: Dict[str, Dict[str, Any]]) -> List[str]:
    replaced: List[str] = []
    for k, entry in enumerate(projects):
        pid = entry["project_id"]
        if entry["imported"] and pid in fresh:
            projects[k] = {**copy.deepcopy(fresh[pid]), "name": entry["name"]}
            replaced.append(pid)
    return replaced


def project_on_roster(project: Project, by_id: Dict[str, Student],
                      matcher: IdMatcher) -> Project:
    teams = []
    for team in project.teams:
        members = []
        for member in team.members:
            match = matcher.resolve(member.student_id)
            members.append(by_id[match] if match is not None else member)
        teams.append(Team(team_id=team.team_id, members=members, project_id=team.project_id))
    return Project(project_id=project.project_id, name=project.name, teams=teams,
                   imported=project.imported)


def changed_notice(project_ids: Sequence[str]) -> str:
    return (f"The imported history of project(s) {', '.join(project_ids)} changed with the "
            "roster (students who joined bring their earlier teams)")
