from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence, Set, Tuple

from ...engine.adapters import teams_from_legacy
from ...engine.history import PriorProject
from ..ids import IdMatcher, canonical_id
from ..models import Project, Student, Team

PlainProject = Dict[str, Any]


def team_name_at(names: Sequence[Any], k: int) -> str:
    return names[k] if k < len(names) and names[k] else f"team_{k + 1}"


def plain_projects(projects: Sequence[Any]) -> List[PlainProject]:
    out: List[PlainProject] = []
    for item in projects or []:
        if isinstance(item, Project):
            out.append({
                "project_id": str(item.project_id),
                "name": item.name or f"Project {item.project_id}",
                "imported": bool(item.imported),
                "teams": [[str(m.student_id).strip() for m in t.members] for t in item.teams],
                "team_names": [str(t.team_id) for t in item.teams],
            })
            continue
        entry = dict(item)
        pid = str(entry.get("project_id", len(out) + 1))
        out.append({
            "project_id": pid,
            "name": entry.get("name") or f"Project {pid}",
            "imported": bool(entry.get("imported", False)),
            "teams": [[str(s).strip() for s in team if str(s).strip()]
                      for team in entry.get("teams") or []],
            "team_names": [str(n) if n else "" for n in entry.get("team_names") or []],
        })
    return out


def team_names_for(project: PlainProject) -> List[str]:
    names = project.get("team_names") or []
    return [team_name_at(names, k) for k in range(len(project.get("teams") or []))]


def _plain_teams(project: Any) -> Tuple[str, List[List[str]]]:
    if isinstance(project, Project):
        return str(project.project_id), teams_from_legacy(project)
    if isinstance(project, PriorProject):
        return str(project.project_id), [[str(s) for s in t] for t in project.teams]
    return str(project["project_id"]), [[str(s) for s in t] for t in project.get("teams", [])]


def unmatched_ids(project_teams: Sequence[Dict[str, Any]],
                  roster_ids: Sequence[Any]) -> List[str]:
    matcher = IdMatcher(roster_ids)
    missing: List[str] = []
    seen: Set[str] = set()
    for entry in project_teams:
        for team in entry.get("teams", []):
            for sid in team:
                text = str(sid)
                if text not in seen and matcher.resolve(sid) is None:
                    seen.add(text)
                    missing.append(text)
    return missing


def projects_from_ids(project_teams: Sequence[Dict[str, Any]],
                      students: Sequence[Student]) -> List[Project]:
    by_id = {str(s.student_id): s for s in students}
    matcher = IdMatcher(by_id)
    projects = []
    for entry in project_teams:
        pid = str(entry["project_id"])
        names = entry.get("team_names") or []
        teams = []
        for k, ids in enumerate(entry.get("teams", [])):
            members, seen = [], set()
            for sid in ids:
                match = matcher.resolve(sid)
                if match is not None and match not in seen:
                    seen.add(match)
                    members.append(by_id[match])
            team_id = team_name_at(names, k)
            teams.append(Team(team_id=team_id, members=members, project_id=pid))
        projects.append(Project(project_id=pid, name=entry.get("name") or f"Project {pid}",
                                teams=teams, imported=bool(entry.get("imported", False))))
    return projects


def projects_to_ids(projects: Sequence[Project]) -> List[Dict[str, Any]]:
    return [
        {
            "project_id": str(p.project_id),
            "name": p.name,
            "imported": bool(p.imported),
            "teams": teams_from_legacy(p),
            "team_names": [t.team_id for t in p.teams],
        }
        for p in projects
    ]


def digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def fingerprint(projects: Sequence[Dict[str, Any]]) -> str:
    return digest([
        [str(p["project_id"]), bool(p.get("imported", False)),
         [[canonical_id(s) for s in team] for team in p.get("teams", []) if team]]
        for p in projects
    ])
