from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.ids import IdMatcher, team_number
from .exporters import neutralise_cell, plain_projects, pretty_team_name, team_names_for
from .roster_view import RosterView

DEFAULT_GROUP_PATTERN = "P{project_id} Team {nn}"
DEFAULT_CATEGORY_PATTERN = "{project}"
DEFAULT_CODE_PATTERN = "P{project_id}-T{nn}"


@dataclass(frozen=True)
class LmsFormat:
    key: str
    title: str
    id_options: Tuple[str, ...]
    default_id: str
    help: str


FORMATS: Dict[str, LmsFormat] = {
    "canvas": LmsFormat(
        "canvas", "Canvas", ("user_id", "login_id", "canvas_user_id"), "user_id",
        "Group set import: identifier column + group_name. `user_id` is the student's "
        "SIS ID; `login_id` the login; `canvas_user_id` Canvas' internal number.",
    ),
    "brightspace": LmsFormat(
        "brightspace", "Brightspace (D2L)", ("Username", "OrgDefinedId"), "Username",
        "Groups tool import into an existing group category: one ENROLL row per student "
        "with group name and group code. Username and an existing Group Code are required "
        "by default; OrgDefinedId is usually the student number.",
    ),
    "moodle": LmsFormat(
        "moodle", "Moodle", ("username",), "username",
        "Upload users: username, course1 (course short name), group1 (created when "
        "missing). Needs site-administrator rights.",
    ),
}


_TEAM_NAME_RE = re.compile(r"(?i)\s*team[\s_-]*\d+\s*")


def _team_number(name: str, index: int) -> int:
    text = str(name)
    if _TEAM_NAME_RE.fullmatch(text):
        number = team_number(text.strip())
        if number is not None:
            return number
    return index + 1


def _fill(pattern: str, values: Dict[str, Any]) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), pattern).strip()


def _team_values(project: Dict[str, Any], team_index: int) -> Dict[str, Any]:
    names = team_names_for(project)
    raw = names[team_index] if team_index < len(names) else f"team_{team_index + 1}"
    number = _team_number(raw, team_index)
    width = max(2, len(str(len(names))))
    return {
        "project_id": project["project_id"],
        "project": project["name"],
        "team": pretty_team_name(raw),
        "n": str(number),
        "nn": str(number).zfill(width),
    }


def group_name(project: Dict[str, Any], team_index: int, *,
               pattern: str = DEFAULT_GROUP_PATTERN) -> str:
    return _fill(pattern, _team_values(project, team_index))


def category_name(project: Dict[str, Any], pattern: str = DEFAULT_CATEGORY_PATTERN) -> str:
    return _fill(pattern, {"project_id": project["project_id"], "project": project["name"]})


def group_code(project: Dict[str, Any], team_index: int, *,
               pattern: str = DEFAULT_CODE_PATTERN) -> str:
    values = _team_values(project, team_index)
    values["project_id"] = re.sub(r"[^A-Za-z0-9]+", "", str(project["project_id"])) or "X"
    code = _fill(pattern, values)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", code).strip("_") or values["nn"]


def identifiers_from_roster(roster: Any, column: str) -> Dict[str, str]:
    view = RosterView(roster)
    if column == "student_id":
        return {sid: sid for sid in view.ids}
    out = {}
    for sid in view.ids:
        value = view.text(sid, column)
        if value:
            out[sid] = value
    return out


def identifier_candidates(roster: Any, min_coverage: float = 0.5) -> List[str]:
    view = RosterView(roster)
    n = len(view.ids)
    out = ["student_id"]
    for column in view.frame.columns:
        if column == "student_id":
            continue
        values = [view.text(sid, column) for sid in view.ids]
        filled = [v for v in values if v]
        if n and len(filled) >= min_coverage * n and len(set(filled)) == len(filled):
            out.append(str(column))
    return out


def _resolve(identifiers: Optional[Dict[str, str]]) -> Callable[[str], str]:
    if not identifiers:
        return lambda sid: sid
    matcher = IdMatcher(identifiers)

    def lookup(sid: str) -> str:
        match = matcher.resolve(sid)
        return (identifiers.get(match) if match is not None else None) or sid

    return lookup


def unmapped_students(projects: Sequence[Any], identifiers: Optional[Dict[str, str]]) -> List[str]:
    if not identifiers:
        return []
    matcher = IdMatcher(identifiers)
    seen: List[str] = []
    for project in plain_projects(projects):
        for team in project["teams"]:
            for sid in team:
                if matcher.resolve(sid) is None and sid not in seen:
                    seen.append(sid)
    return seen


def membership_rows(projects: Sequence[Any], *, project_ids: Optional[Sequence[str]] = None,
                    identifiers: Optional[Dict[str, str]] = None,
                    pattern: str = DEFAULT_GROUP_PATTERN,
                    category_pattern: str = DEFAULT_CATEGORY_PATTERN,
                    code_pattern: str = DEFAULT_CODE_PATTERN) -> List[Dict[str, Any]]:
    wanted = None if project_ids is None else {str(p) for p in project_ids}
    resolve = _resolve(identifiers)
    rows = []
    for project in plain_projects(projects):
        if wanted is None and project.get("imported"):
            continue
        if wanted is not None and project["project_id"] not in wanted:
            continue
        category = category_name(project, category_pattern)
        names = team_names_for(project)
        for k, members in enumerate(project["teams"]):
            group = group_name(project, k, pattern=pattern)
            code = group_code(project, k, pattern=code_pattern)
            for sid in members:
                rows.append({
                    "project_id": project["project_id"], "category": category,
                    "group_name": group, "group_code": code, "team": names[k],
                    "student_id": sid, "identifier": resolve(sid),
                })
    return rows


def _csv(header: Optional[Sequence[str]], rows: Sequence[Sequence[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if header:
        writer.writerow(header)
    writer.writerows([neutralise_cell(v) for v in row] for row in rows)
    return buffer.getvalue()


def canvas_group_csv(projects: Sequence[Any], *, project_ids: Optional[Sequence[str]] = None,
                     id_column: str = "user_id", identifiers: Optional[Dict[str, str]] = None,
                     names: Optional[Dict[str, str]] = None,
                     pattern: str = DEFAULT_GROUP_PATTERN) -> str:
    if id_column not in FORMATS["canvas"].id_options:
        raise ValueError(f"Canvas identifier column must be one of {FORMATS['canvas'].id_options}")
    rows = membership_rows(projects, project_ids=project_ids, identifiers=identifiers,
                           pattern=pattern)
    if names:
        return _csv(["name", id_column, "group_name"],
                    [[names.get(r["student_id"], ""), r["identifier"], r["group_name"]] for r in rows])
    return _csv([id_column, "group_name"], [[r["identifier"], r["group_name"]] for r in rows])


BRIGHTSPACE_HEADER = ("Action", "OrgDefinedId", "Username", "Group Name", "Group Code")


def brightspace_group_csv(projects: Sequence[Any], *, project_ids: Optional[Sequence[str]] = None,
                          id_field: str = "Username",
                          identifiers: Optional[Dict[str, str]] = None,
                          org_identifiers: Optional[Dict[str, str]] = None,
                          pattern: str = DEFAULT_GROUP_PATTERN,
                          code_pattern: str = DEFAULT_CODE_PATTERN,
                          header: bool = True) -> str:
    if id_field not in FORMATS["brightspace"].id_options:
        raise ValueError(f"Brightspace identifier must be one of {FORMATS['brightspace'].id_options}")
    rows = membership_rows(projects, project_ids=project_ids, identifiers=identifiers,
                           pattern=pattern, code_pattern=code_pattern)
    resolve_org = _resolve(org_identifiers) if org_identifiers else None
    out = []
    for r in rows:
        if id_field == "OrgDefinedId":
            org, user = r["identifier"], ""
        else:
            org = resolve_org(r["student_id"]) if resolve_org else ""
            user = r["identifier"]
        out.append(["ENROLL", org, user, r["group_name"], r["group_code"]])
    return _csv(BRIGHTSPACE_HEADER if header else None, out)


def moodle_group_csv(projects: Sequence[Any], *, course_shortname: str,
                     project_ids: Optional[Sequence[str]] = None,
                     identifiers: Optional[Dict[str, str]] = None,
                     pattern: str = DEFAULT_GROUP_PATTERN) -> str:
    course = str(course_shortname or "").strip()
    if not course:
        raise ValueError("Moodle needs the course short name (course1)")
    rows = membership_rows(projects, project_ids=project_ids, identifiers=identifiers,
                           pattern=pattern)
    return _csv(["username", "course1", "group1"],
                [[r["identifier"], course, r["group_name"]] for r in rows])


def lms_group_csv(lms: str, projects: Sequence[Any], **options: Any) -> str:
    generators = {
        "canvas": canvas_group_csv,
        "brightspace": brightspace_group_csv,
        "moodle": moodle_group_csv,
    }
    if lms not in generators:
        raise ValueError(f"Unknown LMS '{lms}' (use one of {', '.join(generators)})")
    return generators[lms](projects, **options)


def lms_zip(lms: str, projects: Sequence[Any], project_ids: Sequence[str],
            **options: Any) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for pid in project_ids:
            text = lms_group_csv(lms, projects, project_ids=[pid], **options)
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(pid))
            archive.writestr(f"{lms}_groups_project_{safe}.csv", text)
    return buffer.getvalue()
