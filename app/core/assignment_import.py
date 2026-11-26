import csv
import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.core.csv_check import unterminated_quote_problem
from app.core.ids import IdMatcher, trailing_number_key
from app.core.models import Project, Student, Team

REQUIRED_COLUMNS = ("team_id", "project_id", "student_id")

_MAX_IMPORTED_GROUP = 256

MAX_PROJECT_NUMBER = 9999

CONTIGUITY_ERROR_PREFIX = "Imported projects must form a contiguous prefix"

COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "team_id": ("team_id", "group_id", "team", "group", "team_name", "group_name",
                "team_number", "group_number", "teamid", "groupid"),
    "project_id": ("project_id", "project", "project_number", "project_nr", "projectid"),
    "student_id": ("student_id", "identifier", "student", "student_number", "student_nr",
                   "studentid", "id"),
}

@dataclass
class AssignmentImport:
    teams_by_project: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    imported_project_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dropped_rows: int = 0
    unknown_students: Set[str] = field(default_factory=set)

    @property
    def project_count(self) -> int:
        return len(self.imported_project_ids)


def _csv_reader(raw: bytes | str):
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raw = raw.decode("cp1252", errors="replace")
    raw = raw.lstrip("\ufeff")
    header = next((line for line in raw.splitlines() if line.strip()), "")
    counts = {d: header.count(d) for d in (",", ";", "\t", "|")}
    delimiter = max(counts, key=lambda d: counts[d]) if any(counts.values()) else ","
    problem = unterminated_quote_problem(raw, delimiter)
    if problem:
        raise ValueError(f"Malformed CSV, {problem}. Fix or remove the stray quote.")
    return csv.reader(io.StringIO(raw), delimiter=delimiter)


def _header_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _resolve_columns(header: List[str], wanted: Tuple[str, ...]):
    index: Dict[str, int] = {}
    for idx, name in enumerate(header):
        index.setdefault(_header_key(name), idx)
    cols: Dict[str, int] = {}
    notes: List[str] = []
    missing: List[str] = []
    for column in wanted:
        present = [alias for alias in COLUMN_ALIASES[column] if alias in index]
        if not present:
            missing.append(column)
            continue
        cols[column] = index[present[0]]
        if len(present) > 1 and present[0] != column:
            others = ", ".join(repr(header[index[a]].strip()) for a in present[1:])
            notes.append(f"Column {header[cols[column]].strip()!r} was read as {column} "
                         f"(also present: {others}).")
    if not missing:
        return cols, None, notes
    hints = ", ".join(
        f"{c} (or {COLUMN_ALIASES[c][1]})" if c in missing else c for c in wanted
    )
    error = (f"Missing required column(s): {', '.join(missing)} — the header needs "
             f"the columns {hints}.")
    if "project_id" in missing:
        error += (" Add a project_id column with each row's project number (1, 2, ...); "
                  "one file holds every project.")
    return cols, error, notes


MAX_PROJECT_ID_LENGTH = 32


def project_number_problem(project_id: str) -> Optional[str]:
    if not project_id.isdecimal():
        return "must be a positive integer."
    if len(project_id) > MAX_PROJECT_ID_LENGTH:
        return f"is too long; use a project number between 1 and {MAX_PROJECT_NUMBER}."
    number = int(project_id)
    if number < 1:
        return "must be a positive integer."
    if number > MAX_PROJECT_NUMBER:
        return (f"is above {MAX_PROJECT_NUMBER}; use a project number between 1 and "
                f"{MAX_PROJECT_NUMBER}.")
    return None

def parse_assignments_csv(raw: bytes | str) -> Tuple[AssignmentImport, List[str]]:
    errors: List[str] = []
    try:
        reader = _csv_reader(raw)
        header = next(reader)
    except ValueError as exc:
        return AssignmentImport(), [str(exc)]
    except StopIteration:
        return AssignmentImport(), ["File is empty."]

    cols, error, notes = _resolve_columns(header, REQUIRED_COLUMNS)
    if error:
        return AssignmentImport(), [error]

    last_idx = max(cols[c] for c in REQUIRED_COLUMNS)
    imp = AssignmentImport(warnings=notes)
    seen: Set[Tuple[str, str]] = set()
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) <= last_idx:
            errors.append(f"Line {line_no}: expected at least {len(REQUIRED_COLUMNS)} columns.")
            continue
        team_id = row[cols["team_id"]].strip()
        project_id = row[cols["project_id"]].strip()
        student_id = row[cols["student_id"]].strip()
        if not team_id or not project_id or not student_id:
            errors.append(f"Line {line_no}: team_id, project_id and student_id must all be non-empty.")
            continue
        problem = project_number_problem(project_id)
        if problem:
            errors.append(f"Line {line_no}: project_id {project_id!r} {problem}")
            continue
        key = (project_id, student_id)
        if key in seen:
            errors.append(f"Line {line_no}: student {student_id} appears twice in project {project_id}.")
            continue
        seen.add(key)
        imp.teams_by_project.setdefault(project_id, {}).setdefault(team_id, []).append(student_id)

    if not imp.teams_by_project:
        errors.append("No assignment rows found.")
        return imp, errors

    imp.imported_project_ids = sorted(imp.teams_by_project, key=int)
    expected = [str(i) for i in range(1, len(imp.imported_project_ids) + 1)]
    if imp.imported_project_ids != expected:
        errors.append(
            f"{CONTIGUITY_ERROR_PREFIX} 1..K, got: "
            + ", ".join(imp.imported_project_ids)
            + "."
        )
    return imp, errors


def _roster_keys(roster_ids: Set[str]) -> IdMatcher:
    return IdMatcher(sorted(roster_ids))


def validate_against_roster(imp: AssignmentImport, roster_ids: Set[str]) -> None:
    roster_keys = _roster_keys(roster_ids)
    dropped_rows = 0
    empty_teams: List[str] = []
    duplicates: List[str] = []
    for project_id in list(imp.teams_by_project):
        teams = imp.teams_by_project[project_id]
        placed: Set[str] = set()
        for team_id in list(teams):
            if len(teams[team_id]) > _MAX_IMPORTED_GROUP:
                imp.warnings.append(
                    f"Team {team_id} of project {project_id} has {len(teams[team_id])} members "
                    f"and was dropped: an earlier group larger than {_MAX_IMPORTED_GROUP} is "
                    "not a team history."
                )
                del teams[team_id]
                continue
            kept = []
            for sid in teams[team_id]:
                canonical = roster_keys.resolve(sid)
                if canonical is None:
                    continue
                if canonical in placed:
                    duplicates.append(f"{canonical} (project {project_id})")
                    continue
                placed.add(canonical)
                kept.append(canonical)
            unknown = [sid for sid in teams[team_id] if roster_keys.resolve(sid) is None]
            if unknown:
                dropped_rows += len(unknown)
                imp.unknown_students.update(unknown)
            if kept:
                teams[team_id] = kept
            else:
                del teams[team_id]
                empty_teams.append(f"project {project_id} / team {team_id}")
    imp.dropped_rows = dropped_rows
    if dropped_rows:
        sample = ", ".join(sorted(imp.unknown_students)[:10])
        more = (
            f" … (+{len(imp.unknown_students) - 10} more)"
            if len(imp.unknown_students) > 10
            else ""
        )
        imp.warnings.append(
            f"{dropped_rows} assignment row(s) dropped: student(s) not in the current "
            f"cohort (e.g. {sample}{more})."
        )
    if duplicates:
        imp.warnings.append(
            f"{len(duplicates)} student(s) are listed twice in the same project under "
            f"different spellings of one id (leading zeros); only the first entry was "
            f"kept: {', '.join(duplicates[:10])}{' …' if len(duplicates) > 10 else ''}."
        )
    for label in empty_teams:
        imp.warnings.append(f"Team {label} dropped: no members remain in the current cohort.")


def seed_pair_counts(imp: AssignmentImport) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for teams in imp.teams_by_project.values():
        for member_ids in teams.values():
            for i in range(len(member_ids)):
                for j in range(i + 1, len(member_ids)):
                    pair = tuple(sorted((member_ids[i], member_ids[j])))
                    counts[pair] = counts.get(pair, 0) + 1
    return counts


def build_imported_projects(
    imp: AssignmentImport, students_by_id: Dict[str, Student]
) -> List[Project]:
    projects: List[Project] = []
    for project_id in imp.imported_project_ids:
        teams = [
            Team(
                team_id=team_id,
                members=[students_by_id[sid] for sid in member_ids],
                project_id=project_id,
            )
            for team_id, member_ids in sorted(
                imp.teams_by_project[project_id].items(),
                key=lambda kv: trailing_number_key(kv[0])
            )
        ]
        projects.append(
            Project(project_id=project_id, name=f"Project {project_id}", teams=teams, imported=True)
        )
    return projects


@dataclass
class ParticipantSpec:
    participants: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    dropped_rows: int = 0
    unknown_students: Set[str] = field(default_factory=set)

    def pool_for(self, project_id: str) -> Optional[List[str]]:
        return self.participants.get(str(project_id))


def parse_participants_csv(raw: bytes | str) -> Tuple[ParticipantSpec, List[str]]:
    errors: List[str] = []
    try:
        reader = _csv_reader(raw)
        header = next(reader)
    except ValueError as exc:
        return ParticipantSpec(), [str(exc)]
    except StopIteration:
        return ParticipantSpec(), ["File is empty."]

    cols, error, notes = _resolve_columns(header, ("project_id", "student_id"))
    if error:
        return ParticipantSpec(), [error]

    last_idx = max(cols[c] for c in ("project_id", "student_id"))
    spec = ParticipantSpec(warnings=notes)
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) <= last_idx:
            errors.append(f"Line {line_no}: expected at least 2 columns.")
            continue
        project_id = row[cols["project_id"]].strip()
        student_id = row[cols["student_id"]].strip()
        if not project_id or not student_id:
            errors.append(f"Line {line_no}: project_id and student_id must both be non-empty.")
            continue
        problem = project_number_problem(project_id)
        if problem:
            errors.append(f"Line {line_no}: project_id {project_id!r} {problem}")
            continue
        # INVARIANT: '03' or an Arabic-Indic '٣' must key project 3.
        # project_number_problem bounds the length, so int() is safe here.
        project_id = str(int(project_id))
        members = spec.participants.setdefault(project_id, [])
        if student_id not in members:
            members.append(student_id)

    if not spec.participants:
        errors.append("No participant rows found.")
    return spec, errors


def validate_participants(
    spec: ParticipantSpec,
    roster_ids: Set[str],
    imported_count: int,
    num_projects: int,
) -> None:
    roster_keys = _roster_keys(roster_ids)

    dropped_locked = 0
    dropped_future = 0
    dropped_unknown = 0
    for project_id in list(spec.participants):
        pid = int(project_id)
        if pid <= imported_count:
            dropped_locked += len(spec.participants.pop(project_id))
            continue
        if pid > num_projects:
            dropped_future += len(spec.participants.pop(project_id))
            continue
        kept: List[str] = []
        for sid in spec.participants[project_id]:
            canonical = roster_keys.resolve(sid)
            if canonical is None:
                dropped_unknown += 1
                spec.unknown_students.add(sid)
            elif canonical not in kept:
                kept.append(canonical)
        if kept:
            spec.participants[project_id] = kept
        else:
            del spec.participants[project_id]

    spec.dropped_rows = dropped_locked + dropped_future + dropped_unknown
    if dropped_locked:
        spec.warnings.append(
            f"{dropped_locked} participant row(s) dropped for imported (locked) "
            "projects — their teams are fixed by the assignments import."
        )
    if dropped_future:
        spec.warnings.append(
            f"{dropped_future} participant row(s) dropped for projects beyond "
            f"Projects={num_projects} — raise the Projects setting to use them."
        )
    if dropped_unknown:
        sample = ", ".join(sorted(spec.unknown_students)[:10])
        more = (
            f" … (+{len(spec.unknown_students) - 10} more)"
            if len(spec.unknown_students) > 10
            else ""
        )
        spec.warnings.append(
            f"{dropped_unknown} participant row(s) dropped: student(s) not in the "
            f"cohort (e.g. {sample}{more})."
        )
