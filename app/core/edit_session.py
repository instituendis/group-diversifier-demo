from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..engine.sizing import size_problems
from .engine_session import fingerprint
from .ids import IdMatcher, display_team_name, team_name_number
from . import import_sync

UNDO_LIMIT = 50
LOG_LIMIT = 200

_BOOKKEEPING_ACTIONS = ("undo", "redo", "save", "lock", "unlock", "locks", "lock_team",
                        "unlock_team")

Op = Dict[str, Any]


class EditError(ValueError): ...


def normalise_projects(projects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for entry in projects:
        pid = str(entry["project_id"])
        teams = [[str(s) for s in team] for team in entry.get("teams", [])]
        names = [str(n) if n else "" for n in (entry.get("team_names") or [])][:len(teams)]
        names += [f"team_{k + 1}" for k in range(len(names), len(teams))]
        out.append({
            "project_id": pid,
            "name": entry.get("name") or f"Project {pid}",
            "imported": bool(entry.get("imported", False)),
            "teams": teams,
            "team_names": [n or f"team_{k + 1}" for k, n in enumerate(names)],
        })
    return out


@dataclass
class EditSession:
    projects: List[Dict[str, Any]]
    project_id: str = ""
    locks: Dict[str, Dict[str, int]] = field(default_factory=dict)
    locked_teams: Dict[str, List[int]] = field(default_factory=dict)
    reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stale: List[str] = field(default_factory=list)
    original: List[Dict[str, Any]] = field(default_factory=list)
    original_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    base_run_id: Optional[str] = None
    base_fingerprint: str = ""
    log: List[Dict[str, Any]] = field(default_factory=list)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls, projects: Sequence[Dict[str, Any]], *, project_id: Optional[str] = None,
              base_run_id: Optional[str] = None,
              reports: Optional[Dict[str, Dict[str, Any]]] = None) -> "EditSession":
        plain = normalise_projects(projects)
        ids = [p["project_id"] for p in plain]
        editable = [p["project_id"] for p in plain if not p["imported"]]
        if project_id is not None and str(project_id) in ids:
            selected = str(project_id)
        else:
            selected = editable[0] if editable else (ids[0] if ids else "")
        teams = {p["project_id"]: {frozenset(t) for t in p["teams"] if t} for p in plain}
        known = {str(k): v for k, v in (reports or {}).items()
                 if teams.get(str(k)) == {frozenset(str(s) for s in t.get("members") or [])
                                          for t in (v or {}).get("teams") or []}}
        session = cls(projects=plain, project_id=selected, reports=dict(known),
                      original=copy.deepcopy(plain), original_reports=dict(known),
                      base_run_id=base_run_id, base_fingerprint=fingerprint(plain))
        session.stale = [pid for pid in editable if pid not in known]
        return session

    @property
    def dirty(self) -> bool:
        return fingerprint(self.projects) != self.base_fingerprint

    @property
    def has_changes(self) -> bool:
        return fingerprint(self.projects) != fingerprint(self.original) or any(
            len(p["teams"]) != len(o["teams"]) for p, o in zip(self.projects, self.original))

    @property
    def edit_count(self) -> int:
        return sum(1 for e in self.log if e["action"] not in _BOOKKEEPING_ACTIONS)

    def project_ids(self) -> List[str]:
        return [p["project_id"] for p in self.projects]

    def editable_project_ids(self) -> List[str]:
        return [p["project_id"] for p in self.projects if not p["imported"]]

    def _pid(self, project_id: Optional[str]) -> str:
        return str(project_id) if project_id is not None else self.project_id

    def project(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        pid = self._pid(project_id)
        for entry in self.projects:
            if entry["project_id"] == pid:
                return entry
        raise EditError(f"Project {pid} is not part of this allocation")

    def project_index(self, project_id: Optional[str] = None) -> int:
        pid = self._pid(project_id)
        for k, entry in enumerate(self.projects):
            if entry["project_id"] == pid:
                return k
        raise EditError(f"Project {pid} is not part of this allocation")

    def is_editable(self, project_id: Optional[str] = None) -> bool:
        return not self.project(project_id)["imported"]

    def select(self, project_id: str) -> None:
        self.project(project_id)
        self.project_id = str(project_id)

    def teams(self, project_id: Optional[str] = None) -> List[List[str]]:
        return self.project(project_id)["teams"]

    def team_names(self, project_id: Optional[str] = None) -> List[str]:
        return list(self.project(project_id)["team_names"])

    def display_names(self, project_id: Optional[str] = None) -> List[str]:
        return [display_team_name(n, k) for k, n in enumerate(self.team_names(project_id))]

    def student_ids(self, project_id: Optional[str] = None) -> List[str]:
        return [sid for team in self.teams(project_id) for sid in team]

    def team_of(self, student_id: Any, project_id: Optional[str] = None) -> Optional[int]:
        sid = str(student_id).strip()
        for t, team in enumerate(self.teams(project_id)):
            if sid in team:
                return t
        return None

    def explicit_locks(self, project_id: Optional[str] = None) -> Dict[str, int]:
        return dict(self.locks.get(self._pid(project_id), {}))

    def locked_team_indices(self, project_id: Optional[str] = None) -> List[int]:
        return sorted(self.locked_teams.get(self._pid(project_id), []))

    def is_team_locked(self, team: int, project_id: Optional[str] = None) -> bool:
        return int(team) in self.locked_teams.get(self._pid(project_id), [])

    def effective_locks(self, project_id: Optional[str] = None) -> Dict[str, int]:
        pid = self._pid(project_id)
        teams = self.teams(pid)
        out: Dict[str, int] = {}
        for sid, t in self.locks.get(pid, {}).items():
            current = self.team_of(sid, pid)
            if current is not None:
                out[sid] = current
        for t in self.locked_teams.get(pid, []):
            if 0 <= t < len(teams):
                for sid in teams[t]:
                    out[sid] = t
        return out

    def is_student_locked(self, student_id: Any, project_id: Optional[str] = None) -> bool:
        return str(student_id).strip() in self.effective_locks(project_id)

    def affected_project_ids(self, project_id: Optional[str] = None) -> List[str]:
        k = self.project_index(project_id)
        return [p["project_id"] for p in self.projects[k:] if not p["imported"]]

    def export_projects(self) -> List[Dict[str, Any]]:
        out = []
        for entry in self.projects:
            keep = [k for k, team in enumerate(entry["teams"]) if team]
            out.append({
                "project_id": entry["project_id"],
                "name": entry["name"],
                "imported": entry["imported"],
                "teams": [list(entry["teams"][k]) for k in keep],
                "team_names": [entry["team_names"][k] for k in keep],
            })
        return out

    def report_team_map(self, roster_ids: Iterable[str],
                        project_id: Optional[str] = None) -> Dict[int, int]:
        roster = IdMatcher(roster_ids)
        mapping: Dict[int, int] = {}
        k = 0
        for t, team in enumerate(self.teams(project_id)):
            if any(roster.resolve(sid) is not None for sid in team):
                mapping[t] = k
                k += 1
        return mapping

    def changed_project_ids(self) -> List[str]:
        before = {p["project_id"]: [t for t in p["teams"] if t] for p in self.original}
        return [p["project_id"] for p in self.projects
                if [t for t in p["teams"] if t] != before.get(p["project_id"], [])]

    def size_report(self, size_min: int, size_max: int,
                    project_id: Optional[str] = None) -> List[str]:
        teams = self.teams(project_id)
        names = self.display_names(project_id)
        sizes = [len(t) for t in teams if t]
        problems = size_problems(sizes, int(size_min), int(size_max))
        empty = [names[k] for k, t in enumerate(teams) if not t]
        if empty:
            problems.append(f"{', '.join(empty)} {'is' if len(empty) == 1 else 'are'} empty "
                            f"(left out until students are moved in)")
        return problems

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def _snapshot(self, label: str, project_id: str) -> Dict[str, Any]:
        return {"label": label, "project_id": project_id,
                "projects": copy.deepcopy(self.projects), "locks": copy.deepcopy(self.locks),
                "locked_teams": copy.deepcopy(self.locked_teams),
                "reports": dict(self.reports), "stale": list(self.stale)}

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.projects = snap["projects"]
        self.locks = snap["locks"]
        self.locked_teams = snap["locked_teams"]
        self.reports = snap["reports"]
        self.stale = snap["stale"]
        if snap.get("project_id") in self.project_ids():
            self.project_id = snap["project_id"]

    def _checkpoint(self, label: str, project_id: str) -> None:
        self.undo_stack.append(self._snapshot(label, project_id))
        del self.undo_stack[:-UNDO_LIMIT]
        self.redo_stack.clear()

    def undo(self) -> str:
        if not self.undo_stack:
            raise EditError("Nothing to undo")
        snap = self.undo_stack.pop()
        self.redo_stack.append(self._snapshot(snap["label"], snap["project_id"]))
        self._restore(snap)
        self._log("undo", snap["project_id"], f"Undo: {snap['label']}")
        return snap["label"]

    def redo(self) -> str:
        if not self.redo_stack:
            raise EditError("Nothing to redo")
        snap = self.redo_stack.pop()
        self.undo_stack.append(self._snapshot(snap["label"], snap["project_id"]))
        self._restore(snap)
        self._log("redo", snap["project_id"], f"Redo: {snap['label']}")
        return snap["label"]

    def invalidate_reports(self) -> None:
        self.reports = {}
        self.stale = self.editable_project_ids()
        for snap in self.undo_stack + self.redo_stack:
            snap["reports"] = {}
            snap["stale"] = [p["project_id"] for p in snap["projects"] if not p["imported"]]

    def refresh_imported(self, fresh: Dict[str, Dict[str, Any]]) -> None:
        for projects in [self.projects, self.original,
                         *(snap["projects"] for snap in self.undo_stack + self.redo_stack)]:
            import_sync.refresh_plain_projects(projects, fresh)

    def mark_saved(self, run_id: Optional[str]) -> None:
        self.base_run_id = run_id
        self.base_fingerprint = fingerprint(self.projects)
        self._log("save", self.project_id, f"Saved as run {run_id}")

    def _log(self, action: str, project_id: str, text: str, **extra: Any) -> None:
        self.log.append({"at": datetime.now().isoformat(timespec="seconds"), "action": action,
                         "project_id": project_id, "text": text, **extra})
        del self.log[:-LOG_LIMIT]

    def _edited(self, label: str, project_id: str, action: str, **extra: Any) -> None:
        for pid in self.affected_project_ids(project_id):
            if pid not in self.stale:
                self.stale.append(pid)
        self._log(action, project_id, label, **extra)

    def _require_editable(self, project_id: Optional[str]) -> str:
        pid = self._pid(project_id)
        entry = self.project(pid)
        if entry["imported"]:
            raise EditError(f"{entry['name']} is imported history and cannot be edited")
        return pid

    def _team_index(self, team: Any, pid: str) -> int:
        try:
            t = int(team)
        except (TypeError, ValueError):
            raise EditError(f"Unknown team {team!r}") from None
        n = len(self.teams(pid))
        if not 0 <= t < n:
            raise EditError(f"Team {t + 1} does not exist (the project has {n} teams)")
        return t

    def _name(self, t: int, pid: str) -> str:
        return display_team_name(self.project(pid)["team_names"][t], t)

    def check_move(self, student_id: Any, to_team: Any,
                   project_id: Optional[str] = None) -> Tuple[str, int, int]:
        pid = self._require_editable(project_id)
        sid = str(student_id).strip()
        source = self.team_of(sid, pid)
        if source is None:
            raise EditError(f"Student {sid} is not in {self.project(pid)['name']}")
        target = self._team_index(to_team, pid)
        if target == source:
            raise EditError(f"Student {sid} is already in {self._name(target, pid)}")
        self._check_free(sid, source, pid)
        if self.is_team_locked(target, pid):
            raise EditError(f"{self._name(target, pid)} is locked; unlock it to add students")
        return sid, source, target

    def check_swap(self, a: Any, b: Any,
                   project_id: Optional[str] = None) -> Tuple[str, str, int, int]:
        pid = self._require_editable(project_id)
        sa, sb = str(a).strip(), str(b).strip()
        ta, tb = self.team_of(sa, pid), self.team_of(sb, pid)
        for sid, t in ((sa, ta), (sb, tb)):
            if t is None:
                raise EditError(f"Student {sid} is not in {self.project(pid)['name']}")
        if ta == tb:
            raise EditError(f"Students {sa} and {sb} are already in the same team")
        self._check_free(sa, ta, pid)
        self._check_free(sb, tb, pid)
        return sa, sb, ta, tb

    def _check_free(self, sid: str, team: int, pid: str) -> None:
        if self.is_team_locked(team, pid):
            raise EditError(f"{self._name(team, pid)} is locked; unlock it to move {sid}")
        if sid in self.locks.get(pid, {}):
            raise EditError(f"Student {sid} is locked to {self._name(team, pid)}; "
                            f"unlock them first")

    def move(self, student_id: Any, to_team: Any, project_id: Optional[str] = None,
             *, label: Optional[str] = None) -> None:
        sid, source, target = self.check_move(student_id, to_team, project_id)
        pid = self._pid(project_id)
        text = label or (f"Moved {sid} from {self._name(source, pid)} to "
                         f"{self._name(target, pid)}")
        self._checkpoint(text, pid)
        teams = self.teams(pid)
        teams[source].remove(sid)
        teams[target].append(sid)
        self._edited(text, pid, "move", students=[sid])

    def swap(self, a: Any, b: Any, project_id: Optional[str] = None,
             *, label: Optional[str] = None) -> None:
        sa, sb, ta, tb = self.check_swap(a, b, project_id)
        pid = self._pid(project_id)
        text = label or (f"Swapped {sa} ({self._name(ta, pid)}) with {sb} "
                         f"({self._name(tb, pid)})")
        self._checkpoint(text, pid)
        teams = self.teams(pid)
        ia, ib = teams[ta].index(sa), teams[tb].index(sb)
        teams[ta][ia], teams[tb][ib] = sb, sa
        self._edited(text, pid, "swap", students=[sa, sb])

    def apply(self, op: Op, project_id: Optional[str] = None) -> None:
        kind, students, target = parse_op(op)
        if kind == "swap":
            self.swap(students[0], students[1], project_id)
        else:
            self.move(students[0], target, project_id)

    def apply_suggestion(self, suggestion: Dict[str, Any],
                         project_id: Optional[str] = None) -> None:
        pid = self._pid(project_id)
        kind, students, target = parse_op(suggestion)
        source = suggestion.get("from_team")
        current = [self.team_of(s, pid) for s in students]
        stale = source is not None and current[0] != int(source)
        if kind == "swap":
            stale = stale or (target is not None and current[1] != int(target))
        if stale or None in current:
            raise EditError("This suggestion is out of date; refresh the suggestions")
        text = "Suggestion: " + describe_op(suggestion, self.display_names(pid), current)
        if kind == "swap":
            self.swap(students[0], students[1], pid, label=text)
        else:
            self.move(students[0], target, pid, label=text)

    def lock_student(self, student_id: Any, project_id: Optional[str] = None) -> bool:
        pid = self._require_editable(project_id)
        sid = str(student_id).strip()
        team = self.team_of(sid, pid)
        if team is None:
            raise EditError(f"Student {sid} is not in {self.project(pid)['name']}")
        if sid in self.locks.get(pid, {}):
            return False
        text = f"Locked {sid} to {self._name(team, pid)}"
        self._checkpoint(text, pid)
        self.locks.setdefault(pid, {})[sid] = team
        self._log("lock", pid, text, students=[sid])
        return True

    def unlock_student(self, student_id: Any, project_id: Optional[str] = None) -> bool:
        pid = self._require_editable(project_id)
        sid = str(student_id).strip()
        if sid not in self.locks.get(pid, {}):
            return False
        text = f"Unlocked {sid}"
        self._checkpoint(text, pid)
        del self.locks[pid][sid]
        self._log("unlock", pid, text, students=[sid])
        return True

    def set_locked_students(self, student_ids: Iterable[Any],
                            project_id: Optional[str] = None) -> bool:
        pid = self._require_editable(project_id)
        wanted = {str(s).strip() for s in student_ids}
        for sid in wanted:
            if self.team_of(sid, pid) is None:
                raise EditError(f"Student {sid} is not in {self.project(pid)['name']}")
        current = self.locks.get(pid, {})
        if wanted == set(current):
            return False
        added = sorted(wanted - set(current))
        removed = sorted(set(current) - wanted)
        parts = []
        if added:
            parts.append("locked " + ", ".join(added))
        if removed:
            parts.append("unlocked " + ", ".join(removed))
        text = "; ".join(parts).capitalize()
        self._checkpoint(text, pid)
        self.locks[pid] = {sid: self.team_of(sid, pid) for sid in sorted(wanted)}
        self._log("locks", pid, text, students=added + removed)
        return True

    def lock_team(self, team: Any, project_id: Optional[str] = None) -> bool:
        pid = self._require_editable(project_id)
        t = self._team_index(team, pid)
        if self.is_team_locked(t, pid):
            return False
        text = f"Locked {self._name(t, pid)}"
        self._checkpoint(text, pid)
        self.locked_teams.setdefault(pid, []).append(t)
        self.locked_teams[pid].sort()
        self._log("lock_team", pid, text, team=t)
        return True

    def unlock_team(self, team: Any, project_id: Optional[str] = None) -> bool:
        pid = self._require_editable(project_id)
        t = self._team_index(team, pid)
        if not self.is_team_locked(t, pid):
            return False
        text = f"Unlocked {self._name(t, pid)}"
        self._checkpoint(text, pid)
        self.locked_teams[pid].remove(t)
        self._log("unlock_team", pid, text, team=t)
        return True

    def add_team(self, name: Optional[str] = None, project_id: Optional[str] = None) -> int:
        pid = self._require_editable(project_id)
        entry = self.project(pid)
        if name is None or not str(name).strip():
            numbers = [n for n in (team_name_number(str(n)) for n in entry["team_names"])
                       if n is not None]
            name = f"team_{max(numbers + [len(entry['teams'])]) + 1}"
        name = str(name).strip()
        if name in entry["team_names"]:
            raise EditError(f"There is already a team called {name}")
        text = f"Added empty team {display_team_name(name, len(entry['teams']))}"
        self._checkpoint(text, pid)
        entry["teams"].append([])
        entry["team_names"].append(name)
        self._log("add_team", pid, text)
        return len(entry["teams"]) - 1

    def remove_team(self, team: Any, project_id: Optional[str] = None) -> None:
        pid = self._require_editable(project_id)
        t = self._team_index(team, pid)
        entry = self.project(pid)
        if entry["teams"][t]:
            raise EditError(f"{self._name(t, pid)} still has {len(entry['teams'][t])} "
                            f"member(s); move them out before removing it")
        text = f"Removed empty team {self._name(t, pid)}"
        self._checkpoint(text, pid)
        del entry["teams"][t]
        del entry["team_names"][t]
        locks = self.locks.get(pid, {})
        for sid, k in list(locks.items()):
            locks[sid] = k - 1 if k > t else k
        self.locked_teams[pid] = [k - 1 if k > t else k
                                  for k in self.locked_teams.get(pid, []) if k != t]
        self._log("remove_team", pid, text)

    def place_student(self, student_id: Any, to_team: Any,
                      project_id: Optional[str] = None) -> None:
        pid = self._require_editable(project_id)
        sid = str(student_id).strip()
        if self.team_of(sid, pid) is not None:
            raise EditError(f"Student {sid} is already in {self.project(pid)['name']}")
        t = self._team_index(to_team, pid)
        if self.is_team_locked(t, pid):
            raise EditError(f"{self._name(t, pid)} is locked; unlock it to add students")
        text = f"Placed new student {sid} in {self._name(t, pid)}"
        self._checkpoint(text, pid)
        self.teams(pid)[t].append(sid)
        self._edited(text, pid, "place", students=[sid])

    def remove_students(self, student_ids: Iterable[Any],
                        project_id: Optional[str] = None) -> List[str]:
        pid = self._require_editable(project_id)
        wanted = [str(s).strip() for s in student_ids]
        present = [sid for sid in wanted if self.team_of(sid, pid) is not None]
        if not present:
            raise EditError("None of these students is in the project")
        text = f"Removed {len(present)} student(s) from the project: {', '.join(present[:10])}"
        self._checkpoint(text, pid)
        gone = set(present)
        entry = self.project(pid)
        entry["teams"] = [[s for s in team if s not in gone] for team in entry["teams"]]
        for sid in present:
            self.locks.get(pid, {}).pop(sid, None)
        self._edited(text, pid, "remove", students=present)
        return present

    def set_teams(self, teams: Sequence[Sequence[str]], *, label: str,
                  project_id: Optional[str] = None,
                  team_names: Optional[Sequence[str]] = None) -> None:
        pid = self._require_editable(project_id)
        entry = self.project(pid)
        new_teams = [[str(s) for s in team] for team in teams]
        names = list(team_names or entry["team_names"])[:len(new_teams)]
        names += [f"team_{k + 1}" for k in range(len(names), len(new_teams))]
        self._checkpoint(label, pid)
        entry["teams"] = new_teams
        entry["team_names"] = names
        locks = {}
        for sid in self.locks.get(pid, {}):
            t = self.team_of(sid, pid)
            if t is not None:
                locks[sid] = t
        self.locks[pid] = locks
        self.locked_teams[pid] = [t for t in self.locked_teams.get(pid, []) if t < len(new_teams)]
        self._edited(label, pid, "replace")

    def revert_all(self) -> bool:
        if not self.has_changes:
            return False
        text = "Reverted every project to the original allocation"
        self._checkpoint(text, self.project_id)
        self.projects = copy.deepcopy(self.original)
        for pid in list(self.locks):
            if pid not in self.project_ids():
                del self.locks[pid]
                continue
            self.locks[pid] = {sid: t for sid in self.locks[pid]
                               if (t := self.team_of(sid, pid)) is not None}
        for pid in list(self.locked_teams):
            n = len(self.teams(pid)) if pid in self.project_ids() else 0
            self.locked_teams[pid] = [t for t in self.locked_teams[pid] if t < n]
        self.reports = dict(self.original_reports)
        self.stale = [pid for pid in self.editable_project_ids() if pid not in self.reports]
        self._log("revert", self.project_id, text)
        return True


def parse_op(op: Op) -> Tuple[str, List[str], Optional[int]]:
    if not isinstance(op, dict):
        raise EditError("An edit must be a dict with a 'kind'")
    kind = str(op.get("kind", "swap"))
    students = [str(s) for s in (op.get("students") or [])]
    if "student" in op:
        students = [str(op["student"])] + students
    target = op.get("to_team")
    if kind == "swap":
        if len(students) != 2:
            raise EditError("A swap needs exactly two students")
        return kind, students, (int(target) if target is not None else None)
    if kind == "move":
        if len(students) != 1 or target is None:
            raise EditError("A move needs one student and a target team")
        return kind, students, int(target)
    raise EditError(f"Unknown edit '{kind}'")


def describe_op(op: Op, team_names: Sequence[str],
                teams_now: Optional[Sequence[Optional[int]]] = None) -> str:
    kind, students, target = parse_op(op)

    def name(t: Optional[int]) -> str:
        if t is None:
            return "?"
        return team_names[t] if 0 <= t < len(team_names) else f"Team {t + 1}"

    source = op.get("from_team")
    if source is None and teams_now:
        source = teams_now[0]
    if kind == "swap":
        other = target if target is not None else (teams_now[1] if teams_now else None)
        return f"Swap {students[0]} ({name(source)}) ↔ {students[1]} ({name(other)})"
    return f"Move {students[0]}: {name(source)} → {name(target)}"
