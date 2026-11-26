from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import session_state

HISTORY_KEY = "run_history"
ACTIVE_KEY = "active_run_id"
COUNTER_KEY = "_run_counter"
MAX_RUNS = 25
MAX_IMPORT_RUNS = 100

SOURCES = ("engine", "legacy", "edited", "imported", "evaluated")


@dataclass
class RunRecord:
    run_id: str
    label: str
    source: str
    created_at: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    projects: List[Dict[str, Any]] = field(default_factory=list)
    reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    parent_run_id: Optional[str] = None
    pinned: bool = False
    roster_signature: str = ""

    def project(self, project_id: str) -> Optional[Dict[str, Any]]:
        for entry in self.projects:
            if str(entry.get("project_id")) == str(project_id):
                return entry
        return None

    def generated_project_ids(self) -> List[str]:
        return [str(p["project_id"]) for p in self.projects if not p.get("imported")]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        names = {f.name for f in fields(cls)}
        payload = {k: copy.deepcopy(v) for k, v in dict(data).items() if k in names}
        run_id = payload.get("run_id")
        payload["run_id"] = run_id if isinstance(run_id, str) and run_id else "R?"
        label = payload.get("label", payload["run_id"])
        payload["label"] = label if isinstance(label, str) else ""
        source = payload.get("source", "imported")
        payload["source"] = source if source in SOURCES else "imported"
        for key in ("created_at", "notes", "roster_signature"):
            value = payload.get(key)
            payload[key] = value if isinstance(value, str) else ""
        parent = payload.get("parent_run_id")
        payload["parent_run_id"] = parent if isinstance(parent, str) else None
        payload["pinned"] = payload.get("pinned") is True
        for key in ("config", "policy", "reports", "summary"):
            if not isinstance(payload.get(key), dict):
                payload[key] = {}
        projects = payload.get("projects")
        payload["projects"] = ([p for p in projects if isinstance(p, dict)]
                               if isinstance(projects, list) else [])
        return cls(**payload)


_RUN_ID_PREFIX = re.compile(r"^R\d+ · ")


def display_label(label: Optional[str]) -> str:
    return _RUN_ID_PREFIX.sub("", label or "")


def run_title(record: RunRecord) -> str:
    label = display_label(record.label)
    return f"{record.run_id} · {label}" if label else record.run_id


def _session():
    return session_state.resolve_state()


def runs(state=None) -> List[RunRecord]:
    state = _session() if state is None else state
    return list(state.get(HISTORY_KEY) or [])


def get_run(run_id: Optional[str], state=None) -> Optional[RunRecord]:
    for record in runs(state):
        if record.run_id == run_id:
            return record
    return None


def active_run(state=None) -> Optional[RunRecord]:
    state = _session() if state is None else state
    return get_run(state.get(ACTIVE_KEY), state)


def next_run_id(state=None) -> str:
    state = _session() if state is None else state
    counter = int(state.get(COUNTER_KEY) or 0) + 1
    state[COUNTER_KEY] = counter
    return f"R{counter}"


def add_run(record: RunRecord, *, activate: bool = True, state=None) -> RunRecord:
    state = _session() if state is None else state
    if not record.run_id or record.run_id == "R?" or get_run(record.run_id, state):
        record.run_id = next_run_id(state)
    if not record.created_at:
        record.created_at = datetime.now().isoformat(timespec="seconds")
    history = runs(state)
    history.append(record)
    active_id = record.run_id if activate else state.get(ACTIVE_KEY)
    while len(history) > MAX_RUNS:
        victim = next((r for r in history if not r.pinned and r.run_id != active_id), None)
        if victim is None:
            break
        history.remove(victim)
    state[HISTORY_KEY] = history
    if activate:
        state[ACTIVE_KEY] = record.run_id
    return record


def delete_run(run_id: str, state=None) -> bool:
    state = _session() if state is None else state
    history = runs(state)
    kept = [r for r in history if r.run_id != run_id]
    state[HISTORY_KEY] = kept
    if state.get(ACTIVE_KEY) == run_id:
        state[ACTIVE_KEY] = None
    return len(kept) != len(history)


def rename_run(run_id: str, label: str, state=None) -> None:
    record = get_run(run_id, state)
    if record is not None and label.strip():
        record.label = label.strip()


def set_pinned(run_id: str, pinned: bool, state=None) -> None:
    record = get_run(run_id, state)
    if record is not None:
        record.pinned = bool(pinned)


def make_record(*, label: str, source: str, config: Dict[str, Any], policy: Dict[str, Any],
                projects: List[Dict[str, Any]], reports: Dict[str, Dict[str, Any]],
                summary: Dict[str, Any], notes: str = "",
                parent_run_id: Optional[str] = None,
                roster_signature: str = "") -> RunRecord:
    return RunRecord(run_id="", label=label, source=source,
                     config=copy.deepcopy(config), policy=copy.deepcopy(policy),
                     projects=copy.deepcopy(projects), reports=copy.deepcopy(reports),
                     summary=dict(summary), notes=notes, parent_run_id=parent_run_id,
                     roster_signature=roster_signature)


def export_runs(records: List[RunRecord]) -> Dict[str, Any]:
    return {"format": "group-diversifier-runs", "version": 1,
            "runs": [r.to_dict() for r in records]}


def import_runs(bundle: Dict[str, Any]) -> List[RunRecord]:
    if not isinstance(bundle, dict) or bundle.get("format") != "group-diversifier-runs":
        raise ValueError("Not a Group Diversifier run export")
    runs = bundle.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("The export's 'runs' section is not a list")
    if len(runs) > MAX_IMPORT_RUNS:
        raise ValueError(f"The export holds {len(runs)} runs; this app imports at most "
                         f"{MAX_IMPORT_RUNS} per file. Export fewer runs or split the file.")
    records = []
    for entry in runs:
        record = RunRecord.from_dict(entry)
        record.pinned = False
        record.roster_signature = ""
        records.append(record)
    return records
