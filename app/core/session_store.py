from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app import hosting
from app.utils.serialization import json_safe

from . import engine_session as es
from . import run_activation
from . import run_history as rh
from .config_session import coerce_hosted_algorithm, sanitize_config
from .data_session import (
    DEMO_SEEDED_KEY,
    _IMPORT_SOURCES_KEY,
    _LOADER_KEY,
    _UPLOAD_KEY,
    load_active_dataset,
    refresh_imports,
)
from .ids import IdMatcher
from .models import Configuration

SNAPSHOT_FORMAT = "group-diversifier-session"
SNAPSHOT_VERSION = 1

PRESERVED_PREFIX = "_session_restore"

_MAX_RESTORED_PROJECTS = 1000

_MAX_RESTORED_TEAM_PLACES = 100_000


class _View:
    def __init__(self, state: Any):
        self._state = state

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._state[key] if key in self._state else default
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self._state

    def __getitem__(self, key: str) -> Any:
        return self._state[key]


def _allocation(view: _View) -> Optional[Dict[str, Any]]:
    projects = view.get("projects") or []
    if not projects:
        return None
    plain = es.projects_to_ids(projects)
    active_id = view.get(rh.ACTIVE_KEY)
    unsaved = es.unsaved_edits(view)
    result = view.get("formation_result")
    stats = dict(getattr(result, "algorithm_stats", None) or {})
    reports = dict(view.get(es.REPORTS_KEY) or {})
    return {
        "projects": plain,
        "reports": reports,
        "scored_policy": ((es.scored_policy(view) or es.inferred_scored_policy(view))
                          if reports else None),
        "run_id": active_id,
        "unsaved": bool(unsaved),
        "execution_time": float(getattr(result, "execution_time", 0.0) or 0.0),
        "algorithm_stats": {k: v for k, v in stats.items()
                            if k in ("source", "method", "base_run_id", "edits", "run_id",
                                     "per_project")},
    }


def _preferences(view: _View) -> Dict[str, Dict[str, List[str]]]:
    loader = view.get(_LOADER_KEY)
    if loader is None:
        return {}
    return {
        str(s.student_id): {"prefer": [str(x) for x in s.prefer_teammates or []],
                            "avoid": [str(x) for x in s.avoid_teammates or []]}
        for s in loader.get_students()
    }


def snapshot_session(state: Any) -> Dict[str, Any]:
    view = _View(state)
    config = view.get("config") or Configuration()
    uploaded = view.get(_UPLOAD_KEY)
    sources = view.get(_IMPORT_SOURCES_KEY) or {}
    history = rh.runs(view)
    return json_safe({
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "dataset": ({"kind": "upload", "filename": uploaded[0], "text": uploaded[1]}
                    if uploaded else {"kind": "default"}),
        "import_sources": {slot: {"filename": src[0], "text": src[1]}
                           for slot, src in sources.items()},
        "preferences": _preferences(view),
        "run_history": rh.export_runs(history) if history else None,
        "run_counter": int(view.get(rh.COUNTER_KEY) or 0),
        "active_run_id": view.get(rh.ACTIVE_KEY),
        "allocation": _allocation(view),
    })


def snapshot_json(state: Any) -> bytes:
    return json.dumps(snapshot_session(state), indent=1, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _clear(state: Any) -> None:
    for key in list(state.keys()):
        if not str(key).startswith(PRESERVED_PREFIX):
            del state[key]


def _damaged(where: str, expected: str, value: Any) -> ValueError:
    found = {dict: "an object", list: "a list", str: "text", bool: "true/false"}.get(
        type(value), "a number")
    return ValueError(f"The session file is damaged: '{where}' should be {expected}, "
                      f"not {found}.")


def _expect(value: Any, kind: Any, where: str, expected: str) -> None:
    if value is not None and not isinstance(value, kind):
        raise _damaged(where, expected, value)


def _check_projects(projects: Any, where: str) -> None:
    _expect(projects, list, where, "a list of projects")
    for k, entry in enumerate(projects or []):
        _expect(entry, dict, f"{where}[{k}]", "a project object")
        if entry is None or "project_id" not in entry:
            raise ValueError(f"The session file is damaged: '{where}[{k}]' has no project_id.")
        _expect(entry.get("teams"), list, f"{where}[{k}].teams", "a list of teams")
        for t, team in enumerate(entry.get("teams") or []):
            _expect(team, list, f"{where}[{k}].teams[{t}]", "a list of student ids")


def _check(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("This file is not a Group Diversifier session (download one with "
                         "'⬇️ Download session').")
    version = payload.get("version")
    if not isinstance(version, int) or version > SNAPSHOT_VERSION:
        raise ValueError(f"This session file has version {version!r}; this app reads "
                         f"version {SNAPSHOT_VERSION} or older.")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("The session file has no configuration.")
    _expect(payload.get("dataset"), dict, "dataset", "an object")
    _expect(payload.get("import_sources"), dict, "import_sources", "an object")
    _expect(payload.get("preferences"), dict, "preferences", "an object of student ids")
    for sid, prefs in (payload.get("preferences") or {}).items():
        _expect(prefs, dict, f"preferences.{sid}", "an object with prefer/avoid lists")
        for key in ("prefer", "avoid"):
            _expect((prefs or {}).get(key), list, f"preferences.{sid}.{key}", "a list")
    counter = payload.get("run_counter")
    if counter is not None and (isinstance(counter, bool) or not isinstance(counter, int)):
        raise _damaged("run_counter", "a whole number", counter)
    _expect(payload.get("active_run_id"), str, "active_run_id", "text")
    allocation = payload.get("allocation")
    _expect(allocation, dict, "allocation", "an object")
    if allocation:
        _check_projects(allocation.get("projects"), "allocation.projects")
        projects = allocation.get("projects") or []
        if len(projects) > _MAX_RESTORED_PROJECTS:
            raise ValueError(f"The session file holds {len(projects)} projects; this app "
                             f"restores at most {_MAX_RESTORED_PROJECTS}.")
        places = sum(len(team) for entry in projects for team in entry.get("teams") or [])
        if places > _MAX_RESTORED_TEAM_PLACES:
            raise ValueError(f"The session file holds {places} team places; this app restores "
                             f"at most {_MAX_RESTORED_TEAM_PLACES}.")
        found = run_activation.oversized_team(projects)
        if found is not None:
            raise ValueError(run_activation.oversized_team_message(
                found, "the session file's allocation"))
        reports = allocation.get("reports")
        _expect(reports, dict, "allocation.reports", "an object")
        for pid, report in (reports or {}).items():
            _expect(report, dict, f"allocation.reports.{pid}", "a report object")
        _expect(allocation.get("algorithm_stats"), dict, "allocation.algorithm_stats",
                "an object")
    _check_run_team_sizes(payload.get("run_history"))
    _check_table_sizes(payload)
    return payload


def _check_run_team_sizes(bundle: Any) -> None:
    runs = bundle.get("runs") if isinstance(bundle, dict) else None
    if not isinstance(runs, list):
        return
    for run in runs:
        if not isinstance(run, dict):
            continue
        found = run_activation.oversized_team(run.get("projects"))
        if found is not None:
            raise ValueError(run_activation.oversized_team_message(
                found, f"stored run {run.get('run_id')}"))


def _check_table_sizes(payload: Dict[str, Any]) -> None:
    cap = hosting.max_table_upload_mb()
    if cap is None:
        return
    tables = []
    dataset = payload.get("dataset") or {}
    if dataset.get("kind") == "upload" and dataset.get("text") is not None:
        tables.append((dataset.get("filename") or "the uploaded dataset", dataset["text"]))
    for slot, src in (payload.get("import_sources") or {}).items():
        if isinstance(src, dict) and src.get("text") is not None:
            tables.append((src.get("filename") or slot, src["text"]))
        elif isinstance(src, (list, tuple)) and len(src) == 2:
            tables.append((src[0], src[1]))
    for name, text in tables:
        size = len(str(text).encode("utf-8", "surrogatepass"))
        if size > cap * 1024 * 1024:
            raise ValueError(f"The session file carries the table '{name}' of "
                             f"{size / (1024 * 1024):.1f} MB; this server reads tables of at "
                             f"most {cap} MB (a cohort file is usually well under 1 MB).")


def _reapply_preferences(loader: Any, saved: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    students = {str(s.student_id): s for s in loader.get_students()}
    matcher = IdMatcher(students)
    missing: List[str] = []
    unknown: List[str] = []

    def bind(owner: str, targets: Any) -> List[str]:
        bound: List[str] = []
        for target in (str(x) for x in targets or []):
            resolved = matcher.resolve(target)
            if resolved is None:
                unknown.append(f"{owner}->{target}")
                bound.append(target)
            elif resolved != owner and resolved not in bound:
                bound.append(resolved)
        return bound

    for sid, prefs in (saved or {}).items():
        match = matcher.resolve(sid)
        if match is None:
            if prefs.get("prefer") or prefs.get("avoid"):
                missing.append(str(sid))
            continue
        prefer = bind(match, prefs.get("prefer"))
        avoid = bind(match, prefs.get("avoid"))
        current = students[match]
        if prefer != list(current.prefer_teammates or []) or \
                avoid != list(current.avoid_teammates or []):
            loader.update_student_preferences(match, prefer, avoid)
    return missing, unknown


def _restore_runs(state: Any, payload: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    bundle = payload.get("run_history")
    records: List[rh.RunRecord] = []
    if bundle:
        try:
            runs = bundle.get("runs") if isinstance(bundle, dict) else None
            if isinstance(runs, list) and len(runs) > rh.MAX_RUNS:
                warnings.append(f"The session file holds {len(runs)} runs; only the newest "
                                f"{rh.MAX_RUNS} were restored.")
                bundle = {**bundle, "runs": runs[-rh.MAX_RUNS:]}
            records = rh.import_runs(bundle)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            warnings.append(f"The run history could not be restored: {exc}")
    state[rh.HISTORY_KEY] = records
    numbers = [int(r.run_id[1:]) for r in records
               if r.run_id.startswith("R") and r.run_id[1:].isdigit()]
    state[rh.COUNTER_KEY] = max(numbers + [int(payload.get("run_counter") or 0)])
    active = payload.get("active_run_id")
    state[rh.ACTIVE_KEY] = active if any(r.run_id == active for r in records) else None
    return warnings


def _restore_allocation(state: Any, payload: Dict[str, Any], loader: Any,
                        config: Configuration) -> List[str]:
    warnings: List[str] = []
    allocation = payload.get("allocation")
    record = rh.get_run(state.get(rh.ACTIVE_KEY), state)
    if not allocation and record is not None:
        allocation = {"projects": record.projects, "reports": record.reports,
                      "scored_policy": record.policy or None,
                      "run_id": record.run_id, "unsaved": False,
                      "algorithm_stats": {"run_id": record.run_id, "source": record.source}}
    if not allocation or not allocation.get("projects"):
        return warnings
    students = loader.get_students()
    plain = allocation["projects"]
    reports = {str(k): v for k, v in (allocation.get("reports") or {}).items()}
    scored = allocation.get("scored_policy")
    if not isinstance(scored, dict) or not scored:
        scored = (record.policy if record is not None and record.policy
                  and es.run_holds_reports(record, reports) else config.policy) or None
    stats = dict(allocation.get("algorithm_stats") or {})
    stats["restored_session"] = True
    unsaved = bool(allocation.get("unsaved"))
    if unsaved:
        stats["unsaved"] = True
    outcome = run_activation.activate_stored_projects(
        plain, config=config, students=students, raw_df=loader.get_dataframe(),
        state=state, reports=reports, scored=scored,
        run_id=state.get(rh.ACTIVE_KEY), stats=stats,
        execution_time=float(allocation.get("execution_time") or 0.0),
        whose="the saved allocation",
        roster_hint="it was formed with before restoring this session.",
        tolerant=True)
    warnings.extend(outcome.warnings)
    if unsaved and outcome.activated:
        base = allocation.get("run_id")
        warnings.append("The saved allocation had changes that were not saved as a run "
                        "(e.g. Team Editor edits"
                        + (f" on run {base}" if base else "")
                        + "); they are active again. Save them as a run in the Team Editor to "
                          "keep them in the run history.")
    return warnings


def restore_session(state: Any, payload: Dict[str, Any]) -> List[str]:
    payload = _check(payload)
    restored: Dict[str, Any] = {}
    try:
        warnings = _restore_into(restored, payload)
    except ValueError:
        raise
    except Exception as exc:  # SAFETY: a damaged file must never crash the page or wipe the session
        raise ValueError(f"The session file is damaged ({type(exc).__name__}: {exc}).") from exc
    _clear(state)
    for key, value in restored.items():
        state[key] = value
    return warnings


def _restore_into(state: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    try:
        config = Configuration.from_dict(payload["config"])
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        raise ValueError(f"The session file's configuration could not be read "
                         f"({type(exc).__name__}: {exc}).") from exc
    hosted_note = coerce_hosted_algorithm(config)
    if hosted_note:
        warnings.append(f"The saved configuration's {hosted_note}. Stored runs of the classic "
                        "algorithms stay viewable.")
    try:
        clamp_notes = sanitize_config(config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The session file's configuration could not be read "
                         f"({type(exc).__name__}: {exc}).") from exc
    if clamp_notes:
        warnings.append("The saved configuration was adjusted while restoring: "
                        + "; ".join(clamp_notes) + ".")
    state["config"] = config
    state["projects"] = []
    state["formation_result"] = None

    dataset = payload.get("dataset") or {"kind": "default"}
    if dataset.get("kind") == "upload" and dataset.get("text") is not None:
        state[_UPLOAD_KEY] = (str(dataset.get("filename") or "uploaded.csv"),
                              str(dataset["text"]))
    sources = {}
    for slot, src in (payload.get("import_sources") or {}).items():
        if isinstance(src, dict) and src.get("text") is not None:
            sources[str(slot)] = (str(src.get("filename") or slot), str(src["text"]))
        elif isinstance(src, (list, tuple)) and len(src) == 2:
            sources[str(slot)] = (str(src[0]), str(src[1]))
    state[_IMPORT_SOURCES_KEY] = sources
    state[DEMO_SEEDED_KEY] = True

    loader, error = load_active_dataset(state)
    if error:
        warnings.append(error)
    missing, unknown_targets = _reapply_preferences(loader, payload.get("preferences") or {})
    if missing:
        warnings.append(f"Preference edits of {len(missing)} student(s) were not restored: "
                        f"they are not in the loaded roster (e.g. {', '.join(missing[:5])}).")
    if unknown_targets:
        warnings.append(f"{len(unknown_targets)} preference target(s) of the saved session are "
                        f"not in the loaded roster (e.g. {', '.join(unknown_targets[:5])}); "
                        "they have no effect.")
    roster_ids = {s.student_id for s in loader.get_students()}
    for slot, errors in refresh_imports(roster_ids, int(config.num_projects), state).items():
        warnings.append(f"The saved {slot.replace('_', ' ')} upload could not be read: "
                        + "; ".join(errors[:3]))
    warnings.extend(_restore_runs(state, payload))
    warnings.extend(_restore_allocation(state, payload, loader, config))
    return warnings
