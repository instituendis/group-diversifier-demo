from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from app import demo, extensions
from app.core import engine_session as es
from app.core import session_state
from app.core.assignment_import import (
    parse_assignments_csv,
    parse_participants_csv,
    validate_against_roster,
    validate_participants,
)
from app.core.config_loader import ConfigurationLoader
from app.core.config_session import clamp_num_cores, enforce_hosted_configuration, new_configuration
from app.core.data_loader import DataLoader
from app.core import import_sync
from app.core.engine_setup import (  # noqa: F401  names re-exported for callers of this module
    EngineSetup,
    active_rule_count,
    engine_setup,
    feasibility_report,
    planned_sizes,
)
from app.core.models import Configuration
from app.ui_kit import (  # noqa: F401  re-exported: the UI tier's shared kit (app/ui_kit.py)
    NAV_SECTIONS,
    PAGE_REGISTRY,
    PageInfo,
    SAMPLES_DIR,
    SAMPLE_FILES,
    available_pages,
    chip,
    departed_warning,
    ensure_router,
    flash,
    page_exists,
    page_info,
    page_link,
    queue_flash,
    render_flashes,
    require_results,
    sample_download_button,
    switch_to,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR_ENV = "GROUP_DIVERSIFIER_CONFIG_DIR"

_UPLOAD_KEY = "uploaded_csv"
_LOADER_KEY = "_data_loader"
_UPLOAD_SIGNATURES_KEY = "_processed_uploads"
_IMPORT_SOURCES_KEY = "_import_sources"

PRIOR_ASSIGNMENTS_SLOT = "prior_assignments"
PARTICIPANTS_SLOT = "participants"


def _session():
    return session_state.resolve_state()


def get_config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override) if override else ROOT_DIR / "config"


def _default_data_path() -> Path:
    if demo.demo_mode() and demo.DEMO_COHORT.is_file():
        return demo.DEMO_COHORT
    candidates = extensions.hook("default_datasets", ())
    for name in candidates:
        if (ROOT_DIR / name).exists():
            return ROOT_DIR / name
    if candidates:
        return ROOT_DIR / candidates[-1]
    return demo.DEMO_COHORT


def _parse_default_csv(path: str, mtime: float) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return DataLoader.parse_csv_text(handle.read())


_PARSE_DEFAULT_FILE: Optional[Callable[[str, float], pd.DataFrame]] = None


def _parse_default_file(path: str, mtime: float) -> pd.DataFrame:
    global _PARSE_DEFAULT_FILE
    if _PARSE_DEFAULT_FILE is None:
        import streamlit as st

        _PARSE_DEFAULT_FILE = st.cache_data(show_spinner=False)(_parse_default_csv)
    return _PARSE_DEFAULT_FILE(path, mtime)


def _new_loader() -> DataLoader:
    return DataLoader(
        config_loader=ConfigurationLoader(get_config_dir()),
        profile_name=demo.profile_name(),
    )


def _upload_source_key(filename: str, text: str) -> Tuple[str, str, int, str]:
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    return ("upload", filename, len(text), digest)


def _forget_roster_frame(state: Any) -> None:
    es.invalidate_roster_cache(state)


def set_uploaded_csv(filename: str, text: str, state: Any = None) -> None:
    state = _session() if state is None else state
    state[_UPLOAD_KEY] = (filename, text)
    _forget_roster_frame(state)


def clear_uploaded_csv(state: Any = None) -> None:
    state = _session() if state is None else state
    state.pop(_UPLOAD_KEY, None)
    _forget_roster_frame(state)
    if demo.demo_mode():
        state.pop(DEMO_SEEDED_KEY, None)
        state.pop("imported_assignments", None)
        sources = dict(state.get(_IMPORT_SOURCES_KEY, {}))
        sources.pop(PRIOR_ASSIGNMENTS_SLOT, None)
        state[_IMPORT_SOURCES_KEY] = sources
        seed_demo_history(state)


DEMO_SEEDED_KEY = "_demo_history_seeded"


def seed_demo_history(state: Any) -> bool:
    if not demo.demo_mode() or state.get(DEMO_SEEDED_KEY):
        return False
    state[DEMO_SEEDED_KEY] = True
    path = demo.DEMO_PRIOR_PROJECTS
    if state.get(_UPLOAD_KEY) is not None or not path.is_file():
        return False
    if PRIOR_ASSIGNMENTS_SLOT in state.get(_IMPORT_SOURCES_KEY, {}):
        return False
    set_import_source(PRIOR_ASSIGNMENTS_SLOT, path.name,
                      path.read_text(encoding="utf-8-sig"), state)
    loader, _ = load_active_dataset(state)
    config = state.get("config")
    refresh_imports({s.student_id for s in loader.get_students()},
                    int(config.num_projects) if config is not None else 1, state)
    return True


def get_active_source() -> str:
    uploaded = _session().get(_UPLOAD_KEY)
    if uploaded is not None:
        return f"uploaded: {uploaded[0]}"
    return f"default: {_default_data_path().name}"


def load_active_dataset(state: Any) -> Tuple[DataLoader, Optional[str]]:
    loader = state.get(_LOADER_KEY)
    uploaded = state.get(_UPLOAD_KEY)

    if uploaded is not None:
        filename, text = uploaded
        source_key: Tuple[Any, ...] = _upload_source_key(filename, text)
        if loader is not None and getattr(loader, "_source_key", None) == source_key:
            return loader, None
        fresh = _new_loader()
        try:
            fresh.load_dataframe(DataLoader.parse_csv_text(text))
        except Exception as exc:
            # INVARIANT: a committed upload must never brick every page. Drop it,
            # report, and fall back to the default dataset.
            state.pop(_UPLOAD_KEY, None)
            fallback, _ = load_active_dataset(state)
            return fallback, (f"Could not load uploaded file '{filename}' ({exc}). "
                              f"Falling back to {_default_data_path().name}.")
        fresh._source_key = source_key
        state[_LOADER_KEY] = fresh
        if loader is not None:
            _follow_roster_change(state, fresh)
        return fresh, None

    path = _default_data_path()
    mtime = path.stat().st_mtime if path.exists() else 0.0
    source_key = ("file", str(path), mtime)
    if loader is not None and getattr(loader, "_source_key", None) == source_key:
        return loader, None
    fresh = _new_loader()
    if path.exists():
        fresh.load_dataframe(_parse_default_file(str(path), mtime))
    fresh._source_key = source_key
    state[_LOADER_KEY] = fresh
    if loader is not None:
        _follow_roster_change(state, fresh)
    return fresh, None


def _follow_roster_change(state: Any, loader: DataLoader) -> None:
    config = state.get("config")
    students = loader.get_students()
    if not state.get("projects") or config is None or not students:
        return
    try:
        refresh_imports({s.student_id for s in students}, int(config.num_projects), state)
        refreshed = es.refresh_imported_projects(state, students, config=config,
                                                 raw_df=loader.get_dataframe())
    except Exception as exc:
        queue_flash(state, f"The imported history of the active allocation could not follow "
                           f"the new roster ({exc}). Open the Team Editor to refresh it.",
                    "warning")
        return
    if refreshed:
        projects = state.get("projects") or []
        reports = state.get(es.REPORTS_KEY) or {}
        generated = [str(p.project_id) for p in projects if not p.imported]
        scored = [pid for pid in generated if pid in reports]
        notice = es.roster_notice(projects, [s.student_id for s in students])
        if not generated:
            rescored = ""
        elif scored == generated:
            rescored = " and its generated projects were scored again"
        elif scored:
            rescored = f" and project(s) {', '.join(scored)} were scored again"
        else:
            rescored = "; no generated project could be scored with this roster"
        text = (f"{import_sync.changed_notice(refreshed)}, so the active allocation now uses "
                f"it{rescored}. It shows as unsaved edits until you save it as a run in the "
                "Team Editor.")
        if notice is not None and notice[0] == "departed":
            departed = es.departed_students(projects, [s.student_id for s in students])
            text += f" {notice[1]}" + (" The new scores leave them out."
                                       if all(pid in reports for pid in departed) else "")
        elif notice is not None:
            text += f" {notice[1]}"
        queue_flash(state, text, "info" if notice is None or notice[0] == "departed"
                    else "warning")


def get_data_loader() -> DataLoader:
    loader, error = load_active_dataset(_session())
    if error:
        import streamlit as st

        st.error(error)
    return loader


def get_loader_warnings(loader: DataLoader) -> List[str]:
    warnings = getattr(loader, "warnings", None) or []
    return [str(w) for w in warnings]


def _session_defaults() -> Dict[str, Any]:
    return {
        "projects": [],
        "formation_result": None,
    }


def _upgrade_config(config: Configuration) -> None:
    missing = [name for name in Configuration.__dataclass_fields__ if not hasattr(config, name)]
    if missing:
        default = Configuration()
        for name in missing:
            setattr(config, name, copy.deepcopy(getattr(default, name)))


def load_profile_dimensions(profile_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    try:
        profile = ConfigurationLoader(get_config_dir()).load_profile(profile_name)
    except Exception as exc:
        return None, None, str(exc)
    dims = {dim.name: dim.to_dict() for dim in profile.get_enabled_dimensions()}
    return dims, profile.version, None


def sync_profile_dimensions(
    config: Configuration,
    profile_name: str,
    preserve_tuning: bool = False,
) -> Optional[str]:
    dims, version, error = load_profile_dimensions(profile_name)
    if dims is None:
        return error
    if preserve_tuning and config.dimensions_config:
        merged = {}
        for name, dim_cfg in dims.items():
            previous = config.dimensions_config.get(name) or {}
            merged[name] = {
                **dim_cfg,
                **{k: v for k, v in previous.items() if k in ("weight", "enabled")},
            }
        dims = merged
    config.dimensions_config = dims
    config.profile_name = profile_name
    config.profile_version = version or config.profile_version
    return None


def ensure_session_state() -> Configuration:
    ensure_router()
    state = _session()
    for key, default in _session_defaults().items():
        if key not in state:
            state[key] = default

    config = state.get("config")
    if config is None or not hasattr(config, "team_size_min"):
        config = new_configuration()
        state["config"] = config
    _upgrade_config(config)

    if not config.dimensions_config:
        error = sync_profile_dimensions(config, config.profile_name or demo.profile_name())
        if error and config.profile_name != "default":
            sync_profile_dimensions(config, "default")
    seed_demo_history(state)

    config.num_cores = clamp_num_cores(config.num_cores)
    enforce_hosted_configuration(state)
    return config


def upload_signature(uploaded: Any, *extra: Any) -> Tuple[Any, ...]:
    return (
        getattr(uploaded, "file_id", "") or "",
        getattr(uploaded, "name", ""),
        getattr(uploaded, "size", None),
    ) + tuple(extra)


def is_new_upload(slot: str, uploaded: Any, *extra: Any) -> bool:
    seen = _session().get(_UPLOAD_SIGNATURES_KEY, {})
    return seen.get(slot) != upload_signature(uploaded, *extra)


def mark_upload_processed(slot: str, uploaded: Any, *extra: Any) -> None:
    seen = dict(_session().get(_UPLOAD_SIGNATURES_KEY, {}))
    seen[slot] = upload_signature(uploaded, *extra)
    _session()[_UPLOAD_SIGNATURES_KEY] = seen


def set_import_source(slot: str, filename: str, text: str, state: Any = None) -> None:
    state = _session() if state is None else state
    sources = dict(state.get(_IMPORT_SOURCES_KEY, {}))
    sources[slot] = (filename, text)
    state[_IMPORT_SOURCES_KEY] = sources


def get_import_source(slot: str, state: Any = None) -> Optional[Tuple[str, str]]:
    state = _session() if state is None else state
    return state.get(_IMPORT_SOURCES_KEY, {}).get(slot)


def clear_import_source(slot: str) -> None:
    state = _session()
    sources = dict(state.get(_IMPORT_SOURCES_KEY, {}))
    sources.pop(slot, None)
    state[_IMPORT_SOURCES_KEY] = sources
    derived = {
        PRIOR_ASSIGNMENTS_SLOT: "imported_assignments",
        PARTICIPANTS_SLOT: "project_participants",
    }.get(slot)
    if derived:
        state.pop(derived, None)


def refresh_imports(roster_ids: Set[str], num_projects: int,
                    state: Any = None) -> Dict[str, List[str]]:
    state = _session() if state is None else state
    errors: Dict[str, List[str]] = {}
    source = get_import_source(PRIOR_ASSIGNMENTS_SLOT, state)
    if source is not None:
        parsed, parse_errors = parse_assignments_csv(source[1])
        if parse_errors:
            errors[PRIOR_ASSIGNMENTS_SLOT] = parse_errors
            state.pop("imported_assignments", None)
        else:
            validate_against_roster(parsed, set(roster_ids))
            state["imported_assignments"] = parsed

    imported = state.get("imported_assignments")
    imported_count = imported.project_count if imported else 0
    source = get_import_source(PARTICIPANTS_SLOT, state)
    if source is not None:
        spec, spec_errors = parse_participants_csv(source[1])
        if spec_errors:
            errors[PARTICIPANTS_SLOT] = spec_errors
            state.pop("project_participants", None)
        else:
            validate_participants(spec, set(roster_ids), imported_count, int(num_projects))
            state["project_participants"] = spec
    return errors
