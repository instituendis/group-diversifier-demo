from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ...engine.history import PriorProject
from ...engine.policy import Policy
from ...engine.runner import ProjectSpec
from ...engine.sizing import project_config
from .. import import_sync
from .. import run_history as rh
from ..models import Configuration, Student
from .plain_form import _plain_teams
from .roster_cache import roster_frame
from .scoring_session import rules_of_active_reports
from .state import session as _session


REPEAT_EXPLANATIONS_KEY = "_repeat_explanations"


def search_base_run(state: Any = None) -> Any:
    state = _session() if state is None else state
    result = state.get("formation_result")
    stats = (getattr(result, "algorithm_stats", None) or {}) if result is not None else {}
    record = None
    if stats.get("source") == "edited" and stats.get("base_run_id"):
        record = rh.get_run(stats.get("base_run_id"), state)
    if record is None:
        record = rh.active_run(state)
    seen = set()
    while record is not None and record.source == "edited" and record.run_id not in seen:
        seen.add(record.run_id)
        record = rh.get_run(record.parent_run_id, state) if record.parent_run_id else None
    return record


def _search_stats(record: Any) -> Dict[str, Any]:
    if record is None or record.source != "engine":
        return {}
    solver = (record.policy or {}).get("solver") or {}
    summary = record.summary or {}
    return {"time_limited": bool(summary.get("time_limited")),
            "iterations": solver.get("iterations"),
            "restarts": solver.get("restarts")}


def explain_repeat_pairs(projects: Sequence[Any], config: Configuration,
                         students: Sequence[Student], *, policy: Optional[Policy] = None,
                         raw_df: Optional[pd.DataFrame] = None,
                         frame: Optional[pd.DataFrame] = None,
                         state: Any = None) -> List[Dict[str, Any]]:
    from ...engine.repeats import explain_repeats

    state = _session() if state is None else state
    frame = frame if frame is not None else roster_frame(students, raw_df)
    policy = policy or rules_of_active_reports(state)
    base = search_base_run(state)
    editor = state.get("team_editor")
    locks: Dict[str, Dict[str, int]] = {}
    plain = [(_plain_teams(p), import_sync.is_imported(p)) for p in projects]
    for (pid, _teams), imported in plain:
        if editor is not None and not imported:
            try:
                pinned = editor.effective_locks(pid)
            except Exception:  # noqa: BLE001 - another allocation in the editor
                pinned = {}
            if pinned:
                locks[pid] = dict(pinned)
    roster_ids = frame["student_id"].tolist()
    key = hashlib.sha256(json.dumps(
        [[pid, teams, imported] for (pid, teams), imported in plain]
        + [policy.to_dict(), getattr(base, "run_id", None),
           (base.projects if base is not None else None), locks, roster_ids,
           int(pd.util.hash_pandas_object(frame.astype(str), index=False).sum())],
        sort_keys=True, default=str).encode()).hexdigest()
    cached = state.get(REPEAT_EXPLANATIONS_KEY)
    if isinstance(cached, dict) and cached.get("key") == key:
        return copy.deepcopy(cached["rows"])
    roster = set(roster_ids)
    stats = _search_stats(base)
    rows: List[Dict[str, Any]] = []
    for idx, ((pid, original), imported) in enumerate(plain):
        if imported or not policy.history.enabled:
            continue
        kept = [[sid for sid in team if sid in roster] for team in original]
        pool = [sid for team in kept for sid in team]
        if not pool:
            continue
        eff = project_config(config, idx)
        spec = ProjectSpec(project_id=pid, student_ids=pool,
                           team_sizes=[len(t) for t in kept],
                           size_min=int(eff.team_size_min), size_max=int(eff.team_size_max),
                           size_target=int(eff.team_size_target))
        prior = [PriorProject(project_id=p, teams=t) for (p, t), _ in plain[:idx]]
        base_entry = base.project(pid) if base is not None else None
        base_teams = [list(map(str, t)) for t in base_entry.get("teams", [])] \
            if base_entry else None
        rows.extend(explain_repeats(frame, policy, spec, original, prior_projects=prior,
                                    base_teams=base_teams, locks=locks.get(pid),
                                    stats=stats))
    state[REPEAT_EXPLANATIONS_KEY] = {"key": key, "rows": copy.deepcopy(rows)}
    return rows
