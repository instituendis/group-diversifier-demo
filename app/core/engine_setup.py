from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

from ..engine.sizing import resolve_sizes
from . import engine_session as es
from .models import Configuration

_FEASIBILITY_KEY = "_feasibility_cache"


def _session():
    return es._session()


@dataclass
class EngineSetup:
    students: List[Any]
    raw_df: Optional[pd.DataFrame]
    frame: pd.DataFrame
    policy: Any
    specs: List[Any]
    prior: List[Any]
    imported_count: int
    participants: Any = None

    @property
    def roster_ids(self) -> List[str]:
        return self.frame["student_id"].tolist()


def engine_setup(config: Optional[Configuration] = None, *,
                 refresh: bool = True, state: Any = None,
                 students: Optional[List[Any]] = None,
                 raw_df: Optional[pd.DataFrame] = None,
                 frame: Optional[pd.DataFrame] = None, policy: Any = None,
                 imported: Any = None, participants: Any = None) -> Optional[EngineSetup]:
    from .data_session import get_data_loader, refresh_imports

    state = _session() if state is None else state
    config = config if config is not None else state["config"]
    if students is None:
        loader = get_data_loader()
        students = loader.get_students()
        if raw_df is None:
            raw_df = loader.get_dataframe()
    if not students:
        return None
    if frame is None:
        frame = es.roster_frame(students, raw_df)
    if refresh:
        refresh_imports({s.student_id for s in students}, int(config.num_projects), state)
    if imported is None:
        imported = state.get("imported_assignments")
    if participants is None:
        participants = state.get("project_participants")
    prior = es.imported_prior_projects(imported)
    if policy is None:
        policy = es.current_policy(config, frame)
    specs = es.project_specs(config, frame["student_id"].tolist(), participants=participants,
                             imported_count=len(prior))
    return EngineSetup(students=students, raw_df=raw_df, frame=frame, policy=policy,
                       specs=specs, prior=prior, imported_count=len(prior),
                       participants=participants)


def planned_sizes(spec: Any) -> List[int]:
    return resolve_sizes(len(spec.student_ids), int(spec.size_min), int(spec.size_max),
                         int(spec.size_target), int(spec.num_teams or 0), spec.team_sizes)


def _feasibility_signature(setup: EngineSetup) -> str:
    frame_hash = int(pd.util.hash_pandas_object(setup.frame.astype(str), index=False).sum())
    payload = {
        "policy": setup.policy.to_dict(),
        "specs": [
            [s.project_id, list(s.student_ids), s.team_sizes, s.size_min, s.size_max,
             s.size_target, s.num_teams, sorted((s.locks or {}).items())]
            for s in setup.specs
        ],
        "prior": [[p.project_id, p.teams] for p in setup.prior],
        "frame": frame_hash,
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def feasibility_report(setup: EngineSetup, *, force: bool = False,
                       probe_time_s: float = 0.3):
    from app.engine.feasibility import FeasibilityReport, Finding, check_feasibility

    state = _session()
    signature = _feasibility_signature(setup)
    cached = state.get(_FEASIBILITY_KEY)
    if not force and cached and cached[0] == signature:
        return cached[1]
    try:
        report = check_feasibility(setup.frame, setup.policy, setup.specs, setup.prior,
                                   probe_time_s=probe_time_s)
    except Exception as exc:
        report = FeasibilityReport([Finding("error", "policy", "",
                                            f"The feasibility check failed: {exc}")])
    state[_FEASIBILITY_KEY] = (signature, report)
    return report


def active_rule_count(policy: Any) -> int:
    return (
        len(policy.active_attributes())
        + sum(1 for r in policy.counts if r.enabled)
        + sum(1 for r in policy.isolation if r.enabled)
        + sum(1 for r in policy.numeric_bounds if r.enabled)
        + sum(1 for r in policy.shared if r.enabled)
        + sum(1 for r in policy.pairs if r.enabled)
    )
