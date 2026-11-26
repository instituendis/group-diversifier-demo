from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from ...engine.policy import Policy
from .. import run_history as rh
from .scored_rules import (RULES_FROM_DEFAULTS, RULES_FROM_EARLIER, RULES_FROM_RUN,
                           RULES_FROM_SESSION, _policy_dict, _reports_signature,
                           run_holds_reports, same_scoring)
from .state import REPORTS_KEY, SCORED_POLICY_KEY
from .state import session as _session


def set_active_reports(reports: Dict[str, Dict[str, Any]], scored_with: Any = None,
                       state: Any = None) -> None:
    state = _session() if state is None else state
    state[REPORTS_KEY] = reports
    policy = _policy_dict(scored_with)
    if policy is None:
        if SCORED_POLICY_KEY in state:
            del state[SCORED_POLICY_KEY]
    else:
        state[SCORED_POLICY_KEY] = {"policy": policy, "signature": _reports_signature(reports)}


def clear_active_reports(state: Any = None) -> None:
    state = _session() if state is None else state
    for key in (REPORTS_KEY, SCORED_POLICY_KEY):
        if key in state:
            del state[key]


def scored_policy(state: Any = None) -> Optional[Dict[str, Any]]:
    state = _session() if state is None else state
    entry = state.get(SCORED_POLICY_KEY)
    if not isinstance(entry, dict) or not isinstance(entry.get("policy"), dict):
        return None
    if entry.get("signature") != _reports_signature(state.get(REPORTS_KEY) or {}):
        return None
    return entry["policy"]


def inferred_scored_policy(state: Any = None) -> Optional[Dict[str, Any]]:
    state = _session() if state is None else state
    record = rh.active_run(state)
    config = state.get("config")
    if record is not None and record.policy \
            and run_holds_reports(record, state.get(REPORTS_KEY) or {}):
        return _policy_dict(record.policy)
    return _policy_dict(getattr(config, "policy", None) if config is not None else None)


def _rules_info(state: Any) -> Tuple[Policy, str, Optional[str]]:
    record = rh.active_run(state)
    reports = state.get(REPORTS_KEY) or {}
    config = state.get("config")
    session_policy = getattr(config, "policy", None) if config is not None else None
    recorded = scored_policy(state)
    if recorded is not None:
        try:
            policy = Policy.from_dict(copy.deepcopy(recorded))
        except Exception:
            policy = None
        if policy is not None:
            run_policy = record.policy if record is not None else None
            if run_policy and same_scoring(recorded, run_policy) \
                    and run_holds_reports(record, reports):
                return policy, RULES_FROM_RUN, record.run_id
            if same_scoring(recorded, session_policy):
                return policy, RULES_FROM_SESSION, None
            if run_policy and same_scoring(recorded, run_policy):
                return policy, RULES_FROM_RUN, record.run_id
            for other in reversed(rh.runs(state)):
                if other.policy and same_scoring(recorded, other.policy):
                    return policy, RULES_FROM_EARLIER, other.run_id
            return policy, RULES_FROM_EARLIER, None
    candidates = []
    if record is not None and record.policy and run_holds_reports(record, reports):
        candidates.append((record.policy, RULES_FROM_RUN, record.run_id))
    if session_policy:
        candidates.append((session_policy, RULES_FROM_SESSION, None))
    for policy_dict, source, run_id in candidates:
        try:
            return Policy.from_dict(copy.deepcopy(policy_dict)), source, run_id
        except Exception:
            continue
    return Policy(), RULES_FROM_DEFAULTS, None


def active_rules(state: Any = None) -> Tuple[Policy, str]:
    state = _session() if state is None else state
    policy, source, _ = _rules_info(state)
    return policy, source


def rules_of_active_reports(state: Any = None) -> Policy:
    return active_rules(state)[0]


def rules_source_text(state: Any = None) -> str:
    state = _session() if state is None else state
    _, source, run_id = _rules_info(state)
    if source == RULES_FROM_RUN:
        return f"the rules run {run_id} was scored with"
    if source == RULES_FROM_SESSION:
        return "the current allocation rules"
    if source == RULES_FROM_EARLIER:
        if run_id:
            return (f"earlier allocation rules (the same as run {run_id}'s; changed since "
                    "these scores were computed)")
        return "earlier allocation rules (changed since these scores were computed)"
    return "the default allocation rules (none configured yet)"
