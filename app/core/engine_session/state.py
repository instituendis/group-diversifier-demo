from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from .. import session_state

ROSTER_KEY = "_engine_roster"
REPORTS_KEY = "engine_reports"
SCORED_POLICY_KEY = "engine_scored_policy"
META_KEY = "engine_result_meta"


def _session():
    return session_state.resolve_state()


def facade() -> ModuleType:
    return sys.modules[__package__]


def session() -> Any:
    return facade()._session()
