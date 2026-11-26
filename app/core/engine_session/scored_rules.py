from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Optional

from ...engine.policy import Policy
from ...engine.report_views import overall, teams
from .plain_form import digest


RULES_FROM_RUN = "run"

RULES_FROM_SESSION = "session"

RULES_FROM_EARLIER = "earlier"

RULES_FROM_DEFAULTS = "defaults"


def as_policy(policy: Any) -> Policy:
    return policy if isinstance(policy, Policy) else Policy.from_dict(policy)


def scoring_signature(policy: Any) -> str:
    data = as_policy(policy).to_dict()
    for key in ("solver", "name", "description"):
        data.pop(key, None)
    return digest(data)


def _policy_dict(policy: Any) -> Optional[Dict[str, Any]]:
    if policy is None:
        return None
    if isinstance(policy, Policy):
        return policy.to_dict()
    if isinstance(policy, dict) and policy:
        try:
            return Policy.from_dict(copy.deepcopy(policy)).to_dict()
        except Exception:
            return None
    return None


def _reports_signature(reports: Optional[Dict[str, Dict[str, Any]]]) -> str:
    from ...utils.serialization import json_safe

    data = {
        str(pid): [json_safe(overall(report).objective),
                   sorted(sorted(str(s) for s in team.members or [])
                          for team in teams(report))]
        for pid, report in (reports or {}).items() if isinstance(report, dict)
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def run_holds_reports(record: Any, reports: Optional[Dict[str, Dict[str, Any]]]) -> bool:
    from ...utils.serialization import json_safe

    if record is None:
        return False
    return json_safe(record.reports or {}) == json_safe(reports or {})


def same_scoring(a: Any, b: Any) -> bool:
    try:
        return bool(a) and bool(b) and scoring_signature(a) == scoring_signature(b)
    except Exception:
        return False
