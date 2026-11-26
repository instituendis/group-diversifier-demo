from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ...engine.defaults import suggest_policy
from ...engine.policy import Policy
from ...engine.roster import build_roster_frame, infer_attributes
from ..models import Configuration, Student
from .state import ROSTER_KEY
from .state import session as _session


def _data_signature(students: Sequence[Student], raw_df: Optional[pd.DataFrame]) -> str:
    digest = hashlib.sha1()
    for student in students:
        record = student.to_dict()
        digest.update(repr(sorted((str(k), str(v)) for k, v in record.items())).encode("utf-8"))
        digest.update(b"\x1e")
    if raw_df is not None:
        digest.update(repr(list(map(str, raw_df.columns))).encode("utf-8"))
        if len(raw_df):
            cells = pd.util.hash_pandas_object(raw_df.astype(str), index=False).to_numpy()
            digest.update(cells.tobytes())
        digest.update(str(len(raw_df)).encode("utf-8"))
    return digest.hexdigest()


def roster_frame(students: Sequence[Student], raw_df: Optional[pd.DataFrame] = None,
                 *, use_cache: bool = True) -> pd.DataFrame:
    state = _session()
    signature = _data_signature(students, raw_df) if use_cache else None
    cached = state.get(ROSTER_KEY) if use_cache else None
    if cached and cached[0] == signature:
        return cached[1]
    frame = build_roster_frame(list(students), raw_df)
    if use_cache:
        state[ROSTER_KEY] = (signature, frame)
    return frame


def distinct_value_counts(frame: pd.DataFrame, *, limit: int = 2) -> List[Tuple[str, int]]:
    counts: List[Tuple[str, int]] = []
    for name, info in infer_attributes(frame).items():
        if len(counts) >= limit:
            break
        if info.kind not in ("categorical", "multi") or not info.usable:
            continue
        values = {part for cell in frame[name].dropna() for part in str(cell).split("|") if part}
        counts.append((f"Distinct {name.replace('_', ' ')}", len(values)))
    return counts


def invalidate_roster_cache(state: Any = None) -> None:
    state = _session() if state is None else state
    try:
        state.pop(ROSTER_KEY, None)
    except Exception:  # pragma: no cover - mapping without pop
        if ROSTER_KEY in state:
            del state[ROSTER_KEY]


def active_roster() -> Optional[pd.DataFrame]:
    cached = _session().get(ROSTER_KEY)
    return cached[1] if cached else None


def attribute_kinds(frame: pd.DataFrame) -> Dict[str, str]:
    return {name: info.kind for name, info in infer_attributes(frame).items()}


def current_policy(config: Configuration, frame: pd.DataFrame) -> Policy:
    if config.policy:
        return Policy.from_dict(config.policy)
    policy = suggest_policy(frame, team_size=int(config.team_size_target or 5))
    config.policy = policy.to_dict()
    return policy


def save_policy(config: Configuration, policy: Policy) -> None:
    config.policy = policy.to_dict()
