from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..core.ids import IdMatcher
from ..engine.roster import MULTI_SEPARATOR, _clean, _to_float, infer_attribute, is_reserved

MISSING_LABEL = "(missing)"


def roster_table(roster: Any, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if isinstance(roster, pd.DataFrame):
        frame = roster.copy()
    else:
        from ..engine.roster import build_roster_frame

        frame = build_roster_frame(list(roster or []), raw_df)
    if "student_id" not in frame.columns:
        frame.insert(0, "student_id", [str(i + 1) for i in range(len(frame))])
    frame["student_id"] = frame["student_id"].map(lambda v: _clean(v) or "")
    return frame.reset_index(drop=True)


class RosterView:
    def __init__(self, roster: Any, raw_df: Optional[pd.DataFrame] = None):
        self.frame = roster_table(roster, raw_df)
        self.ids: List[str] = self.frame["student_id"].tolist()
        self._positions: Dict[str, int] = {}
        for pos, sid in enumerate(self.ids):
            self._positions.setdefault(sid, pos)
        self._matcher = IdMatcher(self.ids)
        self._kinds: Dict[str, str] = {}
        self._cells: Dict[str, List[Any]] = {}

    def position(self, student_id: Any) -> Optional[int]:
        sid = _clean(student_id)
        if sid is None:
            return None
        match = self._matcher.resolve(sid)
        return None if match is None else self._positions.get(match)

    def __contains__(self, student_id: Any) -> bool:
        return self.position(student_id) is not None

    def resolve(self, student_id: Any) -> Optional[str]:
        sid = _clean(student_id)
        return None if sid is None else self._matcher.resolve(sid)

    @property
    def attributes(self) -> List[str]:
        return [c for c in self.frame.columns if c != "student_id" and not is_reserved(str(c))]

    def kind(self, attribute: str) -> str:
        if attribute not in self._kinds:
            if attribute not in self.frame.columns:
                self._kinds[attribute] = "empty"
            else:
                inferred = infer_attribute(self.frame[attribute], attribute).kind
                self._kinds[attribute] = (inferred if inferred in ("numeric", "multi", "empty")
                                          else "categorical")
        return self._kinds[attribute]

    def usable_attributes(self) -> List[str]:
        return [a for a in self.attributes if self.kind(a) != "empty"]

    def _column(self, attribute: str) -> List[Any]:
        if attribute not in self._cells:
            self._cells[attribute] = (self.frame[attribute].tolist()
                                      if attribute in self.frame.columns else [None] * len(self.ids))
        return self._cells[attribute]

    def raw(self, student_id: Any, attribute: str) -> Any:
        pos = self.position(student_id)
        return None if pos is None else self._column(attribute)[pos]

    def text(self, student_id: Any, attribute: str) -> str:
        value = self.raw(student_id, attribute)
        if self.kind(attribute) == "numeric":
            number = _to_float(value)
            if number is None:
                return ""
            return str(int(number)) if float(number).is_integer() else f"{number:.2f}"
        return _clean(value) or ""

    def labels(self, student_id: Any, attribute: str) -> List[str]:
        text = _clean(self.raw(student_id, attribute))
        if text is None:
            return []
        if self.kind(attribute) == "multi":
            return sorted({p.strip() for p in text.split(MULTI_SEPARATOR) if p.strip()})
        return [text]

    def number(self, student_id: Any, attribute: str) -> Optional[float]:
        return _to_float(self.raw(student_id, attribute))

    def counts(self, members: Sequence[Any], attribute: str) -> Dict[str, int]:
        tally: Counter = Counter()
        for sid in members:
            labels = self.labels(sid, attribute)
            if not labels:
                tally[MISSING_LABEL] += 1
            for label in labels:
                tally[label] += 1
        return dict(sorted(tally.items(), key=lambda kv: (kv[0] == MISSING_LABEL, -kv[1], kv[0])))

    def mean(self, members: Sequence[Any], attribute: str) -> Optional[float]:
        values = [x for x in (self.number(sid, attribute) for sid in members) if x is not None]
        return sum(values) / len(values) if values else None

    def cohort_counts(self, attribute: str, members: Optional[Sequence[Any]] = None) -> Dict[str, int]:
        return self.counts(self.ids if members is None else members, attribute)

    def summary(self, members: Sequence[Any], attribute: str, top: Optional[int] = None) -> str:
        if self.kind(attribute) == "numeric":
            mean = self.mean(members, attribute)
            return "–" if mean is None else f"mean {mean:.1f}"
        items = list(self.counts(members, attribute).items())
        shown = items[:top] if top else items
        text = " · ".join(f"{value} {count}" for value, count in shown)
        return text + (" …" if top and len(items) > top else "")
