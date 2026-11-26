from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_EXCEL_FLOAT_RE = re.compile(r"^(\d+)\.0+$")


def canonical_id(sid: object) -> str:
    text = str(sid).strip()
    match = _EXCEL_FLOAT_RE.match(text)
    if match:
        text = match.group(1)
    if text.isascii() and text.isdigit():
        return str(int(text))
    return text


def id_sort_key(sid: object) -> Tuple[int, int, str]:
    text = str(sid)
    if text.isascii() and text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def id_collisions(ids: Iterable[object]) -> List[List[str]]:
    groups: Dict[str, List[str]] = {}
    for sid in ids:
        text = str(sid).strip()
        members = groups.setdefault(canonical_id(text), [])
        if text not in members:
            members.append(text)
    return [members for members in groups.values() if len(members) > 1]


class IdMatcher:
    def __init__(self, roster_ids: Iterable[object]):
        self.ids: List[str] = []
        self._exact: Dict[str, str] = {}
        self._canonical: Dict[str, Optional[str]] = {}
        for sid in roster_ids:
            text = str(sid).strip()
            if text in self._exact:
                continue
            self.ids.append(text)
            self._exact[text] = text
            key = canonical_id(text)
            self._canonical[key] = None if key in self._canonical else text

    def resolve(self, sid: object) -> Optional[str]:
        text = str(sid).strip()
        hit = self._exact.get(text)
        if hit is not None:
            return hit
        return self._canonical.get(canonical_id(text))

    def __contains__(self, sid: object) -> bool:
        return self.resolve(sid) is not None

    def __len__(self) -> int:
        return len(self.ids)

    def resolve_all(self, ids: Sequence[object]) -> List[Optional[str]]:
        return [self.resolve(sid) for sid in ids]


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def pair_counts(teams: Iterable[Iterable[object]]) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for team in teams:
        members = sorted({str(sid) for sid in team})
        for k, a in enumerate(members):
            for b in members[k + 1:]:
                counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def counts_as_team(group_size: int, ignore_above: Optional[int]) -> bool:
    return ignore_above is None or group_size <= ignore_above


_EMBEDDED_NUMBER_RE = re.compile(r"(\d+)")
_TEAM_NUMBER_RE = re.compile(r"(\d+)$")
_TEAM_NAME_RE = re.compile(r"^team[\s_-]*(\d+)$", re.IGNORECASE)


def team_name_number(name: str) -> Optional[int]:
    match = _TEAM_NAME_RE.match(str(name).strip())
    return int(match.group(1)) if match else None


def display_team_name(name: Optional[str], index: int) -> str:
    text = str(name or "").strip()
    if not text:
        return f"Team {index + 1}"
    number = team_name_number(text)
    return f"Team {number}" if number is not None else text


def team_number(name: str) -> Optional[int]:
    match = _TEAM_NUMBER_RE.search(name)
    return int(match.group(1)) if match else None


def trailing_number_key(name: str) -> tuple:
    number = team_number(name)
    return (number is None, number if number is not None else 0, name)


def team_sort_key(name: str) -> tuple:
    parts = _EMBEDDED_NUMBER_RE.split(str(name).strip())
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part.casefold())
                 for part in parts if part != "")
