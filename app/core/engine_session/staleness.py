from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ...engine.report_views import teams
from ...engine.roster import clean_cell
from ..ids import IdMatcher
from .plain_form import plain_projects


def _roster_matcher(roster: Any) -> Optional[IdMatcher]:
    if roster is None:
        return None
    from ...utils.roster_view import RosterView

    if isinstance(roster, IdMatcher):
        matcher = roster
    elif isinstance(roster, RosterView):
        matcher = roster._matcher
    elif isinstance(roster, pd.DataFrame):
        matcher = IdMatcher(roster["student_id"].tolist()) if "student_id" in roster else None
    else:
        matcher = IdMatcher(list(roster))
    return matcher if matcher is not None and len(matcher) else None


def match_team_reports(teams: Sequence[Sequence[Any]], scored: Sequence[Sequence[Any]],
                       roster: Any = None) -> List[Optional[int]]:
    matcher = _roster_matcher(roster)

    def keys(lists: Sequence[Sequence[Any]]) -> List[tuple]:
        out = []
        for members in lists:
            full = frozenset(clean_cell(s) or "" for s in members)
            on_roster = full if matcher is None else frozenset(
                m for m in (matcher.resolve(sid) for sid in full) if m is not None)
            out.append((on_roster, full))
        return out

    mine, theirs = keys(teams), keys(scored)
    mine_count = Counter(key for key, _ in mine)
    theirs_count = Counter(key for key, _ in theirs)
    by_roster = {key: k for k, (key, _) in enumerate(theirs) if key and theirs_count[key] == 1}
    by_full: Dict[frozenset, int] = {}
    for k, (_, full) in enumerate(theirs):
        by_full.setdefault(full, k)
    used: set = set()
    matched: List[Optional[int]] = []
    for key, full in mine:
        k = by_roster.get(key) if key and mine_count[key] == 1 else None
        if k is None:
            k = by_full.get(full)
        if k in used:
            k = None
        if k is not None:
            used.add(k)
        matched.append(k)
    return matched


def report_matches_teams(project: Any, report: Optional[Dict[str, Any]],
                         roster: Any = None) -> bool:
    if not report:
        return False
    plain = plain_projects([project])[0]
    matcher = _roster_matcher(roster)
    if matcher is None:
        teams_plain = {frozenset(team) for team in plain["teams"]}
        scored = {frozenset(clean_cell(s) or "" for s in t.members or [])
                  for t in teams(report)}
        return teams_plain == scored
    scored_teams = [t.members or [] for t in teams(report)]
    matched = match_team_reports(plain["teams"], scored_teams, matcher)
    if len({k for k in matched if k is not None}) != len(scored_teams):
        return False
    return all(k is not None or all(matcher.resolve(sid) is None for sid in team)
               for team, k in zip(plain["teams"], matched))


def stale_report_ids(projects: Sequence[Any], reports: Dict[str, Dict[str, Any]],
                     roster: Any = None) -> List[str]:
    out = []
    for project in plain_projects(projects):
        report = (reports or {}).get(project["project_id"])
        if report is not None and not report_matches_teams(project, report, roster):
            out.append(project["project_id"])
    return out
