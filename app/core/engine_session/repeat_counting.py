from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ...engine.history import PairHistoryData, PriorProject, build_pair_history
from ...engine.policy import Policy
from ...engine.roster import clean_cell
from .plain_form import _plain_teams


def other_projects_history(projects: Sequence[Any], index: int) -> List[PriorProject]:
    out: List[PriorProject] = []
    for k, project in enumerate(projects):
        if k == index:
            continue
        pid, teams = _plain_teams(project)
        teams = [team for team in teams if team]
        if teams:
            out.append(PriorProject(project_id=pid, teams=teams))
    return out


def history_position(projects: Sequence[Any], index: int) -> int:
    return sum(1 for project in projects[:index] if any(_plain_teams(project)[1]))


def repeat_cap(policy: Policy) -> Optional[int]:
    history = policy.history
    return history.max_times_together if history.enabled else None


def cap_label(policy: Any) -> str:
    history = getattr(policy, "history", policy)
    cap = history.max_times_together if history.enabled else None
    return "none" if cap is None else str(int(cap))


def history_rules_text(policy: Any) -> str:
    history = getattr(policy, "history", policy)
    if not history.enabled:
        return "history is switched off, so no repeats are counted"
    limit = history.ignore_groups_larger_than
    sections = (f"groups larger than {limit} are sections (a weak contact, never a shared team "
                "and never counted toward the repeat cap)" if limit is not None
                else "every earlier group counts as a team, large imported groups included")
    cap = cap_label(history)
    return sections + (f"; repeat cap {cap} team(s) per pair" if cap != "none"
                       else "; no repeat cap")


def allocation_pair_history(projects: Sequence[Any], policy: Policy,
                            roster: Any, *, even_when_off: bool = False) -> PairHistoryData:
    from ...engine.roster import EncodedCohort

    if not policy.history.enabled and not even_when_off:
        return PairHistoryData(project_ids=[_plain_teams(p)[0] for p in projects])
    cohort = EncodedCohort(student_ids=history_ids(roster))
    timeline = [PriorProject(project_id=pid, teams=teams)
                for pid, teams in (_plain_teams(p) for p in projects)]
    return build_pair_history(timeline, policy.history, cohort)


def history_ids(roster: Any) -> List[str]:
    ids = roster["student_id"].tolist() if isinstance(roster, pd.DataFrame) else list(roster)
    return list(dict.fromkeys(clean_cell(sid) or "" for sid in ids))


def shared_team_counts(projects: Sequence[Any], policy: Policy,
                       roster: Any) -> Dict[Tuple[str, str], int]:
    ids = history_ids(roster)
    data = allocation_pair_history(projects, policy, ids, even_when_off=True)
    out: Dict[Tuple[str, str], int] = {}
    for (a, b), count in data.cap_counts.items():
        out[(ids[a], ids[b]) if ids[a] <= ids[b] else (ids[b], ids[a])] = count
    return out
