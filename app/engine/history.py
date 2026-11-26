from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.ids import counts_as_team

from .policy import HistoryPolicy
from .roster import EncodedCohort

Pair = Tuple[int, int]


@dataclass
class PriorProject:
    project_id: str
    teams: List[List[str]] = field(default_factory=list)


@dataclass
class PairHistoryData:
    weights: Dict[Pair, float] = field(default_factory=dict)
    cap_counts: Dict[Pair, int] = field(default_factory=dict)
    met_in: Dict[Pair, List[str]] = field(default_factory=dict)
    project_ids: List[str] = field(default_factory=list)
    unknown_ids: Set[str] = field(default_factory=set)
    has_earlier_teams: bool = False

    def forbidden_pairs(self, max_times_together: Optional[int]) -> List[Pair]:
        if max_times_together is None:
            return []
        return sorted(p for p, c in self.cap_counts.items() if c + 1 > max_times_together)

    def times_together(self, i: int, j: int) -> int:
        return self.cap_counts.get((i, j) if i < j else (j, i), 0)

    def weight(self, i: int, j: int) -> float:
        return self.weights.get((i, j) if i < j else (j, i), 0.0)

    def met(self, i: int, j: int) -> List[str]:
        return self.met_in.get((i, j) if i < j else (j, i), [])

    def pair_weights(self, repeat_weight: float) -> Dict[Pair, float]:
        return {p: w * repeat_weight for p, w in self.weights.items() if w > 0}


def contact_weight(group_size: int, policy: HistoryPolicy) -> float:
    if policy.contact_model == "team_size" and group_size > 1:
        return min(1.0, (policy.reference_team_size - 1) / (group_size - 1))
    return 1.0


def build_pair_history(
    prior_projects: Sequence[PriorProject],
    policy: HistoryPolicy,
    cohort: EncodedCohort,
    upcoming_index: Optional[int] = None,
) -> PairHistoryData:
    data = PairHistoryData(project_ids=[str(p.project_id) for p in prior_projects])
    total = len(prior_projects)
    upcoming = total if upcoming_index is None else max(0, min(int(upcoming_index), total))
    limit = policy.ignore_groups_larger_than
    for position, project in enumerate(prior_projects):
        if position < upcoming:
            ago = upcoming - position
        else:
            ago = position - upcoming + 1
        factor = policy.decay ** (ago - 1) if ago > 1 else 1.0
        project_id = str(project.project_id)
        for team in project.teams:
            ids = _dedupe(str(s).strip() for s in team if str(s).strip())
            size = len(ids)
            data.has_earlier_teams = data.has_earlier_teams or size > 0
            if size < 2:
                continue
            members = []
            for sid in ids:
                idx = cohort.lookup(sid)
                if idx is None:
                    data.unknown_ids.add(sid)
                elif idx not in members:
                    members.append(idx)
            if len(members) < 2:
                continue
            w = contact_weight(size, policy) * factor
            counts_toward_cap = counts_as_team(size, limit)
            members.sort()
            for a_pos, a in enumerate(members):
                for b in members[a_pos + 1:]:
                    pair = (a, b)
                    if w > 0:
                        data.weights[pair] = data.weights.get(pair, 0.0) + w
                    if counts_toward_cap:
                        data.cap_counts[pair] = data.cap_counts.get(pair, 0) + 1
                    met = data.met_in.setdefault(pair, [])
                    if project_id not in met:
                        met.append(project_id)
    return data


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
