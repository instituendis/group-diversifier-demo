from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from app.core.ids import counts_as_team, id_sort_key, team_number
from app.engine.history import PriorProject, build_pair_history
from app.engine.policy import HistoryPolicy, Policy
from app.engine.roster import EncodedCohort

if TYPE_CHECKING:
    from app.utils.roster_view import RosterView

MAX_EXPOSURE_VALUES = 40
MAX_HEATMAP_VALUES = 10
ANY_DIFFERENCE = "__any__"


def project_teams(project) -> List[List[str]]:
    return [list(dict.fromkeys(m.student_id for m in team.members)) for team in project.teams]


def project_label(project) -> str:
    return f"{project.name} (imported)" if project.imported else project.name


def teammates_by_student(project) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    for team in project.teams:
        ids = set(team.member_ids)
        for sid in ids:
            mapping[sid] = ids - {sid}
    return mapping


def teammates_per_student(projects) -> Dict[str, Counter]:
    together: Dict[str, Counter] = {}
    for project in projects:
        for team in project_teams(project):
            for sid in team:
                together.setdefault(sid, Counter()).update(x for x in team if x != sid)
    return together


def _timeline(projects) -> List[PriorProject]:
    return [PriorProject(project_id=str(p.project_id), teams=project_teams(p))
            for p in projects]


def _cohort(cohort_ids: Sequence[str]) -> EncodedCohort:
    return EncodedCohort(student_ids=list(dict.fromkeys(cohort_ids)))


def weighted_repeats(projects, policy: HistoryPolicy, cohort_ids: Sequence[str]
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cohort = _cohort(cohort_ids)
    cap = policy.max_times_together
    limit = policy.ignore_groups_larger_than
    timeline = _timeline(projects)
    rows = []
    for k, project in enumerate(projects):
        data = build_pair_history(timeline[:k], policy, cohort)
        met_before = repeats = over = 0
        weight = 0.0
        for team in timeline[k].teams:
            indices = sorted({i for i in (cohort.lookup(s) for s in team) if i is not None})
            capped = cap is not None and counts_as_team(len(team), limit)
            for a, b in combinations(indices, 2):
                if data.met(a, b):
                    met_before += 1
                    weight += data.weight(a, b)
                if not policy.enabled:
                    continue
                times = data.times_together(a, b)
                repeats += int(times > 0)
                if capped and times + 1 > cap:
                    over += 1
        rows.append({
            "Project": project_label(project),
            "Pairs met before": met_before,
            "Repeat team pairs": repeats,
            "Contact weight": round(weight, 3),
            "Penalty points": round(weight * policy.repeat_weight, 2) if policy.enabled else 0.0,
            "Over the cap": over,
        })
    final = build_pair_history(timeline, policy, cohort)
    labels = {str(p.project_id): project_label(p) for p in projects}
    ids = cohort.student_ids
    pair_rows = []
    for (a, b), met in final.met_in.items():
        if len(met) < 2:
            continue
        counted = final.cap_counts.get((a, b), 0)
        if not policy.enabled:
            status = ""
        elif cap is not None and counted > cap:
            status = "over cap"
        elif cap is not None and counted >= cap:
            status = "at cap (may not meet again)"
        else:
            status = ""
        pair_rows.append({
            "Student A": ids[a], "Student B": ids[b],
            "Times together": len(met),
            "Counted toward cap": counted,
            "Contact weight": round(final.weights.get((a, b), 0.0), 3),
            "Projects": ", ".join(labels.get(pid, pid) for pid in met),
            "Status": status,
        })
    pair_rows.sort(key=lambda r: (-r["Contact weight"], id_sort_key(r["Student A"]),
                                  id_sort_key(r["Student B"])))
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def pair_shared_projects(projects, policy: Policy, cohort_ids: Sequence[str]
                         ) -> Dict[Tuple[int, int], List[int]]:
    cohort = _cohort(cohort_ids)
    met: Dict[Tuple[int, int], List[int]] = {}
    for k, project in enumerate(projects):
        data = build_pair_history(_timeline([project]), policy.history, cohort)
        for pair, count in data.cap_counts.items():
            if count > 0:
                met.setdefault(pair, []).append(k)
    return met


def per_student_repeat_summary(projects, cohort_ids: Sequence[str],
                               shared_teams: Dict[Tuple[int, int], int]) -> pd.DataFrame:
    on_roster = set(cohort_ids)
    taken: Dict[str, int] = {}
    for project in projects:
        for sid in {sid for t in project_teams(project) for sid in t} & on_roster:
            taken[sid] = taken.get(sid, 0) + 1
    per_student: Dict[str, Counter] = {sid: Counter() for sid in taken}
    for (a, b), count in shared_teams.items():
        per_student.setdefault(cohort_ids[a], Counter())[cohort_ids[b]] = count
        per_student.setdefault(cohort_ids[b], Counter())[cohort_ids[a]] = count
    return pd.DataFrame([
        {"student_id": sid, "projects": taken.get(sid, 0),
         "unique_teammates": len(mates),
         "repeat_teammates": sum(1 for c in mates.values() if c >= 2),
         "max_teams_with_one": max(mates.values(), default=0)}
        for sid, mates in sorted(per_student.items(), key=lambda kv: id_sort_key(kv[0]))
    ])


def teams_shared_distribution(shared_teams: Dict[Tuple[int, int], int]) -> pd.DataFrame:
    distribution: Dict[int, int] = {}
    for count in shared_teams.values():
        distribution[count] = distribution.get(count, 0) + 1
    if not distribution:
        return pd.DataFrame()
    return pd.DataFrame({
        "Teams together": sorted(distribution),
        "Number of pairs": [distribution[t] for t in sorted(distribution)],
    })


def _team_of(projects) -> List[Dict[str, str]]:
    mapping: List[Dict[str, str]] = []
    for project in projects:
        per_team: Dict[str, str] = {}
        for team in project.teams:
            for sid in team.member_ids:
                per_team[sid] = team.team_id
        mapping.append(per_team)
    return mapping


def rotation_matrix(projects, student_ids: Sequence[str]) -> pd.DataFrame:
    team_of = _team_of(projects)
    return pd.DataFrame([
        {"Student": sid, **{
            project_label(project): team_of[i].get(sid, "—")
            for i, project in enumerate(projects)
        }}
        for sid in student_ids
    ])


def rotation_team_columns(projects, student_ids: Sequence[str]) -> List[List[float]]:
    team_of = _team_of(projects)
    return [
        [float(team_number(team_of[i][sid])) if sid in team_of[i] else float("nan")
         for sid in student_ids]
        for i in range(len(projects))
    ]


def pair_efficiency(projects) -> pd.DataFrame:
    rows = []
    cumulative_pairs: Set[Tuple[str, str]] = set()
    for project in projects:
        project_pairs = project.get_all_pairs()
        new_pairs = project_pairs - cumulative_pairs
        cumulative_pairs.update(project_pairs)
        rows.append({
            "Project": project_label(project),
            "Total Pairs": len(project_pairs),
            "New Pairs": len(new_pairs),
            "Cumulative Unique": len(cumulative_pairs),
            "Efficiency": round(len(new_pairs) / len(project_pairs), 3) if project_pairs else 0.0,
        })
    return pd.DataFrame(rows)


def repeat_reasons(explanations: Sequence[Dict], cohort_ids: Sequence[str]
                   ) -> Dict[Tuple[int, int], str]:
    position = {sid: i for i, sid in enumerate(cohort_ids)}
    out: Dict[Tuple[int, int], List[str]] = {}
    for row in explanations:
        ia, ib = position.get(row["a"]), position.get(row["b"])
        if ia is None or ib is None:
            continue
        reasons = out.setdefault((min(ia, ib), max(ia, ib)), [])
        if row["short"] not in reasons:
            reasons.append(row["short"])
    return {pair: ", ".join(reasons) for pair, reasons in out.items()}


def repeat_explanation_table(explanations: Sequence[Dict], projects) -> pd.DataFrame:
    names = {str(p.project_id): project_label(p) for p in projects}
    return pd.DataFrame([
        {"Project": names.get(row["project_id"], f"Project {row['project_id']}"),
         "Pair": f"{row['a']} & {row['b']}",
         "Team": row["team_name"],
         "Met before in": ", ".join(names.get(str(m), f"Project {m}") for m in row["met_in"]),
         "Reason": row["reason"],
         "Suggested fix": row["fix"] or ""}
        for row in explanations
    ], columns=["Project", "Pair", "Team", "Met before in", "Reason", "Suggested fix"])


def teammate_reach(projects, cohort_ids: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cohort = len(set(cohort_ids))
    slots: Counter = Counter()
    taken: Counter = Counter()
    mates: Dict[str, Set[str]] = {}
    timeline = []
    for project in projects:
        for team in project_teams(project):
            for sid in team:
                taken[sid] += 1
                slots[sid] += len(team) - 1
                mates.setdefault(sid, set()).update(x for x in team if x != sid)
        seen = list(taken)
        distinct = [len(mates.get(sid, ())) for sid in seen]
        possible = [min(slots[sid], cohort - 1) for sid in seen]
        timeline.append({
            "Project": project_label(project),
            "Mean distinct teammates": float(np.mean(distinct)) if distinct else 0.0,
            "Maximum possible (no repeats)": float(np.mean(possible)) if possible else 0.0,
            "Fewest distinct teammates": int(min(distinct)) if distinct else 0,
            "Students with a repeat teammate": sum(1 for d, p in zip(distinct, possible) if d < p),
        })
    rows = []
    for sid in sorted(taken, key=id_sort_key):
        distinct = len(mates.get(sid, ()))
        possible = min(slots[sid], cohort - 1)
        rows.append({
            "Student": sid,
            "Projects": taken[sid],
            "Teammate slots": slots[sid],
            "Distinct teammates": distinct,
            "Maximum possible": possible,
            "Repeat teammates": slots[sid] - distinct,
            "Reach": distinct / possible if possible else 1.0,
            "Share of cohort met": distinct / (cohort - 1) if cohort > 1 else 0.0,
        })
    return pd.DataFrame(rows), pd.DataFrame(timeline)


def group_of(view: RosterView, attribute: str, focus: str) -> Dict[str, str]:
    groups = {}
    for sid in view.ids:
        labels = view.labels(sid, attribute)
        if not labels:
            continue
        value = labels[0]
        groups[sid] = value if focus == ANY_DIFFERENCE or value == focus else f"not {focus}"
    return groups


def expected_cross_share(groups: Sequence[str]) -> Optional[float]:
    k = len(groups)
    if k < 2:
        return None
    same = sum(c * (c - 1) for c in Counter(groups).values())
    return 1.0 - same / (k * (k - 1))


def cross_group_exposure(projects, groups: Dict[str, str]) -> pd.DataFrame:
    rows = []
    cross_seen: Set[Tuple[str, str]] = set()
    exposed: Set[str] = set()
    participants: Set[str] = set()
    for project in projects:
        pairs = cross = 0
        members_with_group = []
        for team in project_teams(project):
            known = [sid for sid in team if sid in groups]
            members_with_group.extend(known)
            for a, b in combinations(sorted(known, key=id_sort_key), 2):
                pairs += 1
                if groups[a] != groups[b]:
                    cross += 1
                    cross_seen.add((a, b))
                    exposed.update((a, b))
        participants.update(members_with_group)
        expected = expected_cross_share([groups[s] for s in members_with_group])
        rows.append({
            "Project": project_label(project),
            "Team pairs": pairs,
            "Cross-group pairs": cross,
            "Cross-group share": cross / pairs if pairs else None,
            "Random expectation": expected,
            "Distinct cross-group pairs so far": len(cross_seen),
            "Students not yet exposed": len(participants - exposed),
        })
    return pd.DataFrame(rows)


def category_pair_ratio(projects, groups: Dict[str, str], top: int = MAX_HEATMAP_VALUES):
    order = [v for v, _ in Counter(groups.values()).most_common()]
    keep = order[:top]
    label = {v: (v if v in keep else "other") for v in order}
    values = keep + (["other"] if len(order) > top else [])
    pos = {v: k for k, v in enumerate(values)}
    observed = np.zeros((len(values), len(values)))
    expected = np.zeros_like(observed)
    for project in projects:
        teams = [[s for s in team if s in groups] for team in project_teams(project)]
        members = [s for team in teams for s in team]
        k = len(members)
        n_pairs = sum(len(t) * (len(t) - 1) // 2 for t in teams)
        if k < 2 or not n_pairs:
            continue
        counts = Counter(label[groups[s]] for s in members)
        for a in values:
            for b in values:
                ca, cb = counts.get(a, 0), counts.get(b, 0)
                same = ca * (ca - 1) if a == b else 2 * ca * cb
                expected[pos[a], pos[b]] += n_pairs * same / (k * (k - 1))
        for team in teams:
            for x, y in combinations(team, 2):
                i, j = pos[label[groups[x]]], pos[label[groups[y]]]
                observed[i, j] += 1
                if i != j:
                    observed[j, i] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 1e-9, observed / expected, np.nan)
    return values, ratio, observed
