from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .ids import display_team_name, id_sort_key


def teammate_pairs(teams: Sequence[Sequence[str]],
                   only: Optional[Set[str]] = None) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for team in teams:
        ids = sorted({str(s) for s in team if only is None or str(s) in only})
        for a_pos, a in enumerate(ids):
            for b in ids[a_pos + 1:]:
                pairs.add((a, b))
    return pairs


def _comb2(n: int) -> float:
    return n * (n - 1) / 2.0


def adjusted_rand_index(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Both labelings must cover the same items")
    n = len(labels_a)
    if n < 2:
        return 1.0
    cells: Dict[Tuple[Any, Any], int] = {}
    rows: Dict[Any, int] = {}
    cols: Dict[Any, int] = {}
    for a, b in zip(labels_a, labels_b):
        cells[(a, b)] = cells.get((a, b), 0) + 1
        rows[a] = rows.get(a, 0) + 1
        cols[b] = cols.get(b, 0) + 1
    index = sum(_comb2(v) for v in cells.values())
    sum_a = sum(_comb2(v) for v in rows.values())
    sum_b = sum(_comb2(v) for v in cols.values())
    expected = sum_a * sum_b / _comb2(n)
    maximum = (sum_a + sum_b) / 2.0
    if abs(maximum - expected) < 1e-12:
        return 1.0
    return (index - expected) / (maximum - expected)


def match_teams(teams_a: Sequence[Sequence[str]],
                teams_b: Sequence[Sequence[str]]) -> Dict[int, int]:
    if not teams_a or not teams_b:
        return {}
    from scipy.optimize import linear_sum_assignment

    sets_b = [set(map(str, t)) for t in teams_b]
    overlap = [[len(set(map(str, ta)) & sb) for sb in sets_b] for ta in teams_a]
    rows, cols = linear_sum_assignment([[-v for v in row] for row in overlap])
    return {int(r): int(c) for r, c in zip(rows, cols)}


def allocation_similarity(teams_a: Sequence[Sequence[str]], teams_b: Sequence[Sequence[str]], *,
                          names_a: Optional[Sequence[str]] = None,
                          names_b: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    names_a = list(names_a or [])
    names_b = list(names_b or [])

    def name(names: List[str], t: int) -> str:
        return display_team_name(names[t] if t < len(names) else None, t)

    team_a = {str(s): t for t, team in enumerate(teams_a) for s in team}
    team_b = {str(s): t for t, team in enumerate(teams_b) for s in team}
    common = sorted(set(team_a) & set(team_b), key=id_sort_key)
    common_set = set(common)
    pairs_a = teammate_pairs(teams_a, common_set)
    pairs_b = teammate_pairs(teams_b, common_set)
    shared = len(pairs_a & pairs_b)
    union = len(pairs_a | pairs_b)
    matching = match_teams(teams_a, teams_b)
    moved = [{"student": s, "team_a": name(names_a, team_a[s]), "team_b": name(names_b, team_b[s])}
             for s in common if matching.get(team_a[s]) != team_b[s]]
    sets_a = {frozenset(map(str, t)) for t in teams_a if t}
    sets_b = {frozenset(map(str, t)) for t in teams_b if t}
    nothing_common = not common
    return {
        "common_students": len(common),
        "pairs_a": len(pairs_a),
        "pairs_b": len(pairs_b),
        "shared_pairs": shared,
        "pair_overlap": 2.0 * shared / (len(pairs_a) + len(pairs_b))
                        if len(pairs_a) + len(pairs_b) else (0.0 if nothing_common else 1.0),
        "share_of_a": shared / len(pairs_a) if pairs_a else 0.0,
        "share_of_b": shared / len(pairs_b) if pairs_b else 0.0,
        "jaccard": shared / union if union else (0.0 if nothing_common else 1.0),
        "ari": 0.0 if nothing_common
               else adjusted_rand_index([team_a[s] for s in common], [team_b[s] for s in common]),
        "identical_teams": len(sets_a & sets_b),
        "teams_a": len(sets_a),
        "teams_b": len(sets_b),
        "moved": moved,
        "n_moved": len(moved),
        "only_a": sorted(set(team_a) - common_set, key=id_sort_key),
        "only_b": sorted(set(team_b) - common_set, key=id_sort_key),
        "matching": matching,
    }
