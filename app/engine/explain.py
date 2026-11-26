from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .metrics import team_violations
from .objective import (
    HISTORY_CAP_LABEL,
    HISTORY_LABEL,
    PAIR_PENALTY_LABEL,
    PREFERENCES_LABEL,
    ObjectiveModel,
)
from .search_state import SearchState

_EPS = 1e-9


def _members(model: ObjectiveModel, assignment: Sequence[int], team: int) -> List[int]:
    return [i for i, t in enumerate(assignment) if t == team]


def _fmt(x: float) -> str:
    return f"{x:.1f}" if abs(x - round(x)) > 0.05 else str(int(round(x)))


def _attribute_detail(model: ObjectiveModel, members: List[int], term_index: int) -> str:
    info = model.terms[term_index]
    attr = model.cohort.attributes.get(info.attribute or "")
    if attr is None:
        return ""
    s = len(members)
    if attr.kind == "numeric":
        pool = [v for v in attr.values if not math.isnan(v)]
        vals = [attr.values[i] for i in members if not math.isnan(attr.values[i])]
        if not vals or not pool:
            return "no values in this team"
        mu = sum(pool) / len(pool)
        sd_pool = math.sqrt(sum((v - mu) ** 2 for v in pool) / len(pool))
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        if info.goal == "balance":
            return f"mean {mean:.2f} vs pool {mu:.2f}"
        if info.goal == "cluster":
            return f"spread (sd) {sd:.2f} vs pool {sd_pool:.2f}"
        return f"mean {mean:.2f} vs pool {mu:.2f}; sd {sd:.2f} vs pool {sd_pool:.2f}"
    counts: Dict[int, int] = {}
    pool_counts: Dict[int, int] = {}
    member_set = set(members)
    for i in range(model.n):
        codes = attr.multi[i] if attr.kind == "multi" else (
            [attr.codes[i]] if attr.codes[i] >= 0 else [])
        for c in set(codes):
            pool_counts[c] = pool_counts.get(c, 0) + 1
            if i in member_set:
                counts[c] = counts.get(c, 0) + 1
    names = attr.categories
    if info.goal == "balance":
        rows = []
        for c, k in sorted(pool_counts.items(), key=lambda kv: -kv[1]):
            expected = s * k / model.n
            have = counts.get(c, 0)
            if have or expected >= 0.5:
                rows.append((abs(have - expected), f"{names[c]} {have} (≈{expected:.1f})"))
        rows.sort(key=lambda r: -r[0])
        return ", ".join(text for _, text in rows[:5])
    if info.goal == "diversify":
        dup = [(k, names[c]) for c, k in counts.items() if k >= 2]
        if not dup:
            return "all values distinct"
        dup.sort(key=lambda r: (-r[0], r[1]))
        return ", ".join(f"{name} ×{k}" for k, name in dup)
    distinct = sum(1 for k in counts.values() if k > 0)
    return f"{distinct} different values among {s}"


def explain_team(model: ObjectiveModel, assignment: Sequence[int], team: int
                 ) -> List[Dict[str, Any]]:
    members = _members(model, assignment, team)
    breakdown = model.team_breakdown(members)
    violations = team_violations(model, members)
    by_rule: Dict[str, List[str]] = {}
    for v in violations:
        by_rule.setdefault(v["rule"], []).append(v["text"])
    label_index = {info.label: k for k, info in enumerate(model.terms)}
    out = []
    for label, points in breakdown.items():
        k = label_index.get(label)
        family = model.terms[k].family if k is not None else ""
        if family == "attribute":
            detail = _attribute_detail(model, members, k)
        elif label == HISTORY_LABEL:
            detail = _history_detail(model, members)
        elif label == HISTORY_CAP_LABEL:
            detail = "; ".join(by_rule.get(HISTORY_CAP_LABEL, [])) or "no pair over the cap"
        elif label == PREFERENCES_LABEL:
            detail = _preference_detail(model, members)
        elif family == "bounds":
            detail = _bounds_detail(model, members, k, by_rule.get(label, []))
        elif family == "shared":
            detail = _shared_detail(model, members, k, by_rule.get(label, []))
        elif label == PAIR_PENALTY_LABEL:
            detail = "teammate pairs shared with earlier alternatives"
        else:
            detail = "; ".join(by_rule.get(label, [])) or "satisfied"
        out.append({"term": label, "points": round(float(points), 6), "detail": detail})
    out.sort(key=lambda r: -r["points"])
    return out


def _bounds_detail(model: ObjectiveModel, members: List[int], term_index: int,
                   violations: List[str]) -> str:
    rule_index = next(r for r, b in enumerate(model.bound_rules) if b.term == term_index)
    brule = model.bound_rules[rule_index]
    st = model.bounds_stats(rule_index, members)
    if not st["count"]:
        return f"no {brule.attribute} values in this team"
    parts = [f"mean {_fmt_num(st['mean'])}"]
    if brule.max_range is not None:
        parts.append(f"range {_fmt_num(st['range'])} ({_fmt_num(st['min'])}–{_fmt_num(st['max'])})")
    text = ", ".join(parts)
    if violations:
        return f"{'; '.join(violations)} — {text}"
    return f"satisfied: {text}"


def _shared_detail(model: ObjectiveModel, members: List[int], term_index: int,
                   violations: List[str]) -> str:
    rule_index = next(r for r, sr in enumerate(model.shared_rules) if sr.term == term_index)
    srule = model.shared_rules[rule_index]
    st = model.shared_stats(rule_index, members)
    attr = model.cohort.attributes.get(srule.attribute)
    names = attr.categories if attr is not None else []
    common = [names[c] for c in st["common"] if c < len(names)]
    if violations:
        return "; ".join(violations)
    shown = ", ".join(common[:4]) + (" …" if len(common) > 4 else "")
    return f"satisfied: {len(common)} in common ({shown})" if common else "satisfied"


def _fmt_num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else str(int(round(x)))


def _history_detail(model: ObjectiveModel, members: List[int]) -> str:
    history = model.history
    if history is None:
        return ""
    ids = model.cohort.student_ids
    rows = []
    ms = sorted(members)
    for a_pos, a in enumerate(ms):
        for b in ms[a_pos + 1:]:
            w = history.weights.get((a, b), 0.0)
            if w > 0:
                met = ", ".join(history.met_in.get((a, b), []))
                rows.append((w, f"{ids[a]} & {ids[b]} (met in {met})"))
    if not rows:
        return "no earlier contact between members"
    rows.sort(key=lambda r: -r[0])
    return f"{len(rows)} pair(s) met before: " + "; ".join(text for _, text in rows[:6])


def _preference_detail(model: ObjectiveModel, members: List[int]) -> str:
    member_set = set(members)
    prefer = sum(1 for i in members for j in model.cohort.prefer.get(i, []) if j in member_set)
    avoid = sum(1 for i in members for j in model.cohort.avoid.get(i, []) if j in member_set)
    return f"{prefer} preference(s) satisfied, {avoid} avoid(s) violated"


def student_impact(model: ObjectiveModel, assignment: Sequence[int], i: int) -> Dict[str, Any]:
    team = int(assignment[i])
    members = _members(model, assignment, team)
    base = model.team_breakdown(members)
    base_total = sum(base.values())
    others = [j for j in range(model.n) if assignment[j] != team]
    rest = [m for m in members if m != i]
    if others:
        acc = {label: 0.0 for label in base}
        totals = 0.0
        for j in others:
            bd = model.team_breakdown(rest + [j])
            for label, v in bd.items():
                acc[label] += v
            totals += sum(bd.values())
        mean_terms = {label: v / len(others) for label, v in acc.items()}
        contribution = base_total - totals / len(others)
    else:
        mean_terms = dict(base)
        contribution = 0.0
    terms = {label: round(base[label] - mean_terms[label], 6) for label in base}
    return {
        "student": model.cohort.student_ids[i],
        "index": i,
        "team": team,
        "team_penalty": round(base_total, 6),
        "contribution": round(contribution, 6),
        "removal_delta": round(base_total - model.team_penalty(rest), 6),
        "terms": dict(sorted(terms.items(), key=lambda kv: -kv[1])),
    }


def _term_changes(model: ObjectiveModel, before: Dict[int, List[int]],
                  after: Dict[int, List[int]]) -> List[Dict[str, Any]]:
    changes = []
    for t, old_members in before.items():
        old = model.team_breakdown(old_members)
        new = model.team_breakdown(after[t])
        for label in old:
            diff = new[label] - old[label]
            if abs(diff) > 1e-6:
                changes.append({"team": t, "term": label, "before": round(old[label], 6),
                                "after": round(new[label], 6), "change": round(diff, 6)})
    changes.sort(key=lambda c: c["change"])
    return changes


def _reason_text(change: Dict[str, Any], names: Optional[Sequence[str]]) -> str:
    t = change["team"]
    team = names[t] if names and t < len(names) else f"Team {t + 1}"
    sign = "−" if change["change"] < 0 else "+"
    return f"{team} · {change['term']} {sign}{abs(change['change']):.2f}"


def suggest_moves(model: ObjectiveModel, assignment: Sequence[int], *, k: int = 10,
                  locks: Optional[Dict[int, int]] = None,
                  focus_students: Optional[Iterable[int]] = None,
                  focus_team: Optional[int] = None, allow_moves: bool = False,
                  size_bounds: Optional[Tuple[int, int]] = None,
                  team_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    state = SearchState(model, list(assignment))
    locked = set((locks or {}).keys())
    focus = set(focus_students) if focus_students is not None else None
    free = [i for i in range(model.n) if i not in locked]
    assign = state.assignment
    found: List[Tuple[float, str, int, int]] = []
    for a_pos, i in enumerate(free):
        for j in free[a_pos + 1:]:
            ti, tj = assign[i], assign[j]
            if ti == tj:
                continue
            if focus is not None and i not in focus and j not in focus:
                continue
            if focus_team is not None and focus_team not in (ti, tj):
                continue
            d = state.delta_swap(i, j)
            if d < -_EPS:
                found.append((d, "swap", i, j))
    if allow_moves:
        sizes = state.sizes
        lo, hi = size_bounds if size_bounds is not None else (
            max(1, min(sizes) - 1), max(sizes) + 1)
        for i in free:
            if focus is not None and i not in focus:
                continue
            ti = assign[i]
            for t in range(model.n_teams):
                if t == ti or sizes[ti] <= lo or sizes[t] >= hi:
                    continue
                if focus_team is not None and focus_team not in (ti, t):
                    continue
                d = state.delta_move(i, t)
                if d < -_EPS:
                    found.append((d, "move", i, t))
    found.sort(key=lambda f: (f[0], f[2], f[3]))
    ids = model.cohort.student_ids
    out = []
    for d, kind, i, x in found[:max(0, int(k))]:
        ti = assign[i]
        if kind == "swap":
            tj = assign[x]
            before = {ti: list(state.members[ti]), tj: list(state.members[tj])}
            after = {ti: [m for m in before[ti] if m != i] + [x],
                     tj: [m for m in before[tj] if m != x] + [i]}
            students, indices, to_team = [ids[i], ids[x]], [i, x], tj
        else:
            before = {ti: list(state.members[ti]), x: list(state.members[x])}
            after = {ti: [m for m in before[ti] if m != i], x: before[x] + [i]}
            students, indices, to_team = [ids[i]], [i], x
        changes = _term_changes(model, before, after)
        out.append({
            "kind": kind,
            "students": students,
            "indices": indices,
            "from_team": ti,
            "to_team": to_team,
            "delta": round(d, 9),
            "objective_after": state.objective + d,
            "reasons": [_reason_text(c, team_names) for c in changes[:6]],
            "term_changes": changes,
        })
    return out


Change = Union[Dict[str, Any], Sequence[Any]]


def _resolve(model: ObjectiveModel, student: Any) -> int:
    if isinstance(student, int) and not isinstance(student, bool):
        if not 0 <= student < model.n:
            raise ValueError(f"Student index {student} is out of range")
        return student
    i = model.cohort.lookup(student)
    if i is None:
        raise ValueError(f"Unknown student {student!r}")
    return i


def _parse_change(model: ObjectiveModel, change: Change) -> Tuple[str, int, int]:
    if isinstance(change, dict):
        kind = str(change.get("kind", "swap"))
        students = list(change.get("students") or [])
        if "student" in change:
            students = [change["student"]] + students
        if kind == "swap":
            if len(students) != 2:
                raise ValueError("A swap needs exactly two students")
            return "swap", _resolve(model, students[0]), _resolve(model, students[1])
        if kind == "move":
            if len(students) != 1 or "to_team" not in change:
                raise ValueError("A move needs one student and 'to_team'")
            return "move", _resolve(model, students[0]), int(change["to_team"])
        raise ValueError(f"Unknown change kind '{kind}'")
    kind, a, b = change
    if kind == "swap":
        return "swap", _resolve(model, a), _resolve(model, b)
    if kind == "move":
        return "move", _resolve(model, a), int(b)
    raise ValueError(f"Unknown change kind '{kind}'")


def what_if(model: ObjectiveModel, assignment: Sequence[int], changes: Sequence[Change]
            ) -> Dict[str, Any]:
    before = model.team_penalties(assignment)
    objective_before = model.aggregate(before)
    new = list(assignment)
    for change in changes:
        kind, i, x = _parse_change(model, change)
        if kind == "swap":
            new[i], new[x] = new[x], new[i]
        else:
            if not 0 <= x < model.n_teams:
                raise ValueError(f"Team {x} is out of range")
            new[i] = x
    after = model.team_penalties(new)
    objective = model.aggregate(after)
    return {
        "objective_before": objective_before,
        "objective": objective,
        "delta": objective - objective_before,
        "team_penalties_before": before,
        "team_penalties": after,
        "hard_units": model.assignment_hard_units(new),
        "sizes": [sum(1 for t in new if t == team) for team in range(model.n_teams)],
        "assignment": new,
    }
