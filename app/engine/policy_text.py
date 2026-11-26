from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .attributes import _fmt_num
from .policy import (
    AttributeRule,
    CountRule,
    HistoryPolicy,
    IsolationRule,
    NumericBoundsRule,
    PairRule,
    Policy,
    SharedValueRule,
)


def _hardness(hard: bool, weight: float) -> str:
    return " (hard)" if hard else f" — soft, weight {_fmt_num(weight)}"


def _id_list(ids: Sequence[str], limit: int = 5) -> str:
    ids = [str(i) for i in ids]
    text = ", ".join(ids[:limit])
    if len(ids) > limit:
        text += f" +{len(ids) - limit} more"
    return text


def _range_text(attribute: str, rng: Tuple[Optional[float], Optional[float]]) -> str:
    lo, hi = (list(rng) + [None, None])[:2]
    if lo is not None and hi is not None:
        return f"{attribute} between {_fmt_num(lo)} and {_fmt_num(hi)}"
    if lo is not None:
        return f"{attribute} ≥ {_fmt_num(lo)}"
    if hi is not None:
        return f"{attribute} ≤ {_fmt_num(hi)}"
    return f"any {attribute}"


def describe_attribute_rule(rule: AttributeRule, kind: Optional[str] = None) -> str:
    kind = rule.kind or kind
    attr = rule.attribute
    w = f" (weight {_fmt_num(rule.weight)})"
    if rule.goal == "ignore" or not rule.enabled or rule.weight <= 0:
        return f"Ignore {attr}"
    if rule.goal == "spread":
        return f"Spread {attr}: equal team averages and a cohort-like range in every team{w}"
    if kind == "numeric":
        if rule.goal == "balance":
            return f"Balance average {attr} across teams{w}"
        return f"Group students with similar {attr}{w}"
    if rule.goal == "balance":
        return f"Balance {attr} across teams{w}"
    if rule.goal == "diversify":
        return f"Diversify {attr}: as few teammates sharing a value as possible{w}"
    if rule.goal == "cluster":
        return f"Group students with the same {attr}{w}"
    return f"{rule.goal.capitalize()} {attr}{w}"


def _count_headline(rule: CountRule) -> str:
    lo, hi = rule.min_count, rule.max_count
    if lo is not None and hi is not None:
        bound = f"Exactly {lo}" if lo == hi else f"Between {lo} and {hi}"
        number = hi
    elif hi is not None:
        bound, number = f"At most {hi}", hi
    elif lo is not None:
        bound, number = f"At least {lo}", lo
    else:
        bound, number = "Any number of", 2
    students = "student" if number == 1 else "students"
    attr = rule.attribute
    if rule.numeric_range is not None:
        subject = f"{students} with {_range_text(attr, rule.numeric_range)}"
    elif rule.mode == "each" and not rule.values:
        subject = f"of any one {attr}" if hi is not None and lo is None else f"of every {attr}"
    elif rule.mode == "each":
        subject = f"of each {attr} ({', '.join(rule.values)})"
    elif len(rule.values) == 1:
        subject = f"{students} with {attr} = {rule.values[0]}"
    else:
        subject = f"{students} with {attr} in {{{', '.join(rule.values)}}}"
    return f"{bound} {subject} per team"


def describe_count_rule(rule: CountRule) -> str:
    return _count_headline(rule) + _hardness(rule.hard, rule.weight)


def _isolation_headline(rule: IsolationRule) -> str:
    attr = rule.attribute
    if rule.values:
        subject = f"{attr}={' or '.join(rule.values)}"
    else:
        subject = f"member of any {attr} group"
    if rule.min_together <= 2:
        return f"No lone {subject}"
    return f"Teams with any {subject} have at least {rule.min_together} of them"


def describe_isolation_rule(rule: IsolationRule) -> str:
    return _isolation_headline(rule) + _hardness(rule.hard, rule.weight)


def _numeric_bounds_headline(rule: NumericBoundsRule) -> str:
    attr = rule.attribute
    parts = []
    lo, hi = rule.mean_min, rule.mean_max
    if lo is not None and hi is not None:
        parts.append(f"average {attr} between {_fmt_num(lo)} and {_fmt_num(hi)}")
    elif lo is not None:
        parts.append(f"average {attr} at least {_fmt_num(lo)}")
    elif hi is not None:
        parts.append(f"average {attr} at most {_fmt_num(hi)}")
    if rule.max_range is not None:
        parts.append(f"{attr} range at most {_fmt_num(rule.max_range)}")
    body = " and ".join(parts) if parts else f"no bounds on {attr}"
    text = f"Team {body} per team"
    if rule.label:
        text = f"{rule.label}: {text[0].lower()}{text[1:]}"
    return text


def describe_numeric_bounds_rule(rule: NumericBoundsRule) -> str:
    return _numeric_bounds_headline(rule) + _hardness(rule.hard, rule.weight)


def _shared_headline(rule: SharedValueRule, kind: Optional[str] = None) -> str:
    attr = rule.attribute
    if kind == "categorical":
        text = f"Teammates all have the same {attr}"
    else:
        k = int(rule.min_shared)
        text = f"Teammates share at least {k} {attr} value{'s' if k != 1 else ''}"
    if rule.label:
        text = f"{rule.label}: {text[0].lower()}{text[1:]}"
    return text


def describe_shared_rule(rule: SharedValueRule, kind: Optional[str] = None) -> str:
    return _shared_headline(rule, kind) + _hardness(rule.hard, rule.weight)


def _pair_headline(rule: PairRule) -> str:
    ids = _id_list(rule.students)
    text = f"Keep {ids} together" if rule.kind == "together" else f"Keep {ids} in different teams"
    if rule.label:
        text = f"{rule.label}: {text[0].lower()}{text[1:]}"
    if rule.projects:
        text += f" (project{'s' if len(rule.projects) > 1 else ''} {', '.join(map(str, rule.projects))} only)"
    return text


def describe_pair_rule(rule: PairRule) -> str:
    return _pair_headline(rule) + _hardness(rule.hard, rule.weight)


def rule_headline(rule: Any) -> str:
    if isinstance(rule, CountRule):
        return _count_headline(rule)
    if isinstance(rule, IsolationRule):
        return _isolation_headline(rule)
    if isinstance(rule, NumericBoundsRule):
        return _numeric_bounds_headline(rule)
    if isinstance(rule, SharedValueRule):
        return _shared_headline(rule)
    if isinstance(rule, PairRule):
        return _pair_headline(rule)
    raise TypeError(f"no headline for {type(rule).__name__}")


def describe_history(history: HistoryPolicy) -> List[str]:
    if not history.enabled:
        return ["Ignore earlier projects (repeat teammates allowed)"]
    lines = []
    cap = history.max_times_together
    if cap is not None:
        if cap == 1:
            text = "Never repeat a teammate from earlier projects"
        else:
            text = f"Pairs may share at most {cap} teams across all projects"
        if history.hard_cap:
            text += " (hard)"
        else:
            text += f" — soft, weight {_fmt_num(history.cap_weight)} per extra pair"
        lines.append(text)
    if history.repeat_weight > 0:
        if history.contact_model == "team_size":
            model = f"contact scaled to teams of {history.reference_team_size}"
        else:
            model = "every shared group counts fully"
        extras = [model]
        if history.decay < 1:
            extras.append(f"each older project counts ×{_fmt_num(history.decay)}")
        if cap is not None and history.ignore_groups_larger_than is not None:
            extras.append(f"groups larger than {history.ignore_groups_larger_than} "
                          "only count here, not toward the cap")
        lines.append(f"Discourage pairing students who met before "
                     f"(weight {_fmt_num(history.repeat_weight)}; {'; '.join(extras)})")
    return lines


def describe_policy(policy: Policy, kinds: Optional[Dict[str, str]] = None) -> List[str]:
    if isinstance(policy, dict):
        policy = Policy.from_dict(policy)
    kinds = kinds or {}
    lines: List[str] = []
    ignored = []
    for rule in policy.attributes:
        if not rule.enabled or rule.goal == "ignore" or rule.weight <= 0:
            ignored.append(rule.attribute)
            continue
        lines.append(describe_attribute_rule(rule, kinds.get(rule.attribute)))
    lines += [describe_count_rule(r) for r in policy.counts if r.enabled]
    lines += [describe_isolation_rule(r) for r in policy.isolation if r.enabled]
    lines += [describe_numeric_bounds_rule(r) for r in policy.numeric_bounds if r.enabled]
    lines += [describe_shared_rule(r, kinds.get(r.attribute)) for r in policy.shared if r.enabled]
    lines += [describe_pair_rule(r) for r in policy.pairs if r.enabled]
    lines += describe_history(policy.history)
    p = policy.preferences
    if p.use_student_preferences:
        avoid = "hard" if p.avoid_hard else f"soft, weight {_fmt_num(p.avoid_weight)}"
        lines.append(f"Honour student teammate requests (weight {_fmt_num(p.prefer_weight)} "
                     f"each); avoid requests are {avoid}")
    else:
        lines.append("Ignore student teammate requests")
    w = policy.fairness.worst_team_weight
    if w <= 0:
        lines.append("Optimise the total over all teams only")
    elif w >= 1:
        lines.append("Make the worst team as good as possible (pure maximin)")
    else:
        lines.append(f"Fairness: {w:.0%} weight on the worst team, {1 - w:.0%} on the total")
    s = policy.solver
    grows = " (more for cohorts over 200)" if s.scale_time_limit else ""
    solver = (f"Solver: {s.method}, {_fmt_num(s.time_limit_s)} s per project{grows}, "
              f"{s.restarts} restart{'s' if s.restarts != 1 else ''}")
    if s.iterations is not None:
        solver += f", {s.iterations:,} moves per restart"
    if s.seed is not None:
        solver += f", seed {s.seed}"
    if s.flexible_sizes:
        solver += ", flexible team sizes"
    if s.joint_projects:
        solver += ", projects optimised jointly"
    if s.workers > 1:
        solver += f", {s.workers} parallel workers"
    lines.append(solver)
    if ignored:
        lines.append(f"Not used: {', '.join(ignored)}")
    return lines


_SECTION_LABELS = {
    "history": "History",
    "preferences": "Preferences",
    "fairness": "Fairness",
    "solver": "Solver",
}

_FIELD_LABELS = {
    "time_limit_s": "time limit (s)",
    "mip_gap": "MIP gap",
    "worst_team_weight": "worst-team weight",
    "min_count": "minimum",
    "max_count": "maximum",
    "min_together": "min together",
    "ignore_groups_larger_than": "ignore groups larger than",
    "mean_min": "minimum mean",
    "mean_max": "maximum mean",
    "max_range": "maximum range",
    "min_shared": "min shared",
}


def _value_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return _fmt_num(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value_text(v) for v in value) + "]"
    return str(value)


def _field_changes(prefix: str, a: Any, b: Any, skip: Sequence[str] = ()) -> List[str]:
    out = []
    for f in fields(a):
        if f.name in skip:
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, tuple):
            va = list(va)
        if isinstance(vb, tuple):
            vb = list(vb)
        if va != vb:
            label = _FIELD_LABELS.get(f.name, f.name.replace("_", " "))
            out.append(f"{prefix}{label}: {_value_text(va)} → {_value_text(vb)}")
    return out


def _keyed(rules: Sequence[Any], key: Callable[[Any], Tuple]) -> Dict[Tuple, Any]:
    out: Dict[Tuple, Any] = {}
    for rule in rules:
        k = key(rule)
        while k in out:
            k = k + ("dup",)
        out[k] = rule
    return out


def _diff_rules(a_rules, b_rules, key, describe, skip) -> List[str]:
    a_map, b_map = _keyed(a_rules, key), _keyed(b_rules, key)
    out = []
    for k, rule in a_map.items():
        if k not in b_map:
            out.append(f"Removed: {describe(rule)}")
    for k, rule in b_map.items():
        if k not in a_map:
            out.append(f"Added: {describe(rule)}")
        else:
            changes = _field_changes("", a_map[k], rule, skip)
            if changes:
                out.append(f"{rule_headline(a_map[k])}: " + "; ".join(changes))
    return out


def policy_diff(a: Policy, b: Policy) -> List[str]:
    if isinstance(a, dict):
        a = Policy.from_dict(a)
    if isinstance(b, dict):
        b = Policy.from_dict(b)
    out: List[str] = []
    if a.name != b.name:
        out.append(f"Name: {a.name} → {b.name}")

    a_attrs = {r.attribute: r for r in a.attributes}
    b_attrs = {r.attribute: r for r in b.attributes}
    for name, rule in a_attrs.items():
        if name not in b_attrs:
            out.append(f"{name}: rule removed (was {rule.goal}, weight {_fmt_num(rule.weight)})")
    for name, rule in b_attrs.items():
        if name not in a_attrs:
            out.append(f"{name}: rule added — {describe_attribute_rule(rule)}")
            continue
        old = a_attrs[name]
        changes = _field_changes("", old, rule, skip=("attribute",))
        if changes:
            out.append(f"{name}: " + "; ".join(changes))

    out += _diff_rules(
        a.counts, b.counts,
        key=lambda r: (r.attribute, tuple(r.values), r.mode,
                       tuple(r.numeric_range) if r.numeric_range else None),
        describe=describe_count_rule, skip=("attribute", "values", "mode", "numeric_range"))
    out += _diff_rules(
        a.isolation, b.isolation,
        key=lambda r: (r.attribute, tuple(r.values)),
        describe=describe_isolation_rule, skip=("attribute", "values"))
    out += _diff_rules(
        a.numeric_bounds, b.numeric_bounds,
        key=lambda r: (r.attribute,),
        describe=describe_numeric_bounds_rule, skip=("attribute",))
    out += _diff_rules(
        a.shared, b.shared,
        key=lambda r: (r.attribute,),
        describe=describe_shared_rule, skip=("attribute",))
    out += _diff_rules(
        a.pairs, b.pairs,
        key=lambda r: (r.kind, tuple(sorted(map(str, r.students))),
                       tuple(r.projects) if r.projects else None),
        describe=describe_pair_rule, skip=("kind", "students", "projects"))

    for section, label in _SECTION_LABELS.items():
        out += _field_changes(f"{label} ", getattr(a, section), getattr(b, section))
    return out
