from __future__ import annotations

from app import extensions

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from .attributes import _fmt_num, redundant_pairs, tightest_feasible_max
from .feasibility import (
    check_feasibility,
    count_rule_floor,
    count_rule_groups,
    isolation_floor,
    value_bounds,
)
from .policy import (
    AttributeRule,
    CountRule,
    FairnessPolicy,
    HistoryPolicy,
    IsolationRule,
    Policy,
    PreferencePolicy,
    SharedValueRule,
    SolverSettings,
)
from .roster import AttributeInfo, infer_attributes, is_reserved
from .sizing import auto_iterations, plan_team_sizes

_GENERIC_ROLE_PATTERNS: List[Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]] = [
    ("score", ("score", "grade", "points"), ("mark", "marks"), ("numeric",)),
    ("schedule", ("availab", "schedule", "timezone", "time_zone", "timeslot", "time_slot",
                  "slot"), ("tz",), ("categorical", "multi")),
    ("language", ("language", "mother_tongue", "native_tongue"), (), ("categorical", "multi")),
    ("ordinal", ("experience", "skill", "proficien", "seniority"), (),
     ("categorical", "numeric")),
    ("score", ("score", "grade", "points"), ("mark", "marks"), ("categorical",)),
]
_ROLE_PATTERNS = extensions.hook("role_patterns", _GENERIC_ROLE_PATTERNS)

_CATEGORICAL_ROLE_RULES: Dict[str, Tuple[str, float, str]] = extensions.hook(
    "categorical_role_rules", {
        "ordinal": ("balance", 1.0, "recognised as an experience/skill scale: spread it evenly"),
        "schedule": ("cluster", 1.0,
                     "recognised as availability/schedule: teammates need overlapping time"),
        "language": ("balance", 1.0, "recognised as language: spread evenly"),
        "score": ("balance", 1.0, "recognised as a grade: spread evenly"),
    })

_NUMERIC_ROLE_RULES: Dict[str, Tuple[str, float, str]] = extensions.hook(
    "numeric_role_rules", {
        "score": ("balance", 2.0,
                  "recognised as a grade/score: equal team averages (fair start)"),
        "ordinal": ("balance", 1.0,
                    "recognised as an experience/skill score: equal team averages"),
    })

SCHEDULE_ROLES = ("schedule", "language")
CATME_ISOLATION_WEIGHT = 15.0


def _tokens(name: str) -> List[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def attribute_role(name: str, info: Optional[AttributeInfo] = None) -> str:
    kind = info.kind if info is not None else None
    if kind in ("text", "constant", "empty"):
        kind = "categorical"
    lowered = str(name).lower().replace(" ", "_").replace("-", "_")
    tokens = set(_tokens(name))
    for role, substrings, words, kinds in _ROLE_PATTERNS:
        if kind is not None and kind not in kinds:
            continue
        if any(s in lowered for s in substrings) or tokens & set(words):
            return role
    return "generic"


@dataclass
class AttributeSuggestion:
    attribute: str
    kind: str
    role: str
    goal: str
    weight: float
    why: str
    redundant_with: Optional[str] = None


def _base_suggestion(name: str, info: AttributeInfo) -> AttributeSuggestion:
    role = attribute_role(name, info)
    flag = getattr(info, "flag", "")
    if flag == "identifier":
        return AttributeSuggestion(name, info.kind, role, "ignore", 0.0,
                                   f"identifier-like: a different value for (almost) every "
                                   f"student ({info.n_unique} distinct), e.g. ids, names or "
                                   "e-mails")
    if flag == "date":
        return AttributeSuggestion(name, info.kind, role, "ignore", 0.0,
                                   "dates / timestamps: not used by default (set a goal to "
                                   "use them as categories)")
    if info.kind == "text":
        return AttributeSuggestion(name, info.kind, role, "ignore", 0.0,
                                   f"free text ({info.n_unique} distinct values): "
                                   "too many categories to balance")
    if info.kind == "constant":
        value = info.values[0] if info.values else ""
        return AttributeSuggestion(name, info.kind, role, "ignore", 0.0,
                                   f"everyone has the same value ('{value}')")
    if info.kind == "empty":
        return AttributeSuggestion(name, info.kind, role, "ignore", 0.0, "no values")
    if info.kind == "numeric":
        goal, weight, why = _NUMERIC_ROLE_RULES.get(
            role, ("balance", 1.0, "numeric: equal team averages"))
        bad = int(info.stats.get("non_numeric", 0) or 0)
        if bad:
            shown = ", ".join(repr(v) for v in info.non_numeric[:3])
            why += f" ({bad} non-numeric cell{'s' if bad != 1 else ''} such as {shown} " \
                   "count as missing)"
        return AttributeSuggestion(name, info.kind, role, goal, weight, why)
    if role in _CATEGORICAL_ROLE_RULES:
        goal, weight, why = _CATEGORICAL_ROLE_RULES[role]
        return AttributeSuggestion(name, info.kind, role, goal, weight, why)
    if info.kind == "multi":
        return AttributeSuggestion(name, info.kind, role, "diversify", 1.0,
                                   "multi-select: avoid teammates sharing the same items")
    if info.n_unique <= 12:
        return AttributeSuggestion(name, info.kind, role, "balance", 1.0,
                                   f"categorical with {info.n_unique} values: mirror the cohort mix")
    return AttributeSuggestion(name, info.kind, role, "diversify", 0.5,
                               f"categorical with {info.n_unique} values: avoid duplicates")


_KEEP_PRIORITY: Dict[str, int] = extensions.hook(
    "keep_priority", {"score": 1, "ordinal": 3, "generic": 4})


def suggest_attribute_rules(frame: pd.DataFrame,
                            infos: Optional[Dict[str, AttributeInfo]] = None, *,
                            team_size: int = 5,
                            redundancy: Optional[List[Dict[str, Any]]] = None,
                            ) -> Dict[str, AttributeSuggestion]:
    infos = infos or infer_attributes(frame)
    out: Dict[str, AttributeSuggestion] = {
        name: _base_suggestion(name, info)
        for name, info in infos.items() if not is_reserved(name)
    }
    if redundancy is None:
        redundancy = redundant_pairs(frame, infos=infos)

    def active(name: str) -> bool:
        s = out.get(name)
        return s is not None and s.goal != "ignore"

    def switch_off(name: str, keep: str, why: str) -> None:
        s = out[name]
        out[name] = AttributeSuggestion(name, s.kind, s.role, "ignore", 0.0, why,
                                        redundant_with=keep)

    for pair in redundancy:
        a, b, relation = pair["a"], pair["b"], pair["relation"]
        if not (active(a) and active(b)):
            continue
        if relation == "same":
            if out[a].kind == "numeric" and out[b].kind == "numeric":
                rank_a = _KEEP_PRIORITY.get(out[a].role, 4)
                rank_b = _KEEP_PRIORITY.get(out[b].role, 4)
                keep, drop = (a, b) if rank_a <= rank_b else (b, a)
            else:
                keep, drop = (a, b) if out[a].weight >= out[b].weight else (b, a)
            switch_off(drop, keep, f"same information as {keep} ({pair['explanation']})")
        elif relation == "binning":
            switch_off(b, a, f"{pair['explanation']}; {a} is used directly")
        elif (relation == "determines" and pair["strength"] >= 0.99
              and out[a].role == out[b].role):
            switch_off(b, a, f"derived from {a} ({pair['explanation']}); "
                             f"the finer {a} is used instead")
    return out


def _first_with_role(suggestions: Dict[str, AttributeSuggestion], role: str,
                     kinds: Sequence[str] = ("categorical",)) -> Optional[AttributeSuggestion]:
    for s in suggestions.values():
        if s.role == role and s.goal != "ignore" and s.kind in kinds:
            return s
    return None


def _cohort_size(frame: Optional[pd.DataFrame], infos: Dict[str, AttributeInfo]) -> int:
    if frame is not None:
        return len(frame)
    for info in infos.values():
        if info.kind != "multi" and info.coverage > 0 and info.counts:
            return int(round(sum(info.counts.values()) / info.coverage))
    return 0


def suggest_policy(frame: pd.DataFrame, infos: Optional[Dict[str, AttributeInfo]] = None, *,
                   preset: str = "balanced", team_size: int = 5,
                   num_teams: Optional[int] = None) -> Policy:
    infos = infos or infer_attributes(frame)
    n = len(frame)
    sizes = plan_team_sizes(n, team_size, team_size, team_size, num_teams or 0)
    teams = len(sizes)
    suggestions = suggest_attribute_rules(frame, infos, team_size=team_size)

    attributes = [AttributeRule(attribute=s.attribute, goal=s.goal, weight=s.weight)
                  for s in suggestions.values()]
    counts: List[CountRule] = []
    isolation: List[IsolationRule] = []
    notes: List[str] = []

    for s in suggestions.values():
        if s.goal == "ignore":
            notes.append(f"{s.attribute}: not used — {s.why}")
        else:
            notes.append(f"{s.attribute}: {s.goal} (weight {_fmt_num(s.weight)}) — {s.why}")

    extras = extensions.hook("suggest_extras", None)
    if extras is not None:
        extras(suggestions, infos, n=n, sizes=sizes, teams=teams, team_size=team_size,
               counts=counts, isolation=isolation, notes=notes)

    policy = Policy(
        name="Suggested — Balanced",
        description="",
        attributes=attributes,
        counts=counts,
        isolation=isolation,
        history=HistoryPolicy(enabled=True, repeat_weight=3.0, max_times_together=1,
                              hard_cap=True, contact_model="team_size",
                              reference_team_size=int(team_size)),
        preferences=PreferencePolicy(use_student_preferences=True),
        fairness=FairnessPolicy(worst_team_weight=0.25),
        solver=SolverSettings(),
    )
    header = (f"Suggested from the data ({n} students, {teams} teams of about {team_size}). "
              "Inferred:")
    policy.description = "\n".join([header, *[f"- {line}" for line in notes]])
    if preset and preset != "balanced":
        policy = apply_preset(policy, preset, frame, infos=infos, team_size=team_size,
                              num_teams=num_teams)
    return policy


Transform = Callable[..., Policy]


class Preset(NamedTuple):
    title: str
    description: str
    transform: Transform


def _infos_kind(policy_rule_kind: Optional[str], info: Optional[AttributeInfo]) -> Optional[str]:
    if policy_rule_kind:
        return policy_rule_kind
    return info.kind if info is not None else None


def _sizes_from(infos: Dict[str, AttributeInfo], frame: Optional[pd.DataFrame],
                team_size: int, num_teams: Optional[int]) -> List[int]:
    n = _cohort_size(frame, infos)
    return plan_team_sizes(n, team_size, team_size, team_size, num_teams or 0) if n else []


def _count_rule_floor_from_infos(rule: CountRule, infos: Dict[str, AttributeInfo],
                                 sizes: List[int]) -> Optional[int]:
    info = infos.get(rule.attribute)
    if info is None or not sizes:
        return None
    numeric_values = None
    if rule.numeric_range is not None:
        numeric_values = []
        for value, count in info.counts.items():
            try:
                numeric_values.extend([float(value)] * count)
            except ValueError:
                continue
    groups = count_rule_groups(rule, info.counts, numeric_values)
    units, _ = count_rule_floor(rule, groups, sizes)
    return units


def _isolation_value_ok(attribute: str, value: str, min_together: int,
                        policy: Policy, infos: Dict[str, AttributeInfo],
                        sizes: List[int]) -> bool:
    info = infos.get(attribute)
    if info is None or not sizes:
        return False
    matching = info.counts.get(value, 0)
    if matching == 0:
        return True
    bounds = value_bounds(policy, attribute, value)
    caps = [s if bounds.cap is None else min(s, bounds.cap) for s in sizes]
    if isolation_floor(matching, min_together, caps) > 0:
        return False
    per_team = max(bounds.floor, min_together)
    return not (bounds.floor >= 1 and sum(min(per_team, s) for s in sizes) > matching)


def _balanced(policy: Policy, infos: Dict[str, AttributeInfo], **_: Any) -> Policy:
    policy.fairness.worst_team_weight = 0.25
    return policy


def _max_mix(policy: Policy, infos: Dict[str, AttributeInfo], *, frame=None,
             team_size: int = 5, num_teams: Optional[int] = None, **_: Any) -> Policy:
    sizes = _sizes_from(infos, frame, team_size, num_teams)
    for rule in policy.attributes:
        kind = _infos_kind(rule.kind, infos.get(rule.attribute))
        if rule.enabled and rule.goal == "balance" and kind in ("categorical", "multi"):
            rule.goal = "diversify"
    for rule in policy.counts:
        if not rule.enabled or rule.max_count is None or rule.numeric_range is not None:
            continue
        info = infos.get(rule.attribute)
        if info is None or not sizes:
            continue
        values = rule.values or list(info.counts)
        if rule.mode == "each":
            tightest = max((tightest_feasible_max(info.counts.get(v, 0), sizes) for v in values),
                           default=0)
        else:
            tightest = tightest_feasible_max(sum(info.counts.get(v, 0) for v in values), sizes)
        rule.max_count = max(tightest, rule.max_count - 1)
        if rule.label.startswith("At most"):
            rule.label = re.sub(r"At most \d+", f"At most {rule.max_count}", rule.label)
    extras = extensions.hook("max_mix_extras", None)
    if extras is not None:
        extras(policy, infos, sizes)
    policy.fairness.worst_team_weight = 0.4
    return policy


def _catme(policy: Policy, infos: Dict[str, AttributeInfo], *, frame=None,
           team_size: int = 5, num_teams: Optional[int] = None, **_: Any) -> Policy:
    sizes = _sizes_from(infos, frame, team_size, num_teams)
    policy.fairness.worst_team_weight = 0.7
    extras = extensions.hook("catme_extras", None)
    if extras is not None:
        extras(policy, infos, sizes)

    for rule in policy.isolation:
        if not rule.enabled or rule.hard:
            continue
        info = infos.get(rule.attribute)
        values = rule.values or (list(info.counts) if info else [])
        if values and all(_isolation_value_ok(rule.attribute, v, rule.min_together,
                                              policy, infos, sizes) for v in values):
            rule.weight = max(rule.weight, CATME_ISOLATION_WEIGHT)
    return policy


def _drop_minimum_values(policy: Policy, rule: CountRule, name: str,
                         dropped: List[str]) -> None:
    rule.values = [v for v in rule.values if v not in dropped]
    if not rule.values:
        rule.enabled = False
    elif len(rule.values) == 1:
        rule.mode = "any_of"
        rule.label = f"At least {rule.min_count} {name}={rule.values[0]}"
    else:
        rule.label = f"At least {rule.min_count} of each {name} ({', '.join(rule.values)})"
    kept = (f"at least {rule.min_count} of {', '.join(rule.values)} per team"
            if rule.values else "no per-team minimum")
    note = (f"- {name}: {kept}; teams without {', '.join(dropped)} are accepted (too few "
            f"students to give every team two of them)")
    lines = policy.description.splitlines() if policy.description else []
    prefix = f"- {name}: at least"
    lines = [note if line.startswith(prefix) else line for line in lines]
    if note not in lines:
        lines.append(note)
    policy.description = "\n".join(lines)


def _strict_quotas(policy: Policy, infos: Dict[str, AttributeInfo], *, frame=None,
                   team_size: int = 5, num_teams: Optional[int] = None, **_: Any) -> Policy:
    sizes = _sizes_from(infos, frame, team_size, num_teams)
    converted: List[int] = []
    for index, rule in enumerate(policy.counts):
        if not rule.enabled or rule.hard:
            continue
        if _count_rule_floor_from_infos(rule, infos, sizes) == 0:
            rule.hard = True
            converted.append(index)
    if frame is not None and converted and sizes:
        project = {"project_id": "all", "student_ids": None, "team_sizes": sizes}
        for _round in range(3):
            report = check_feasibility(frame, policy, [project], probe_time_s=0.0)
            blamed = {f.detail.get("rule_index") for f in report.errors
                      if f.detail.get("rule_type") == "count"}
            blamed |= {f.detail.get("count_rule_index") for f in report.errors
                       if f.detail.get("rule_type") == "isolation"}
            revert = [i for i in converted if i in blamed]
            if not revert:
                break
            for i in revert:
                policy.counts[i].hard = False
                converted.remove(i)
    return policy


SCHEDULE_SHARED_MIN = 1
SCHEDULE_SHARED_WEIGHT = 5.0


def _similar_schedules(policy: Policy, infos: Dict[str, AttributeInfo], **_: Any) -> Policy:
    shared_on = {r.attribute for r in policy.shared}
    for rule in policy.attributes:
        info = infos.get(rule.attribute)
        kind = _infos_kind(rule.kind, info)
        role = attribute_role(rule.attribute, info)
        if role in SCHEDULE_ROLES and kind in ("categorical", "multi"):
            rule.goal = "cluster"
            rule.weight = max(rule.weight, 3.0)
            rule.enabled = True
            if role == "schedule" and kind == "multi" and rule.attribute not in shared_on:
                policy.shared.append(SharedValueRule(
                    attribute=rule.attribute, min_shared=SCHEDULE_SHARED_MIN, hard=False,
                    weight=SCHEDULE_SHARED_WEIGHT,
                    label=f"Common {rule.attribute} slot"))
                shared_on.add(rule.attribute)
        elif rule.goal == "diversify":
            rule.goal = "balance"
        elif rule.goal == "spread":
            rule.goal = "balance"
    return policy


def _quick(policy: Policy, infos: Dict[str, AttributeInfo], **_: Any) -> Policy:
    policy.solver.time_limit_s = 3.0
    policy.solver.restarts = 2
    return policy


THOROUGH_RESTARTS = 12
THOROUGH_BUDGET_FACTOR = 3


def _thorough(policy: Policy, infos: Dict[str, AttributeInfo], *, frame=None,
              team_size: int = 5, num_teams: Optional[int] = None, **_: Any) -> Policy:
    sizes = _sizes_from(infos, frame, team_size, num_teams)
    n = sum(sizes)
    policy.solver.method = "anneal"
    policy.solver.time_limit_s = 60.0
    policy.solver.restarts = THOROUGH_RESTARTS
    budget = auto_iterations(n, len(sizes)) if sizes else 0
    policy.solver.iterations = THOROUGH_BUDGET_FACTOR * budget if budget else None
    policy.solver.workers = max(1, min(4, os.cpu_count() or 1))
    return policy


def _preset_description(key: str, neutral: str) -> str:
    return extensions.hook("preset_descriptions", {}).get(key, neutral)


PRESETS: Dict[str, Preset] = {
    "balanced": Preset(
        "Balanced",
        _preset_description('balanced', 'Every team mirrors the roster on the recognised attributes, with a 25% emphasis on the worst team; head-count rules of the rule set are kept as they are.'),
        _balanced),
    "max_mix": Preset(
        "Maximum mix",
        _preset_description('max_mix', 'Diversify every categorical attribute (as few same-value teammates as possible), lower each head-count maximum of the rule set by one where the data allows it and put 40% weight on the worst team.'),
        _max_mix),
    "catme": Preset(
        "CATME-style",
        _preset_description('catme', 'Maximin emphasis (70% on the worst team); soft "no isolated member" rules of the rule set get a heavy weight (15) where the data allows it.'),
        _catme),
    "strict_quotas": Preset(
        "Strict quotas",
        "Turn every head-count rule the data can satisfy into a hard quota; rules that "
        "cannot be met stay soft.",
        _strict_quotas),
    "similar_schedules": Preset(
        "Similar schedules",
        "Group students with the same availability, schedule, time zone or language "
        "(multi-slot availability also gets a soft 'at least one common slot' rule); "
        "balance everything else.",
        _similar_schedules),
    "quick": Preset(
        "Quick",
        "Fast preview: 3 seconds and 2 restarts per project.",
        _quick),
    "thorough": Preset(
        "Thorough",
        "For the final allocation: annealing with 12 restarts and a 3× larger move budget, "
        "on up to 4 CPU cores (results do not depend on the core count); 60 s cap per "
        "project.",
        _thorough),
}


def apply_preset(policy: Policy, name: str, frame: Optional[pd.DataFrame] = None, *,
                 infos: Optional[Dict[str, AttributeInfo]] = None,
                 team_size: Optional[int] = None,
                 num_teams: Optional[int] = None) -> Policy:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose one of {', '.join(PRESETS)}")
    preset = PRESETS[name]
    copy = Policy.from_dict(policy.to_dict())
    if infos is None:
        infos = infer_attributes(frame) if frame is not None else {}
    team_size = int(team_size or copy.history.reference_team_size or 5)
    result = preset.transform(copy, infos, frame=frame, team_size=team_size,
                              num_teams=num_teams)
    base = policy.name or "Custom"
    if base.startswith("Suggested"):
        result.name = f"Suggested — {preset.title}"
    elif preset.title not in base:
        result.name = f"{base} + {preset.title}"
    lines = [f"Preset '{preset.title}': {preset.description}"]
    if result.description:
        lines.append(result.description)
    result.description = "\n".join(lines)
    return result
