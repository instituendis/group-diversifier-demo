from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .feasibility import count_rule_floor, isolation_floor
from .objective import ObjectiveModel

_EPS = 1e-12


def _num(x: float, digits: int = 6) -> float:
    value = float(x)
    if math.isnan(value) or math.isinf(value):
        return value
    return round(value, digits)


def _units(x: float) -> Any:
    value = float(x)
    return int(round(value)) if abs(value - round(value)) < 1e-9 else round(value, 3)


def _fmt_value(x: Optional[float]) -> str:
    if x is None:
        return "–"
    value = float(x)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def violation_summary(units: Any, unavoidable: Any = 0) -> str:
    units = float(units or 0)
    unavoidable = min(float(unavoidable or 0), units)
    if units <= 1e-9:
        return "no violations"
    count = _units(units)
    word = "violation" if count == 1 else "violations"
    if unavoidable <= 1e-9:
        return f"{count} {word}"
    if unavoidable >= units - 1e-9:
        return f"{count} {word} (all unavoidable)"
    return f"{count} {word} ({_units(unavoidable)} unavoidable)"


def default_team_names(n_teams: int) -> List[str]:
    return [f"Team {t + 1}" for t in range(n_teams)]


def random_allocations(sizes: Sequence[int], n: int, samples: int, seed: int = 0,
                       locks: Optional[Dict[int, int]] = None) -> np.ndarray:
    locks = dict(locks or {})
    slots: List[int] = []
    remaining = list(sizes)
    for t in locks.values():
        remaining[t] -= 1
    for t, k in enumerate(remaining):
        slots.extend([t] * max(0, k))
    free = np.array([i for i in range(n) if i not in locks], dtype=np.int64)
    slot_arr = np.array(slots[:len(free)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = np.empty((samples, n), dtype=np.int64)
    for i, t in locks.items():
        out[:, i] = t
    if len(free):
        keys = rng.random((samples, len(free)))
        perms = np.argsort(keys, axis=1)
        out[:, free] = slot_arr[perms]
    return out


def random_baseline(model: ObjectiveModel, objective: float, *, sizes: Sequence[int],
                    samples: int = 200, seed: int = 0,
                    locks: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    if samples <= 0 or model.n == 0 or not sizes or sum(sizes) != model.n:
        return {"mean": None, "std": None, "best": None, "percentile_of_solution": None,
                "samples": 0}
    ids = model.cohort.student_ids
    order = sorted(range(model.n), key=lambda i: (ids[i], i))
    canonical = random_allocations(sorted(sizes, reverse=True), model.n, samples, seed)
    allocations = np.empty_like(canonical)
    allocations[:, order] = canonical
    values = np.array([model.evaluate(row.tolist()) for row in allocations], dtype=float)
    return {
        "mean": _num(values.mean()),
        "std": _num(values.std()),
        "best": _num(values.min()),
        "percentile_of_solution": _num(100.0 * float(np.mean(values < objective - 1e-9)), 2),
        "samples": int(samples),
    }


OUTLIER_FACTOR = 3.0
OUTLIER_MIN_POINTS = 1.0


def _quality(objective: float, random_mean: Optional[float]) -> float:
    if random_mean is None or abs(random_mean) < _EPS:
        return 0.0
    value = 100.0 * (random_mean - objective) / abs(random_mean)
    return max(-100.0, min(100.0, value))


def team_violations(model: ObjectiveModel, members: Sequence[int]) -> List[Dict[str, Any]]:
    ids = model.cohort.student_ids
    members = list(members)
    member_set = set(members)
    out: List[Dict[str, Any]] = []
    for r, grule in enumerate(model.group_rules):
        counts = [0] * grule.ngroups
        holders: List[List[int]] = [[] for _ in range(grule.ngroups)]
        for i in members:
            for g in model.group_codes[i][r]:
                counts[g] += 1
                holders[g].append(i)
        for g, n_g in enumerate(counts):
            text = None
            if grule.family == "count":
                if grule.max_count is not None and n_g > grule.max_count:
                    text = f"{n_g} × {grule.group_labels[g]} (max {grule.max_count})"
                elif grule.min_count is not None and n_g < grule.min_count:
                    text = f"{n_g} × {grule.group_labels[g]} (min {grule.min_count})"
            elif grule.family == "isolation":
                k = grule.min_together or 2
                if 0 < n_g < k:
                    text = (f"lone {grule.group_labels[g]} (needs ≥ {k})" if n_g == 1
                            else f"{n_g} × {grule.group_labels[g]} (needs ≥ {k})")
            elif grule.family == "pair":
                group = grule.pair_students
                inside = [i for i in group if i in member_set]
                if grule.rule.kind == "together" and 0 < len(inside) < len(group):
                    outside = [i for i in group if i not in member_set]
                    text = (f"keep-together group split: {', '.join(ids[i] for i in inside)} "
                            f"here, {', '.join(ids[i] for i in outside)} elsewhere")
                elif grule.rule.kind == "apart" and len(inside) >= 2:
                    text = f"keep-apart group together: {', '.join(ids[i] for i in inside)}"
            if text:
                out.append({"rule": grule.label, "family": grule.family, "hard": grule.hard,
                            "text": text, "students": list(holders[g])})
    out.extend(bounds_violations(model, members))
    out.extend(shared_violations(model, members))
    history = model.history
    hp = model.policy.history
    if history is not None and hp.enabled:
        cap = hp.max_times_together
        ms = sorted(members)
        for a_pos, a in enumerate(ms):
            for b in ms[a_pos + 1:]:
                times = history.cap_counts.get((a, b), 0)
                if times <= 0:
                    continue
                met = ", ".join(history.met_in.get((a, b), []))
                over = cap is not None and times + 1 > cap
                out.append({"rule": "history cap" if over else "history", "family": "history",
                            "hard": bool(over and hp.hard_cap),
                            "text": f"repeat: {ids[a]} & {ids[b]} (met in {met})",
                            "students": [a, b]})
    if model.policy.preferences.use_student_preferences:
        for i in members:
            for j in model.cohort.avoid.get(i, []):
                if j in member_set:
                    out.append({"rule": "preferences", "family": "preferences",
                                "hard": bool(model.policy.preferences.avoid_hard),
                                "text": f"avoid violated: {ids[i]} → {ids[j]}",
                                "students": [i, j]})
    return out


def bounds_violations(model: ObjectiveModel, members: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r, brule in enumerate(model.bound_rules):
        st = model.bounds_stats(r, members)
        if not st["broken"]:
            continue
        valued = [(model.bound_values[i][r] + brule.shift, i) for i in members
                  if model.bound_values[i][r] is not None]
        name = brule.attribute
        rule = brule.rule
        mean = st["mean"]
        if brule.mean_min is not None and mean < float(rule.mean_min) - 1e-9:
            text = f"mean {name} {_fmt_value(mean)} (min {_fmt_value(rule.mean_min)})"
            low = [i for v, i in valued if v < float(rule.mean_min)]
            out.append(_issue(brule, "bounds", text, low))
        if brule.mean_max is not None and mean > float(rule.mean_max) + 1e-9:
            text = f"mean {name} {_fmt_value(mean)} (max {_fmt_value(rule.mean_max)})"
            high = [i for v, i in valued if v > float(rule.mean_max)]
            out.append(_issue(brule, "bounds", text, high))
        if brule.max_range is not None and st["range"] > brule.max_range + 1e-9:
            lo = min(v for v, _ in valued)
            hi = max(v for v, _ in valued)
            text = f"{name} range {_fmt_value(st['range'])} (max {_fmt_value(brule.max_range)})"
            out.append(_issue(brule, "bounds", text, [i for v, i in valued if v in (lo, hi)]))
    return out


def shared_violations(model: ObjectiveModel, members: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    members = list(members)
    if not members:
        return out
    ids = model.cohort.student_ids
    for r, srule in enumerate(model.shared_rules):
        st = model.shared_stats(r, members)
        if not st["units"]:
            continue
        name = srule.attribute
        held = st["held"]
        missing = [i for i in members if not model.shared_codes[i][r]]
        top = max(range(len(held)), key=lambda c: (held[c], -c)) if held else None
        if missing:
            blocking = missing
        elif top is not None:
            blocking = [i for i in members if top not in model.shared_codes[i][r]]
        else:
            blocking = []
        if srule.categorical:
            distinct = sum(1 for c in held if c > 0)
            text = f"different {name} ({distinct} values)"
        elif st["shared"] == 0:
            text = f"no common {name}"
        else:
            text = f"only {st['shared']} common {name} value(s) (needs {srule.min_shared})"
        if missing:
            shown = ", ".join(ids[i] for i in missing[:3]) + (" …" if len(missing) > 3 else "")
            text += f"; no {name} for {shown}"
        out.append(_issue(srule, "shared", text, blocking))
    return out


def _issue(rule: Any, family: str, text: str, students: Sequence[int]) -> Dict[str, Any]:
    return {"rule": rule.label, "family": family, "hard": rule.hard, "text": text,
            "students": list(students)}


def _rule_attributes(model: ObjectiveModel) -> List[str]:
    names: List[str] = []
    for info in model.terms:
        if info.attribute and info.attribute not in names:
            names.append(info.attribute)
    return names


def team_composition(model: ObjectiveModel, members: Sequence[int]
                     ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Optional[float]]]:
    composition: Dict[str, Dict[str, int]] = {}
    means: Dict[str, Optional[float]] = {}
    for name in _rule_attributes(model):
        attr = model.cohort.attributes.get(name)
        if attr is None:
            continue
        if attr.kind == "numeric":
            vals = [attr.values[i] for i in members if not math.isnan(attr.values[i])]
            means[name] = _num(sum(vals) / len(vals), 3) if vals else None
            continue
        counts: Dict[str, int] = {}
        for i in members:
            codes = attr.multi[i] if attr.kind == "multi" else (
                [attr.codes[i]] if attr.codes[i] >= 0 else [])
            if not codes:
                counts["(missing)"] = counts.get("(missing)", 0) + 1
            for c in codes:
                value = attr.categories[c]
                counts[value] = counts.get(value, 0) + 1
        composition[name] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return composition, means


def build_report(model: ObjectiveModel, assignment: Sequence[int], *,
                 baseline_samples: int = 200, seed: int = 0,
                 team_names: Optional[Sequence[str]] = None,
                 locks: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    if model.anchor is not None or model.pair_penalties:
        model = model.without_extras()
    teams = model.teams_of(assignment)
    T = len(teams)
    names = list(team_names or [])
    names = names[:T] + default_team_names(T)[len(names[:T]):]
    ids = model.cohort.student_ids

    penalties = [model.team_penalty(m) for m in teams]
    breakdowns = [model.team_breakdown(m) for m in teams]
    values = [model._team_values(m) for m in teams]
    objective = model.aggregate(penalties)
    sizes = [len(m) for m in teams]
    hard_total = model.assignment_hard_units(assignment)

    baseline = random_baseline(model, objective, sizes=sizes, samples=baseline_samples,
                               seed=seed, locks=locks)
    quality_index = _quality(objective, baseline["mean"])
    if model.has_hard_terms:
        soft_model = model.with_hard_penalty(0.0)
        soft_objective = soft_model.evaluate(assignment)
        soft_baseline = random_baseline(soft_model, soft_objective, sizes=sizes,
                                        samples=baseline_samples, seed=seed, locks=locks)
        soft_penalties = [soft_model.team_penalty(m) for m in teams]
    else:
        soft_model = model
        soft_objective, soft_baseline = objective, baseline
        soft_penalties = penalties
    soft_quality = _quality(soft_objective, soft_baseline["mean"])
    outlier = _outlier_team(soft_penalties, lambda t: soft_model.team_breakdown(teams[t]), names,
                            hard_units=lambda t: model.hard_units(teams[t]))

    team_entries = []
    for t, members in enumerate(teams):
        violations = team_violations(model, members)
        composition, means = team_composition(model, members)
        team_entries.append({
            "index": t,
            "name": names[t],
            "members": [ids[i] for i in members],
            "size": len(members),
            "penalty": _num(penalties[t]),
            "hard_units": model.hard_units(members),
            "breakdown": {k: _num(v) for k, v in breakdowns[t].items()},
            "violations": [v["text"] for v in violations],
            "issues": [
                {**v, "students": [ids[i] for i in v["students"]]} for v in violations
            ],
            "composition": composition,
            "numeric_means": means,
        })

    mean_p = sum(penalties) / T if T else 0.0
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in penalties) / T) if T else 0.0
    overall = {
        "objective": _num(objective),
        "mean_team_penalty": _num(mean_p),
        "worst_team_penalty": _num(max(penalties) if penalties else 0.0),
        "std_team_penalty": _num(std_p),
        "hard_units": int(hard_total),
        "quality_index": _num(quality_index, 2),
        "random_baseline": baseline,
        "soft_objective": _num(soft_objective),
        "quality_index_soft": _num(soft_quality, 2),
        "random_baseline_soft": soft_baseline,
        "n_teams": T,
        "n_students": model.n,
        "fairness_weight": model.fairness_weight,
        "outlier_team": outlier,
    }

    rules = []
    unavoidable_hard = 0.0
    for r, grule in enumerate(model.group_rules):
        term_values = [values[t][grule.term] for t in range(T)]
        units = [group_units(model, r, m) for m in teams]
        compliant = sum(1 for u in units if u < 1e-9)
        floor = unavoidable_group_units(model, r, sizes)
        total = sum(units)
        if grule.hard:
            unavoidable_hard += min(floor, total)
        rules.append({
            "label": grule.label,
            "kind": grule.family,
            "hard": grule.hard,
            "weight": float(grule.rule.weight) if grule.rule is not None else 0.0,
            "teams_compliant": compliant,
            "teams_total": T,
            "units": _units(total),
            "unavoidable_units": _units(floor),
            "summary": violation_summary(total, floor),
            "points": _num(sum(term_values)),
            "compliance_pct": _num(100.0 * compliant / T if T else 100.0, 2),
        })
    for r, brule in enumerate(model.bound_rules):
        term_values = [values[t][brule.term] for t in range(T)]
        per_team = [model.bounds_stats(r, m) for m in teams]
        broken = sum(st["broken"] for st in per_team)
        compliant = sum(1 for st in per_team if not st["broken"])
        rules.append({
            "label": brule.label,
            "kind": "bounds",
            "attribute": brule.attribute,
            "hard": brule.hard,
            "weight": brule.weight,
            "teams_compliant": compliant,
            "teams_total": T,
            "units": _units(broken),
            "excess_sd": _num(sum(st["excess_sd"] for st in per_team)),
            "unavoidable_units": 0,
            "summary": violation_summary(broken),
            "points": _num(sum(term_values)),
            "compliance_pct": _num(100.0 * compliant / T if T else 100.0, 2),
        })
    for r, srule in enumerate(model.shared_rules):
        term_values = [values[t][srule.term] for t in range(T)]
        units = [model.shared_stats(r, m)["units"] for m in teams]
        compliant = sum(1 for u in units if not u)
        floor = unavoidable_shared_units(model, r, sizes)
        if srule.hard:
            unavoidable_hard += min(floor, sum(units))
        rules.append({
            "label": srule.label,
            "kind": "shared",
            "attribute": srule.attribute,
            "hard": srule.hard,
            "weight": srule.weight,
            "teams_compliant": compliant,
            "teams_total": T,
            "units": _units(sum(units)),
            "unavoidable_units": _units(floor),
            "summary": violation_summary(sum(units), floor),
            "points": _num(sum(term_values)),
            "compliance_pct": _num(100.0 * compliant / T if T else 100.0, 2),
        })
    for skipped in model.skipped_rules:
        rule = skipped.get("rule")
        rules.append({
            "label": f"{skipped['label']} (skipped)",
            "kind": skipped["family"],
            "hard": bool(skipped.get("hard")),
            "weight": float(getattr(rule, "weight", 0.0) or 0.0),
            "teams_compliant": T,
            "teams_total": T,
            "units": 0,
            "unavoidable_units": 0,
            "summary": f"skipped — {skipped['reason']}",
            "points": 0.0,
            "compliance_pct": 100.0,
            "skipped": True,
        })
    overall["avoidable_hard_units"] = int(max(0, round(hard_total - unavoidable_hard)))
    overall["unavoidable_hard_units"] = _units(unavoidable_hard)

    attribute_rules = []
    for info_index, info in enumerate(model.terms):
        if info.family != "attribute":
            continue
        term_values = [values[t][info_index] for t in range(T)]
        components = 2.0 if info.goal == "spread" else 1.0
        if info.weight > 0 and T:
            score = sum(term_values) / info.weight / components / T
        else:
            score = 0.0
        attribute_rules.append({
            "label": info.label,
            "kind": "attribute",
            "attribute": info.attribute,
            "goal": info.goal,
            "hard": False,
            "weight": info.weight,
            "score_normalised": _num(score),
            "quality_pct": _num(max(0.0, min(100.0, 100.0 * (1.0 - score))), 2),
            "points": _num(sum(term_values)),
        })

    warnings = list(model.warnings)
    if outlier is not None:
        warnings.append(outlier["text"])
    report = {
        "overall": overall,
        "teams": team_entries,
        "rules": rules + attribute_rules,
        "attribute_rules": attribute_rules,
        "history": history_summary(model, teams),
        "preferences": preference_summary(model, teams),
        "warnings": warnings,
    }
    return report


def _outlier_team(soft_penalties: Sequence[float], breakdown: Callable[[int], Dict[str, float]],
                  names: Sequence[str],
                  hard_units: Optional[Callable[[int], float]] = None) -> Optional[Dict[str, Any]]:
    T = len(soft_penalties)
    if T < 3:
        return None
    ordered = sorted(soft_penalties)
    mid = T // 2
    median = ordered[mid] if T % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    worst = max(range(T), key=lambda t: soft_penalties[t])
    value = soft_penalties[worst]
    if value <= OUTLIER_FACTOR * max(median, 0.0) or value - median < OUTLIER_MIN_POINTS:
        return None
    terms = sorted(((k, v) for k, v in breakdown(worst).items() if v > 1e-9),
                   key=lambda kv: -kv[1])[:2]
    reason = ", ".join(f"{k} {v:.1f}" for k, v in terms)
    ratio = value / median if median > _EPS else None
    times = f"{ratio:.1f}×" if ratio is not None else "far above"
    units = int(round(float(hard_units(worst)))) if hard_units is not None else 0
    hard = f" (plus {units} hard-rule violation unit(s))" if units else ""
    text = (f"{names[worst]} scores {value:.1f} soft penalty points{hard}, {times} the median "
            f"team's soft penalty ({median:.1f}){f'; mostly {reason}' if reason else ''}. Check "
            "it in the team view: a single team this far behind usually means a rule only it "
            "breaks, or that more search time (restarts) would help.")
    return {"index": worst, "name": names[worst], "penalty": _num(value),
            "median": _num(median), "ratio": None if ratio is None else _num(ratio, 3),
            "main_terms": [k for k, _ in terms], "hard_units": units, "text": text}


def unavoidable_group_units(model: ObjectiveModel, r: int, sizes: Sequence[int]) -> float:
    grule = model.group_rules[r]
    if not sizes:
        return 0.0
    counts = [0] * grule.ngroups
    for i in range(model.n):
        for g in model.group_codes[i][r]:
            counts[g] += 1
    if grule.family == "count":
        groups = {f"{label}#{g}": counts[g] for g, label in enumerate(grule.group_labels)}
        return float(count_rule_floor(grule.rule, groups, sizes)[0])
    if grule.family == "isolation":
        k = grule.min_together or 2
        return float(sum(isolation_floor(c, k, sizes) for c in counts))
    if grule.family == "pair" and grule.rule is not None and grule.rule.kind == "apart":
        g, teams = len(grule.pair_students), len(sizes)
        if g > teams:
            q, rem = divmod(g, teams)
            return float(rem * math.comb(q + 1, 2) + (teams - rem) * math.comb(q, 2))
    return 0.0


def unavoidable_shared_units(model: ObjectiveModel, r: int, sizes: Sequence[int]) -> float:
    srule = model.shared_rules[r]
    if not sizes:
        return 0.0
    short = [len(model.shared_codes[i][r]) for i in range(model.n)
             if len(model.shared_codes[i][r]) < srule.min_shared]
    if not short:
        return 0.0
    teams = math.ceil(len(short) / max(sizes))
    return float(teams * (srule.min_shared - max(short)))


def group_units(model: ObjectiveModel, r: int, members: Sequence[int]) -> float:
    grule = model.group_rules[r]
    counts = [0] * grule.ngroups
    for i in members:
        for g in model.group_codes[i][r]:
            counts[g] += 1
    return float(sum(grule.table[c] for c in counts))


def history_summary(model: ObjectiveModel, teams: Sequence[Sequence[int]]) -> Dict[str, Any]:
    history = model.history
    ids = model.cohort.student_ids
    hp = model.policy.history
    available = bool(history is not None and history.has_earlier_teams)
    out: Dict[str, Any] = {"available": available, "enabled": bool(hp.enabled),
                           "cap": hp.max_times_together if hp.enabled else None,
                           "repeated_pairs": 0, "over_cap_pairs": 0, "max_times": 0,
                           "contact_pairs": 0, "section_reunions": 0, "section_pairs": [],
                           "pairs": []}
    if history is None or not hp.enabled:
        return out
    cap = hp.max_times_together
    pairs = []
    sections = []
    repeated = over = 0
    max_times = 0
    for t, members in enumerate(teams):
        ms = sorted(members)
        if len(ms) >= 2:
            max_times = max(max_times, 1)
        for a_pos, a in enumerate(ms):
            for b in ms[a_pos + 1:]:
                times = history.cap_counts.get((a, b), 0)
                met = history.met_in.get((a, b))
                if times <= 0:
                    if met:
                        sections.append({"a": ids[a], "b": ids[b], "team": t,
                                         "weight": _num(history.weights.get((a, b), 0.0)),
                                         "met_in": list(met)})
                    continue
                repeated += 1
                is_over = cap is not None and times + 1 > cap
                over += int(is_over)
                max_times = max(max_times, times + 1)
                pairs.append({
                    "a": ids[a], "b": ids[b], "team": t, "times_before": times,
                    "weight": _num(history.weights.get((a, b), 0.0)),
                    "met_in": list(met or []), "over_cap": is_over,
                })
    out.update(repeated_pairs=repeated, over_cap_pairs=over, max_times=max_times,
               contact_pairs=repeated + len(sections), section_reunions=len(sections),
               section_pairs=sections, pairs=pairs)
    return out


def preference_summary(model: ObjectiveModel, teams: Sequence[Sequence[int]]) -> Dict[str, Any]:
    team_of: Dict[int, int] = {}
    for t, members in enumerate(teams):
        for i in members:
            team_of[i] = t
    ids = model.cohort.student_ids
    prefer_total = prefer_ok = avoid_total = avoid_bad = 0
    violated = []
    for i, listed in model.cohort.prefer.items():
        for j in listed:
            prefer_total += 1
            prefer_ok += int(team_of.get(i) is not None and team_of.get(i) == team_of.get(j))
    for i, listed in model.cohort.avoid.items():
        for j in listed:
            avoid_total += 1
            if team_of.get(i) is not None and team_of.get(i) == team_of.get(j):
                avoid_bad += 1
                violated.append({"student": ids[i], "avoids": ids[j], "team": team_of[i]})
    return {"used": bool(model.policy.preferences.use_student_preferences),
            "prefer_satisfied": prefer_ok, "prefer_total": prefer_total,
            "avoid_violated": avoid_bad, "avoid_total": avoid_total, "violations": violated}
