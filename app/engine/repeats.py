from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from .history import PriorProject
from .metrics import history_summary
from .objective import HISTORY_CAP_LABEL, ObjectiveModel
from .policy import Policy
from .runner import ProjectSpec, evaluation_setup
from .search_state import SearchState

_EPS = 1e-9
_NAMED_BLOCKERS = 25
_CHAIN_FIRST_STEPS = 40
_CHAINS_KEPT = 60
_CHAIN_PAIRS = 30

CAUSES = ("requested", "locked", "edited", "search", "rules_allow", "unavoidable")
CAUSE_SHORT = {
    "requested": "requested together",
    "locked": "pinned together",
    "edited": "placed by hand",
    "search": "search ended early",
    "rules_allow": "rules allow repeats",
    "unavoidable": "unavoidable here",
}
_SUMMARY = {
    "search": ("could be avoided — the search ended early",
               "could be avoided — the search ended early"),
    "unavoidable": ("is unavoidable with these participants and rules",
                    "are unavoidable with these participants and rules"),
    "rules_allow": ("is allowed by your repeat rule (traded for better balance)",
                    "are allowed by your repeat rule (traded for better balance)"),
    "requested": ("was requested (pair rule or preference)",
                  "were requested (pair rule or preference)"),
    "locked": ("is pinned together", "are pinned together"),
    "edited": ("was placed together by hand", "were placed together by hand"),
}
_TEAM_RE = re.compile(r"^team[\s_-]*(\d+)$", re.IGNORECASE)


def team_label(name: Optional[str], index: int) -> str:
    text = str(name or "").strip()
    match = _TEAM_RE.match(text)
    if match:
        return f"Team {int(match.group(1))}"
    return text or f"Team {index + 1}"


def summarize_explanations(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "No repeat pairs in the generated projects."
    counts = Counter(r["cause"] for r in rows)
    parts = []
    for cause in ("search", "unavoidable", "rules_allow", "requested", "locked", "edited"):
        k = counts.get(cause, 0)
        if k:
            parts.append(f"{k} {_SUMMARY[cause][0 if k == 1 else 1]}")
    n = len(rows)
    return f"{n} repeat pair{'s' if n != 1 else ''}: " + "; ".join(parts) + "."


def _plural(k: int, word: str) -> str:
    return f"{k} {word}{'' if k == 1 else 's'}"


def _search_detail(stats: Optional[Dict[str, Any]]) -> str:
    stats = stats or {}
    bits = []
    if stats.get("time_limited"):
        bits.append("the search hit its time limit")
    iterations = stats.get("iterations")
    restarts = stats.get("restarts")
    if iterations:
        budget = f"{int(iterations):,} moves"
        if restarts:
            budget += f", {_plural(int(restarts), 'restart')}"
        bits.append(f"it ran with a fixed budget of {budget}")
    elif restarts and int(restarts) <= 1:
        bits.append("it ran a single restart")
    if not bits:
        return "The search stopped before finding it"
    text = " and ".join(bits)
    return text[0].upper() + text[1:]


class _Explainer:
    def __init__(self, model: ObjectiveModel, assignment: List[int], names: List[str],
                 project_id: str, size_min: int, size_max: int, flexible: bool) -> None:
        self.model = model
        self.assignment = assignment
        self.names = names
        self.pid = project_id
        self.ids = model.cohort.student_ids
        self.history = model.history
        self.hard_cap = bool(model.policy.history.hard_cap)
        self.state = SearchState(model, list(assignment))
        self.size_min = size_min
        self.size_max = size_max
        self.flexible = flexible
        self.hard_labels = {t.label for t in model.terms if t.hard} - {HISTORY_CAP_LABEL}
        self._chain_state: Optional[SearchState] = None

    def times(self, i: int, j: int) -> int:
        if self.history is None:
            return 0
        return self.history.times_together(i, j)

    def team_name(self, t: int) -> str:
        return team_label(self.names[t] if t < len(self.names) else None, t)

    def free_partners(self, i: int) -> int:
        return sum(1 for j in range(self.model.n) if j != i and self.times(i, j) == 0)

    def _repeats_of(self, i: int, members: Sequence[int]) -> int:
        return sum(1 for j in members if j != i and self.times(i, j) > 0)

    def _other_hard(self, members: Sequence[int]) -> float:
        units = self.model.team_hard_units_exact(members)
        if self.hard_cap and self.model.hist_cap:
            ms = sorted(members)
            for p, a in enumerate(ms):
                for b in ms[p + 1:]:
                    if (a, b) in self.model.hist_cap:
                        units -= 1
        return units

    def _repeat_pairs(self, members: Sequence[int]) -> int:
        ms = list(members)
        return sum(1 for p, x in enumerate(ms) for y in ms[p + 1:] if self.times(x, y) > 0)

    def candidates(self, movers: Sequence[int], locked: Set[int]) -> List[Dict[str, Any]]:
        st = self.state
        out: List[Dict[str, Any]] = []
        members = st.members
        for i in movers:
            ti = self.assignment[i]
            base_ti = [m for m in members[ti] if m != i]
            hard_ti: Optional[float] = None
            for j in range(self.model.n):
                tj = self.assignment[j]
                if tj == ti or j in locked:
                    continue
                rest_tj = [m for m in members[tj] if m != j]
                after_i = base_ti + [j]
                after_j = rest_tj + [i]
                created = self._repeats_of(i, rest_tj) + self._repeats_of(j, base_ti)
                hard_delta = None
                if created == 0:
                    if hard_ti is None:
                        hard_ti = self._other_hard(members[ti])
                    hard_delta = (self._other_hard(after_i) + self._other_hard(after_j)
                                  - hard_ti - self._other_hard(members[tj]))
                out.append({"kind": "swap", "i": i, "j": j, "from": ti, "to": tj,
                            "delta": st.delta_swap(i, j), "created": created,
                            "hard_delta": hard_delta,
                            "before": {ti: list(members[ti]), tj: list(members[tj])},
                            "after": {ti: after_i, tj: after_j}})
            if not self.flexible or st.sizes[ti] - 1 < self.size_min:
                continue
            for t in range(self.model.n_teams):
                if t == ti or st.sizes[t] + 1 > self.size_max:
                    continue
                after_t = list(members[t]) + [i]
                created = self._repeats_of(i, members[t])
                hard_delta = None
                if created == 0:
                    hard_delta = (self._other_hard(base_ti) + self._other_hard(after_t)
                                  - self._other_hard(members[ti]) - self._other_hard(members[t]))
                out.append({"kind": "move", "i": i, "j": None, "from": ti, "to": t,
                            "delta": st.delta_move(i, t), "created": created,
                            "hard_delta": hard_delta,
                            "before": {ti: list(members[ti]), t: list(members[t])},
                            "after": {ti: base_ti, t: after_t}})
        return out

    def chains(self, first: List[Dict[str, Any]], a: int, b: int, locked: Set[int],
               limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        order = sorted(first, key=lambda c: (c["created"], c["delta"], c["i"], c["to"]))
        n = self.model.n
        if self._chain_state is None:
            self._chain_state = SearchState(self.model, list(self.assignment))
        st2 = self._chain_state
        for c in order[:limit]:
            i, j = c["i"], c["j"]
            removed = self._repeats_of(i, self.state.members[c["from"]])
            if j is not None:
                removed += self._repeats_of(j, self.state.members[c["to"]])
            net1 = c["created"] - removed
            if j is not None:
                st2.apply_swap(i, j)
            else:
                st2.apply_move(i, c["to"])
            try:
                assign2 = st2.assignment
                mem2 = st2.members
                moved = {i} if j is None else {i, j}
                conflicted = set()
                for t in c["after"]:
                    ms = mem2[t]
                    for p, x in enumerate(ms):
                        for y in ms[p + 1:]:
                            if (x in moved or y in moved) and self.times(x, y) > 0 \
                                    and {x, y} != {a, b}:
                                conflicted.update((x, y))
                for x in sorted(conflicted - locked):
                    tx = assign2[x]
                    rest_x = [m for m in mem2[tx] if m != x]
                    out_x = self._repeats_of(x, rest_x)
                    for y in range(n):
                        ty = assign2[y]
                        if ty == tx or y in locked or y in moved:
                            continue
                        ta = ty if a == x else (tx if a == y else assign2[a])
                        tb = ty if b == x else (tx if b == y else assign2[b])
                        if ta == tb:
                            continue
                        rest_y = [m for m in mem2[ty] if m != y]
                        net2 = (self._repeats_of(y, rest_x) + self._repeats_of(x, rest_y)
                                - out_x - self._repeats_of(y, rest_y))
                        if net1 + net2 >= 0:
                            continue
                        teams = set(c["after"]) | {tx, ty}
                        final = {t: list(mem2[t]) for t in teams}
                        final[tx] = rest_x + [y]
                        final[ty] = rest_y + [x]
                        hard = (sum(self._other_hard(ms) for ms in final.values())
                                - sum(self._other_hard(self.state.members[t]) for t in teams))
                        if hard > 1e-9:
                            continue
                        second = {"kind": "swap", "i": x, "j": y, "from": tx, "to": ty}
                        out.append({"steps": [c, second],
                                    "delta": c["delta"] + st2.delta_swap(x, y)})
                        if len(out) >= _CHAINS_KEPT:
                            return out
            finally:
                if j is not None:
                    st2.apply_swap(i, j)
                else:
                    st2.apply_move(i, c["from"])
        return out

    def describe(self, c: Dict[str, Any]) -> str:
        i = self.ids[c["i"]]
        if c["kind"] == "swap":
            return (f"swap {i} ({self.team_name(c['from'])}) with {self.ids[c['j']]} "
                    f"({self.team_name(c['to'])})")
        return f"move {i} from {self.team_name(c['from'])} to {self.team_name(c['to'])}"

    def describe_chain(self, steps: Sequence[Dict[str, Any]]) -> str:
        return ", then ".join(self.describe(c) for c in steps)

    def step_dict(self, c: Dict[str, Any]) -> Dict[str, Any]:
        return {"kind": c["kind"], "student": self.ids[c["i"]],
                "with": self.ids[c["j"]] if c["j"] is not None else None,
                "from_team": c["from"], "to_team": c["to"]}

    def broken_rules(self, c: Dict[str, Any]) -> List[str]:
        names = []
        for t, before in c["before"].items():
            old = self.model.team_breakdown(before)
            new = self.model.team_breakdown(c["after"][t])
            for label in self.hard_labels:
                if new.get(label, 0.0) - old.get(label, 0.0) > 1e-6 and label not in names:
                    names.append(label)
        return names


def _chain_key(chain: Dict[str, Any]) -> Any:
    return (chain["delta"], len(chain["steps"]),
            [(c["i"], c["to"]) for c in chain["steps"]])


def _pair_rule_for(model: ObjectiveModel, policy: Policy, project_id: str, a: int, b: int
                   ) -> Optional[Any]:
    for rule in policy.pairs:
        if not rule.enabled or rule.kind != "together":
            continue
        if rule.projects is not None and str(project_id) not in {str(p) for p in rule.projects}:
            continue
        members = {model.cohort.lookup(sid) for sid in rule.students}
        if a in members and b in members:
            return rule
    return None


def explain_repeats(roster_frame: pd.DataFrame, policy: Any, project: ProjectSpec,
                    teams: List[List[str]], prior_projects: Sequence[PriorProject] = (), *,
                    base_teams: Optional[List[List[str]]] = None,
                    locks: Optional[Dict[str, int]] = None,
                    stats: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    policy = Policy.from_dict(policy) if not isinstance(policy, Policy) else policy
    setup = evaluation_setup(roster_frame, policy, project, teams, prior_projects)
    model, assignment = setup.model, setup.assignment
    pid = str(project.project_id)
    summary = history_summary(model, model.teams_of(assignment))
    if not summary["pairs"]:
        return []
    hp = policy.history
    ex = _Explainer(model, assignment, list(setup.prep.team_names), pid,
                    int(project.size_min), int(project.size_max),
                    bool(policy.solver.flexible_sizes))
    cohort = model.cohort
    locked: Dict[int, Any] = {}
    for sid, t in (locks or {}).items():
        i = cohort.lookup(sid)
        if i is not None:
            locked[i] = t
    base_team_of: Dict[int, int] = {}
    for t, members in enumerate(base_teams or []):
        for sid in members:
            i = cohort.lookup(sid)
            if i is not None:
                base_team_of[i] = t
    rules_text = _rules_text(hp)
    chain_pairs = 0
    out: List[Dict[str, Any]] = []
    for pair in summary["pairs"]:
        a, b = cohort.lookup(pair["a"]), cohort.lookup(pair["b"])
        t = int(pair["team"])
        sa, sb = cohort.student_ids[a], cohort.student_ids[b]
        team_name = ex.team_name(t)
        hard_forbidden = bool(pair["over_cap"]) and hp.hard_cap
        free = {sa: ex.free_partners(a), sb: ex.free_partners(b)}
        facts: List[str] = []
        row = {"project_id": pid, "a": sa, "b": sb, "team": t, "team_name": team_name,
               "met_in": list(pair["met_in"]), "times_before": int(pair["times_before"]),
               "over_cap": bool(pair["over_cap"]), "free_partners": free,
               "pool_size": model.n, "facts": facts, "change": None}

        def done(cause: str, reason: str, fix: Optional[str]) -> None:
            row.update(cause=cause, short=CAUSE_SHORT[cause], reason=reason, fix=fix)
            out.append(row)

        rule = _pair_rule_for(model, policy, pid, a, b)
        prefers = [(x, y) for x, y in ((a, b), (b, a)) if y in cohort.prefer.get(x, [])]
        use_prefs = policy.preferences.use_student_preferences
        requested = None
        if rule is not None:
            name = rule.label or ", ".join(map(str, rule.students))
            text = f"Kept together by the {'hard' if rule.hard else 'soft'} pair rule '{name}'"
            if rule.hard or not hard_forbidden:
                requested = (text + ".", "Intended: nothing to fix. To let them split up, "
                             "limit the rule to some projects (Allocation Rules › Pairs).")
            else:
                facts.append(text + " (soft, it cannot outweigh the hard repeat cap).")
        if requested is None and prefers and use_prefs:
            x, y = prefers[0]
            both = len(prefers) == 2
            text = (f"Students {sa} and {sb} asked to work with each other (preference)" if both
                    else f"Student {cohort.student_ids[x]} asked to work with "
                         f"{cohort.student_ids[y]} (preference)")
            if not hard_forbidden:
                requested = (text + ", which outweighed the repeat penalty.",
                             "Intended: nothing to fix. To prioritise new teammates, raise the "
                             "repeat weight or lower the preference weight (Allocation Rules › "
                             "🔁 History & preferences).")
            else:
                facts.append(text + " (a preference cannot outweigh the hard repeat cap).")
        if requested is not None:
            done("requested", *requested)
            continue
        if a in locked and b in locked and locked[a] == locked[b]:
            done("locked", f"Both are pinned to {team_name} in the Team Editor.",
                 "Unpin one of them in the Team Editor, then re-optimise or move them.")
            continue
        for i in (a, b):
            if i in locked:
                facts.append(f"{cohort.student_ids[i]} is pinned to {team_name}.")
        if base_team_of and (base_team_of.get(a) != base_team_of.get(b)
                             or a not in base_team_of or b not in base_team_of):
            where = []
            for i in (a, b):
                bt = base_team_of.get(i)
                where.append(f"{cohort.student_ids[i]} in " +
                             (team_label(None, bt) if bt is not None else "no team"))
            done("edited", "Placed together by hand in the Team Editor (the run had them apart: "
                 + ", ".join(where) + ").",
                 "Undo that edit, or move one of them in the Team Editor.")
            continue
        movers = [i for i in (a, b) if i not in locked]
        cands = ex.candidates(movers, set(locked))
        clean = [{"steps": [c], "delta": c["delta"]} for c in cands
                 if c["created"] == 0 and c["hard_delta"] <= 1e-9]
        best = min(clean, key=_chain_key) if clean else None
        tried_chains = False
        if (best is None or best["delta"] >= -_EPS) and chain_pairs < _CHAIN_PAIRS:
            chain_pairs += 1
            tried_chains = True
            chains = ex.chains([c for c in cands if c["created"] > 0], a, b, set(locked),
                               _CHAIN_FIRST_STEPS)
            best2 = min(chains, key=_chain_key) if chains else None
            if best2 is not None and (best is None or best2["delta"] < best["delta"] - _EPS):
                best = best2
        allowed = not hard_forbidden
        if allowed:
            facts.append(f"Your rules allow this repeat ({rules_text}).")
        if best is not None:
            change = ex.describe_chain(best["steps"])
            row["change"] = {"steps": [ex.step_dict(c) for c in best["steps"]],
                             "delta": round(best["delta"], 6), "text": change}
        if best is not None and (best["delta"] < -_EPS or not allowed):
            done("search",
                 f"A better arrangement exists: {change} removes this repeat with no new "
                 f"problems (objective {best['delta']:+.2f}). {_search_detail(stats)}.",
                 "Run again with more search (e.g. the Thorough preset), or "
                 f"{change} in the Team Editor.")
            continue
        if best is not None:
            done("rules_allow",
                 f"Your rules allow pairs to meet again ({rules_text}): separating them costs "
                 f"{best['delta']:+.2f} points elsewhere at best ({change}), so the search "
                 "kept this repeat for better balance.",
                 "To rule repeats out, make the repeat cap hard or raise the repeat weight "
                 "(Allocation Rules › 🔁 History & preferences), then run again.")
            continue
        done("unavoidable", _unavoidable_text(ex, a, b, cands, movers, free, pid, bool(locked),
                                               tried_chains),
             _unavoidable_fix(hp, allowed, free, ex))
    return out


def _rules_text(hp: Any) -> str:
    cap = hp.max_times_together
    parts = []
    if cap is None:
        parts.append("no repeat cap")
    elif not hp.hard_cap:
        parts.append(f"soft cap of {cap} team(s), weight {hp.cap_weight:g}")
    elif cap >= 2:
        parts.append(f"pairs may share up to {cap} teams")
    if hp.repeat_weight:
        parts.append(f"repeat weight {hp.repeat_weight:g}")
    return "; ".join(parts) or "repeats allowed"


def _tightness(ex: "_Explainer", a: int, b: int, free: Dict[str, int], pid: str) -> str:
    sa, sb = ex.ids[a], ex.ids[b]
    need = max(1, len(ex.state.members[ex.assignment[a]]) - 1)
    text = (f"{sa} has {free[sa]} and {sb} has {free[sb]} classmates in Project {pid}'s pool "
            f"({ex.model.n} students) they have not shared a team with yet")
    if min(free[sa], free[sb]) < need:
        text += f" — fewer than the {need} teammates a team of {need + 1} needs"
    return text


def _unavoidable_text(ex: _Explainer, a: int, b: int, cands: List[Dict[str, Any]],
                      movers: List[int], free: Dict[str, int], pid: str,
                      pinned: bool = False, chains: bool = True) -> str:
    ids = ex.ids
    sa, sb = ids[a], ids[b]
    tightness = _tightness(ex, a, b, free, pid)
    tight = tightness[0].upper() + tightness[1:]
    if not movers:
        return f"Both are pinned, so neither can be moved. {tight}."
    if not cands:
        if pinned:
            return f"Every student they could swap with is pinned, so no swap separates them. {tight}."
        return f"There is no other team to move them to. {tight}."
    repeat_blocked = [c for c in cands if c["created"] > 0]
    hard_blocked = [c for c in cands if c["created"] == 0 and (c["hard_delta"] or 0) > 1e-9]
    rules: Counter = Counter()
    for c in sorted(hard_blocked, key=lambda c: c["delta"])[:_NAMED_BLOCKERS]:
        rules.update(ex.broken_rules(c))
    rule_text = ", ".join(f"'{name}'" for name, _ in rules.most_common(3))
    every = (f"Every swap{' or move' if ex.flexible else ''} that separates {sa} and {sb} "
             + ("(also followed by a second swap)" if chains else "(single swaps checked)"))
    if hard_blocked and not repeat_blocked:
        main = f"{every} breaks a hard rule" + (f" ({rule_text})" if rule_text else "")
    elif repeat_blocked and not hard_blocked:
        main = f"{every} puts one of them with someone they already worked with"
    else:
        main = (f"{every} creates another repeat ({len(repeat_blocked)} of {len(cands)} "
                f"single changes) or breaks a hard rule ({len(hard_blocked)}"
                + (f": {rule_text}" if rule_text else "") + ")")
    return f"{main}. {tight}."


def _unavoidable_fix(hp: Any, allowed: bool, free: Dict[str, int], ex: "_Explainer") -> str:
    if allowed:
        return ("Nothing to gain by small changes. Fewer repeats need more participants, "
                "smaller teams or looser hard rules.")
    return ("No small change helps with these rules: add participants, use smaller teams, relax "
            "a hard rule, or allow pairs to meet twice (soft or higher repeat cap). A "
            "longer search (Thorough preset) may still find a different arrangement.")


__all__ = ["CAUSES", "CAUSE_SHORT", "explain_repeats", "summarize_explanations", "team_label"]
