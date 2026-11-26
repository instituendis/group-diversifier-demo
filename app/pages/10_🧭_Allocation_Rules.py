import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import hosting  # noqa: E402
from app import extensions  # noqa: E402
from app.core import engine_session as es  # noqa: E402
from app.core import policy_files  # noqa: E402
from app.core.data_session import (  # noqa: E402
    active_rule_count,
    engine_setup,
    ensure_session_state,
    feasibility_report,
    flash,
    get_config_dir,
    is_new_upload,
    mark_upload_processed,
    page_link,
    planned_sizes,
    render_flashes,
)
from app.engine.attributes import (  # noqa: E402
    association_matrix,
    attribute_overview,
    count_rule_targets,
    redundant_pairs,
)
from app.engine.defaults import (  # noqa: E402
    PRESETS,
    apply_preset,
    suggest_policy,
)
from app.engine.policy import (  # noqa: E402
    ATTRIBUTE_KINDS,
    CATEGORICAL_GOALS,
    NUMERIC_GOALS,
    AttributeRule,
    CountRule,
    IsolationRule,
    NumericBoundsRule,
    PairRule,
    Policy,
    SharedValueRule,
)
from app.engine.policy_text import (  # noqa: E402
    describe_count_rule,
    describe_isolation_rule,
    describe_numeric_bounds_rule,
    describe_pair_rule,
    describe_policy,
    describe_shared_rule,
)
from app.engine.roster import AttributeInfo, infer_attributes  # noqa: E402
from app.engine.runner import JOINT_ROUNDS  # noqa: E402
from app.engine.solver import effective_time_limit  # noqa: E402
from app.utils.fmt import fmt_g  # noqa: E402

st.set_page_config(page_title="Allocation Rules | Group Diversifier", page_icon="🧭",
                   layout="wide")

config = ensure_session_state()

REV_KEY = "_rules_rev"
SEEN_KEY = "_rules_seen_policy"
POLICY_UPLOAD_SLOT = "policy_json"
ALL_GOALS = ["balance", "diversify", "spread", "cluster", "ignore"]
TO_NUMERIC_GOAL = {"diversify": "spread"}
TO_CATEGORICAL_GOAL = {"spread": "diversify"}
AREA_LABELS = {
    "policy": "Rule set", "attributes": "Attributes & data quality", "sizes": "Team sizes",
    "pool": "Participants", "counts": "Head-count rules", "isolation": "No-lone rules",
    "numeric": "Numeric bounds", "shared": "Shared values", "pairs": "Keep together / apart",
    "history": "Repeat pairings", "preferences": "Student preferences",
}
LEVEL_RENDER = {"error": st.error, "warning": st.warning, "info": st.info}
METHOD_HELP = {
    "anneal": "**Annealing** — simulated annealing plus a local-search polish. Handles every "
              "rule, scales to any cohort and is the recommended default.",
    "milp": "**MILP** — an exact mixed-integer model solved by HiGHS on a linearised "
            "objective. Proves optimality on small problems; on larger ones it usually stops "
            "at its time limit without a proof, and its result is then polished by annealing "
            "automatically (a warning reports the MILP status and gap). Approximates some soft "
            "terms.",
    "milp+anneal": "**MILP + annealing** — the MILP (at most half the time limit) provides one "
                   "extra starting point for the normal annealing restarts; it is kept only if "
                   "it wins. Slower than annealing and rarely better on real cohorts.",
}
GOAL_HELP = """
| Goal | Categorical / multi-value attribute | Numeric attribute |
|---|---|---|
| **balance** | every team mirrors the cohort's mix (e.g. ≈24% Red) | equal team averages |
| **diversify** | as few teammates sharing a value as possible | — |
| **spread** | — | equal averages *and* a cohort-like range inside every team |
| **cluster** | students with the same value together | similar values together |
| **ignore** | not used | not used |

**Weight** (0–10) sets how much the goal matters relative to the other rules: 1 point is
roughly one team's worth of random-level imbalance on that attribute.
"""
GOAL_HELP = GOAL_HELP.replace("≈24% Red", extensions.text("rules.goal_example", "≈24% Red"))

COUNT_EXAMPLE = extensions.text("rules.count_example", "e.g. *at most 1 Chocolate fan*, *at least 1 Dog person*, *at most 2 of any one city*.")
ISOLATION_EXAMPLE = extensions.text("rules.isolation_example", "e.g. no lone Fish owner, no lone Parrot owner.")
SHARED_EXAMPLE = extensions.text("rules.shared_example", "e.g. a common hobby or the same city.")


def rev() -> int:
    return int(st.session_state.get(REV_KEY, 0))


def policy_fingerprint(policy_dict: Dict[str, Any]) -> str:
    return json.dumps(policy_dict or {}, sort_keys=True, default=str)


def k(name: str) -> str:
    return f"rules_{name}_{rev()}"


def commit(policy: Policy, message: Optional[str] = None, *, structural: bool = True) -> None:
    es.save_policy(config, policy)
    st.session_state[SEEN_KEY] = policy_fingerprint(config.policy)
    if structural:
        st.session_state[REV_KEY] = rev() + 1
    if message:
        flash(message)
    st.rerun()


def opt_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def num(value: Any, default: float) -> float:
    number = opt_float(value)
    return default if number is None else number


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "rules"


def same(a: Any, b: Any) -> bool:
    a_missing = a is None or (isinstance(a, float) and math.isnan(a))
    b_missing = b is None or (isinstance(b, float) and math.isnan(b))
    if a_missing or b_missing:
        return a_missing and b_missing
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


_NO_RULES_TEXT = ("no allocation rules found (expected fields such as 'attributes', 'counts' "
                  "or 'history', a top-level 'policy', or 'configuration.policy').")


def _upload_problem(exc: policy_files.PolicyFileError) -> str:
    if exc.kind == "json":
        return f"not valid JSON ({exc.detail})."
    if exc.kind == "object":
        return "expected a JSON object with allocation rules."
    if exc.kind == "config-empty":
        return "this configuration export has no allocation rules yet."
    if exc.kind == "runs":
        return ("this is a run export; import it on Compare Runs (each run keeps the rules "
                "it was formed with).")
    return _NO_RULES_TEXT


def policy_from_upload(raw: bytes) -> Tuple[Policy, str]:
    try:
        loaded = policy_files.read_policy_bytes(raw)
    except policy_files.PolicyFileError as exc:
        raise ValueError(_upload_problem(exc)) from None
    if loaded.problems:
        shown = loaded.problems[:6] + ([f"… and {len(loaded.problems) - 6} more"]
                                       if len(loaded.problems) > 6 else [])
        raise ValueError("wrongly typed fields: " + "; ".join(shown) + ".")
    try:
        origin = f" ({loaded.origin})" if loaded.origin else ""
        return Policy.from_dict(policy_files.numbers_as_text(loaded.policy)), origin
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ValueError(f"not a valid rule set ({exc}).") from None


def rule_problems(rule_policy: Policy, kinds: Dict[str, str]) -> List[str]:
    return rule_policy.validate(kinds)


def suggested_ignore(name: str, infos: Dict[str, AttributeInfo]) -> str:
    info = infos.get(name)
    return f"ignore ({info.flag})" if info is not None and info.flag else "ignore"


def data_notes(infos: Dict[str, AttributeInfo]) -> List[str]:
    notes = []
    for name, info in infos.items():
        bad = int(info.stats.get("non_numeric", 0) or 0)
        if bad and info.non_numeric:
            shown = ", ".join(str(v) for v in info.non_numeric[:5])
            more = " …" if len(info.non_numeric) > 5 else ""
            notes.append(f"{bad} non-numeric value{'s' if bad != 1 else ''} in **{name}**: "
                         f"{shown}{more} — treated as missing.")
        if info.flag == "identifier":
            notes.append(f"**{name}** looks like an identifier (a different value for every "
                         "student), so no rule is suggested for it.")
        elif info.flag == "date":
            notes.append(f"**{name}** holds dates, so no rule is suggested for it (pick a goal "
                         "to use the dates as categories).")
    return notes


def numeric_kind(name: str, infos: Dict[str, AttributeInfo]) -> bool:
    info = infos.get(name)
    return info is not None and info.kind == "numeric"


def summary_lines(policy: Policy, kinds: Dict[str, str]) -> List[str]:
    return describe_policy(policy, kinds)


def student_labels(frame: pd.DataFrame) -> Dict[str, str]:
    preferred = [c for c in ("name", "full_name", "first_name", "last_name") if c in frame.columns]
    if not preferred:
        label_columns = (extensions.hook("ui.rules.label_columns", ())
                         if extensions.active() else ())
        preferred = ([c for c in label_columns if c in frame.columns] if label_columns
                     else [c for c in frame.columns if c != "student_id"][:2])
    labels = {}
    for row in frame[["student_id", *preferred[:3]]].itertuples(index=False):
        parts = [str(v) for v in row[1:] if v is not None and str(v).strip()]
        labels[str(row[0])] = " · ".join([str(row[0]), *parts])
    return labels


_ASSOCIATION_KEY = "_rules_association_cache"


def association_data(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    cached = st.session_state.get(_ASSOCIATION_KEY)
    if cached is not None and cached[0] is frame:
        return cached[1]
    infos = infer_attributes(frame)
    result = (association_matrix(frame, infos=infos), redundant_pairs(frame, infos=infos))
    st.session_state[_ASSOCIATION_KEY] = (frame, result)
    return result


def edit_rule_table(name: str, rules: List[Any], to_row: Callable[[Any], Dict[str, Any]],
                    apply_row: Callable[[Any, Dict[str, Any]], None],
                    column_config: Dict[str, Any], disabled: Sequence[str]) -> Optional[str]:
    if not rules:
        st.caption("No rules of this type yet — add one below.")
        return None
    base = pd.DataFrame([{**to_row(r), "delete": False} for r in rules])
    edited = st.data_editor(
        base, key=k(f"{name}_table"), hide_index=True, width="stretch", num_rows="fixed",
        disabled=list(disabled),
        column_config={**column_config,
                       "delete": st.column_config.CheckboxColumn("🗑️", help="Tick to delete")},
    )
    doomed = [i for i in range(len(rules)) if bool(edited.iloc[i]["delete"])]
    if doomed:
        for i in sorted(doomed, reverse=True):
            del rules[i]
        return "deleted"
    changed = False
    for i, rule in enumerate(rules):
        old, new = base.iloc[i].to_dict(), edited.iloc[i].to_dict()
        if any(not same(old[c], new[c]) for c in old):
            apply_row(rule, new)
            changed = True
    return "changed" if changed else None


COMMON_RULE_COLUMNS = {
    "rule": st.column_config.TextColumn("Rule", width="large"),
    "label": st.column_config.TextColumn("Label"),
    "hard": st.column_config.CheckboxColumn("Hard", help="Hard rules must hold; soft rules "
                                                         "are traded off by weight"),
    "weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=100.0,
                                            step=0.5, format="%.1f"),
    "enabled": st.column_config.CheckboxColumn("On"),
}


def apply_common(rule: Any, row: Dict[str, Any]) -> None:
    rule.label = str(row.get("label") or "").strip()
    rule.hard = bool(row.get("hard"))
    rule.weight = max(0.0, num(row.get("weight"), rule.weight))
    rule.enabled = bool(row.get("enabled"))


st.title("🧭 Allocation Rules")
st.markdown("What a good allocation looks like: per-attribute goals, head-count quotas, "
            "no-lone rules, repeat-pairing limits and the solver settings. Changes apply "
            "immediately to the next run on **Team Formation**.")
render_flashes()

setup = engine_setup(config)
if setup is None:
    st.warning("No students loaded. Load a roster first.")
    page_link("data", "Go to Data Overview")
    st.stop()

frame = setup.frame
policy: Policy = setup.policy
seen = st.session_state.get(SEEN_KEY)
if seen is not None and seen != policy_fingerprint(config.policy):
    st.session_state[REV_KEY] = rev() + 1
infos = infer_attributes(frame)
kinds = {name: info.kind for name, info in infos.items()}
team_size = int(config.team_size_target or 5)
num_teams = int(config.num_teams or 0) or None
first_sizes = planned_sizes(setup.specs[0]) if setup.specs else []
generated_ids = [s.project_id for s in setup.specs]
categorical_attrs = [n for n, i in infos.items() if i.kind in ("categorical", "multi", "text")]
numeric_attrs = [n for n in infos if numeric_kind(n, infos)]

if config.algorithm != "engine":
    st.info("A classic algorithm is selected on **Configuration**. These rules still score "
            "every run (so runs stay comparable) but only the rules engine optimises them.")

main_col, side_col = st.columns([7, 3], gap="large")
with side_col:
    side_box = st.container()

with main_col:
    (tab_presets, tab_attrs, tab_counts, tab_iso, tab_numeric, tab_pairs, tab_history,
     tab_fair, tab_solver, tab_feas) = st.tabs([
        "🧩 Presets & files", "📊 Attributes", "🔢 Head counts", "🧍 No lone…",
        "📏 Numeric & shared", "🤝 Together / apart", "🔁 History & preferences",
        "⚖️ Fairness", "🛠️ Solver", "✅ Feasibility",
    ])

with tab_presets:
    st.markdown("#### Start from a preset")
    preset_names = list(PRESETS)
    chosen = st.selectbox("Preset", preset_names, format_func=lambda n: PRESETS[n].title,
                          key="rules_preset_choice")
    st.caption(PRESETS[chosen].description)
    pc1, pc2 = st.columns(2)
    if pc1.button("Apply preset to the current rules", key="rules_apply_preset", width="stretch",
                  help="Transform the rules below with this preset"):
        commit(apply_preset(policy, chosen, frame, team_size=team_size, num_teams=num_teams),
               f"Preset '{PRESETS[chosen].title}' applied.")
    if pc2.button("✨ Suggest from data", key="rules_suggest", width="stretch",
                  help="Start over: rules inferred from the roster, then this preset"):
        commit(suggest_policy(frame, preset=chosen, team_size=team_size, num_teams=num_teams),
               f"Rules suggested from the data ({PRESETS[chosen].title}).")

    st.markdown("#### Example rule sets")
    config_dir = get_config_dir()
    if not (config_dir / "policies").is_dir():
        config_dir = ROOT_DIR / "config"
    examples: Dict[str, Tuple[str, str]] = {}
    example_paths: Dict[str, Path] = {}
    for example_dir in extensions.example_policy_dirs(config_dir):
        for path in sorted(example_dir.glob("*.json")) if example_dir.is_dir() else []:
            if path.name in examples:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            examples[path.name] = (str(data.get("name") or path.stem),
                                   str(data.get("description") or "").split("\n")[0])
            example_paths[path.name] = path
    examples = dict(sorted(examples.items()))
    if examples:
        example = st.selectbox("Example", list(examples), format_func=lambda f: examples[f][0],
                               key="rules_example_choice")
        if examples[example][1]:
            st.caption(examples[example][1])
        if st.button("Load example", key="rules_load_example"):
            try:
                loaded = Policy.from_json(example_paths[example].read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                st.error(f"Could not load {example}: {exc}")
            else:
                commit(loaded, f"Example rules '{loaded.name}' loaded.")
    else:
        st.caption("No example rule sets found.")

    st.markdown("#### Import / export")
    upload = st.file_uploader("Rules JSON", type=["json"], key="rules_policy_upload",
                              help="A rules file downloaded from this page or written with the "
                                   "CLI, a CLI report (--report) or a configuration export: "
                                   "their rules are used.")
    if upload is not None and is_new_upload(POLICY_UPLOAD_SLOT, upload):
        mark_upload_processed(POLICY_UPLOAD_SLOT, upload)
        try:
            imported, origin = policy_from_upload(upload.getvalue())
        except ValueError as exc:
            st.error(f"Could not read rules from {upload.name}: {exc} The current rules were "
                     "kept.")
        else:
            notes = imported.validate(kinds)
            if notes:
                flash(f"The imported rules have {len(notes)} problem(s) to fix — see the "
                      "panel on the right.", "warning")
            commit(imported, f"Rules imported from {upload.name}{origin}.")
    st.caption("Download the current rules from the panel on the right.")

    st.markdown("#### Name & notes")
    new_name = st.text_input("Rule set name", value=policy.name, key=k("name"))
    new_description = st.text_area("Description", value=policy.description, key=k("description"),
                                   height=120)
    policy.name = new_name.strip() or "Custom"
    policy.description = new_description

with tab_attrs:
    head_col, help_col = st.columns([5, 1.3], vertical_alignment="bottom")
    head_col.markdown("#### Attribute goals")
    with help_col.popover("ℹ️ Goals explained", width="stretch"):
        st.markdown(GOAL_HELP)
    st.caption("One row per roster column. Pick a goal and a weight; untick **On** to switch a "
               "rule off without losing it. *Suggested* is what the data analysis recommends.")
    overview = attribute_overview(frame, infos, team_size=team_size)
    suggested = {row["name"]: row for row in overview}
    attr_rows = []
    for row in overview:
        name = row["name"]
        rule = policy.attribute_rule(name)
        kind = (rule.kind if rule is not None and rule.kind else row["kind"])
        values = row["stats"] if kind == "numeric" and row["stats"] else row["top_values"]
        attr_rows.append({
            "attribute": name,
            "kind": kind,
            "coverage": round(100.0 * float(row["coverage"]), 1),
            "values": values,
            "goal": rule.goal if rule is not None else "ignore",
            "weight": float(rule.weight) if rule is not None else 0.0,
            "enabled": bool(rule.enabled) if rule is not None else False,
            "suggested": (f"{row['suggested_goal']} · {row['suggested_weight']:g}"
                          if row["suggested_goal"] != "ignore" else suggested_ignore(name, infos)),
            "why": row["why"],
        })
    attr_base = pd.DataFrame(attr_rows)
    if attr_base.empty:
        st.info("The roster has no attribute columns.")
    else:
        kind_options = list(ATTRIBUTE_KINDS) + sorted(set(attr_base["kind"]) - set(ATTRIBUTE_KINDS))
        attr_edited = st.data_editor(
            attr_base, key=k("attr_editor"), hide_index=True, width="stretch", num_rows="fixed",
            disabled=["attribute", "coverage", "values", "suggested", "why"],
            column_config={
                "attribute": st.column_config.TextColumn("Attribute"),
                "kind": st.column_config.SelectboxColumn(
                    "Kind", options=kind_options, required=True,
                    help="categorical: one value per student · numeric: numbers (or an ordered "
                         "scale) · multi: several values separated by '|'"),
                "coverage": st.column_config.ProgressColumn("Coverage", min_value=0,
                                                            max_value=100, format="%.0f%%"),
                "values": st.column_config.TextColumn("Values", width="medium"),
                "goal": st.column_config.SelectboxColumn("Goal", options=ALL_GOALS, required=True),
                "weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=10.0,
                                                        step=0.5, format="%.1f"),
                "enabled": st.column_config.CheckboxColumn("On"),
                "suggested": st.column_config.TextColumn("Suggested"),
                "why": st.column_config.TextColumn("Why", width="large"),
            },
        )
        adjusted = False
        for i in range(len(attr_base)):
            old, new = attr_base.iloc[i].to_dict(), attr_edited.iloc[i].to_dict()
            if all(same(old[c], new[c]) for c in ("kind", "goal", "weight", "enabled")):
                continue
            name = old["attribute"]
            rule = policy.attribute_rule(name)
            if rule is None:
                rule = AttributeRule(attribute=name, goal="ignore", weight=0.0, enabled=False)
                policy.attributes.append(rule)
            inferred = infos[name].kind
            new_kind = str(new["kind"])
            rule.kind = None if new_kind == inferred or new_kind not in ATTRIBUTE_KINDS else new_kind
            goal = str(new["goal"])
            if not same(old["kind"], new["kind"]):
                mapping = TO_NUMERIC_GOAL if new_kind == "numeric" else TO_CATEGORICAL_GOAL
                if goal in mapping:
                    goal, adjusted = mapping[goal], True
            rule.goal = goal
            rule.weight = max(0.0, num(new["weight"], 0.0))
            rule.enabled = bool(new["enabled"])
            hint = suggested.get(name, {})
            if old["goal"] == "ignore" and goal != "ignore" and same(old["weight"], new["weight"]):
                rule.weight = rule.weight if rule.weight > 0 else max(
                    float(hint.get("suggested_weight") or 0.0), 1.0)
                rule.enabled, adjusted = True, True
            elif float(old["weight"] or 0) == 0 and rule.weight > 0 and goal == "ignore":
                fallback = hint.get("suggested_goal") or "balance"
                valid = NUMERIC_GOALS if (rule.kind or inferred) == "numeric" else CATEGORICAL_GOALS
                rule.goal = fallback if fallback in valid and fallback != "ignore" else "balance"
                rule.enabled, adjusted = True, True
            elif rule.weight > 0 and not same(old["weight"], new["weight"]) and not rule.enabled \
                    and same(old["enabled"], new["enabled"]):
                rule.enabled, adjusted = True, True
        if adjusted:
            commit(policy, structural=False)

    for note in data_notes(infos):
        st.caption(f"ℹ️ {note}")

    missing = [r.attribute for r in policy.attributes if r.attribute not in infos]
    if missing:
        st.warning(f"Rules refer to column(s) the loaded roster does not have: "
                   f"{', '.join(missing)}. The engine refuses to run until they are removed "
                   "(or the matching roster is loaded).")
        if st.button("Remove rules for missing columns", key=k("drop_missing")):
            policy.attributes = [r for r in policy.attributes if r.attribute in infos]
            commit(policy, f"Removed attribute rules for {', '.join(missing)}.")

    with st.expander("🔗 Redundant attributes & associations", expanded=False):
        matrix, redundant = association_data(frame)
        if redundant:
            st.markdown("Weighting two columns that carry the same information counts it twice:")
            for record in redundant[:12]:
                st.markdown(f"- **{record['a']} ↔ {record['b']}** ({record['relation']}, "
                            f"{record['strength']:.2f}): {record['explanation']}")
        else:
            st.caption("No strongly redundant attribute pairs found.")
        if matrix is not None and not matrix.empty and len(matrix) > 1:
            fig = px.imshow(matrix, zmin=0, zmax=1, color_continuous_scale="Blues",
                            text_auto=".2f", aspect="auto",
                            labels={"color": "association"})
            fig.update_layout(height=max(320, 32 * len(matrix)), margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch", key="rules_assoc_heatmap")
            st.caption("Cramér's V (categorical pairs), correlation ratio (categorical × numeric) "
                       "and |Pearson r| (numeric pairs); 1 = the same information.")

def count_row(rule: CountRule) -> Dict[str, Any]:
    return {
        "rule": describe_count_rule(rule), "label": rule.label, "mode": rule.mode,
        "min": float(rule.min_count) if rule.min_count is not None else float("nan"),
        "max": float(rule.max_count) if rule.max_count is not None else float("nan"),
        "hard": rule.hard, "weight": float(rule.weight), "enabled": rule.enabled,
    }


def apply_count_row(rule: CountRule, row: Dict[str, Any]) -> None:
    apply_common(rule, row)
    rule.mode = str(row.get("mode") or rule.mode)
    rule.min_count = opt_int(row.get("min"))
    rule.max_count = opt_int(row.get("max"))


with tab_counts:
    st.markdown("#### Head-count rules")
    st.caption("Per-team minimum/maximum numbers of students with certain values — "
               + COUNT_EXAMPLE)
    outcome = edit_rule_table(
        "counts", policy.counts, count_row, apply_count_row,
        {**COMMON_RULE_COLUMNS,
         "mode": st.column_config.SelectboxColumn(
             "Mode", options=["any_of", "each"], required=True,
             help="any_of: count students with any listed value together · each: the bounds "
                  "apply to every listed value (no values = every value)"),
         "min": st.column_config.NumberColumn("Min", min_value=0, step=1, format="%d"),
         "max": st.column_config.NumberColumn("Max", min_value=0, step=1, format="%d")},
        disabled=["rule"],
    )
    if outcome == "deleted":
        commit(policy, "Head-count rule deleted.")

    st.markdown("##### ➕ Add a head-count rule")
    attr_choices = categorical_attrs + numeric_attrs
    if not attr_choices:
        st.caption("No usable attributes.")
    else:
        attr = st.selectbox("Attribute", attr_choices, key=k("cr_attr"),
                            format_func=lambda n: f"{n} ({'numeric' if n in numeric_attrs else infos[n].kind})")
        is_numeric = attr in numeric_attrs
        c1, c2 = st.columns(2)
        values: List[str] = []
        lo = hi = None
        mode = "any_of"
        if is_numeric:
            stats = infos[attr].stats
            lo = c1.number_input("Range from (inclusive)", value=None, key=k(f"cr_lo_{attr}"),
                                 placeholder=f"min {fmt_g(stats.get('min'))}")
            hi = c2.number_input("Range to (inclusive)", value=None, key=k(f"cr_hi_{attr}"),
                                 placeholder=f"max {fmt_g(stats.get('max'))}")
        else:
            values = c1.multiselect("Values", infos[attr].values, key=k(f"cr_values_{attr}"),
                                    help="Leave empty with mode 'each' to cover every value")
            mode = c2.radio("Mode", ["any_of", "each"], key=k("cr_mode"), horizontal=True,
                            format_func=lambda m: "Any of these (counted together)"
                            if m == "any_of" else "Each value separately")
        c3, c4, c5, c6 = st.columns(4)
        min_count = c3.number_input("At least", min_value=0, value=None, step=1, key=k("cr_min"))
        max_count = c4.number_input("At most", min_value=0, value=None, step=1, key=k("cr_max"))
        weight = c5.number_input("Weight", min_value=0.0, value=5.0, step=0.5, key=k("cr_weight"))
        hard = c6.toggle("Hard", value=False, key=k("cr_hard"))
        label = st.text_input("Label (optional)", key=k("cr_label"))

        try:
            if is_numeric:
                targets = (count_rule_targets(frame, attr, team_size, team_sizes=first_sizes or None,
                                              numeric_range=(lo, hi))
                           if lo is not None or hi is not None else [])
            else:
                targets = count_rule_targets(frame, attr, team_size, team_sizes=first_sizes or None,
                                             values=values or None, top=None if values else 6)
        except Exception:
            targets = []
        for target in targets:
            st.caption(f"💡 {target['hint']}")

        if st.button("➕ Add head-count rule", key=k("cr_add"), type="primary"):
            new_rule = CountRule(
                attribute=attr, values=list(values), mode=mode,
                min_count=opt_int(min_count), max_count=opt_int(max_count),
                numeric_range=(opt_float(lo), opt_float(hi)) if is_numeric else None,
                hard=bool(hard), weight=float(weight), label=label.strip(),
            )
            problems = rule_problems(Policy(counts=[new_rule]), kinds)
            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                policy.counts.append(new_rule)
                commit(policy, f"Added: {describe_count_rule(new_rule)}")

def iso_row(rule: IsolationRule) -> Dict[str, Any]:
    return {"rule": describe_isolation_rule(rule), "label": rule.label,
            "min_together": int(rule.min_together), "hard": rule.hard,
            "weight": float(rule.weight), "enabled": rule.enabled}


def apply_iso_row(rule: IsolationRule, row: Dict[str, Any]) -> None:
    apply_common(rule, row)
    rule.min_together = max(2, opt_int(row.get("min_together")) or 2)


with tab_iso:
    st.markdown("#### No lone members")
    st.caption("If a team has any student with a protected value it must have at least *k* of "
               "them (zero is always fine) — " + ISOLATION_EXAMPLE)
    outcome = edit_rule_table(
        "isolation", policy.isolation, iso_row, apply_iso_row,
        {**COMMON_RULE_COLUMNS,
         "min_together": st.column_config.NumberColumn("At least together", min_value=2,
                                                       step=1, format="%d")},
        disabled=["rule"],
    )
    if outcome == "deleted":
        commit(policy, "No-lone rule deleted.")

    st.markdown("##### ➕ Add a no-lone rule")
    if not categorical_attrs:
        st.caption("No categorical attributes.")
    else:
        attr = st.selectbox("Attribute", categorical_attrs, key=k("iso_attr"))
        c1, c2 = st.columns([3, 1])
        values = c1.multiselect("Protected values (empty = every value)", infos[attr].values,
                                key=k(f"iso_values_{attr}"))
        min_together = c2.number_input("At least together", min_value=2, value=2, step=1,
                                       key=k("iso_min"))
        c3, c4, c5 = st.columns([1, 1, 2])
        weight = c3.number_input("Weight", min_value=0.0, value=5.0, step=0.5, key=k("iso_weight"))
        hard = c4.toggle("Hard", value=False, key=k("iso_hard"))
        label = c5.text_input("Label (optional)", key=k("iso_label"))
        counts_info = infos[attr].counts
        lonely = [v for v in (values or infos[attr].values) if counts_info.get(v, 0) < int(min_together)]
        if lonely:
            st.caption(f"⚠️ Fewer than {int(min_together)} students have "
                       f"{', '.join(lonely[:6])}{' …' if len(lonely) > 6 else ''}: "
                       "they will always be alone in their team.")
        if st.button("➕ Add no-lone rule", key=k("iso_add"), type="primary"):
            new_rule = IsolationRule(attribute=attr, values=list(values),
                                     min_together=int(min_together), hard=bool(hard),
                                     weight=float(weight), label=label.strip())
            problems = rule_problems(Policy(isolation=[new_rule]), kinds)
            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                policy.isolation.append(new_rule)
                commit(policy, f"Added: {describe_isolation_rule(new_rule)}")

def bounds_row(rule: NumericBoundsRule) -> Dict[str, Any]:
    nan = float("nan")
    return {"rule": describe_numeric_bounds_rule(rule), "label": rule.label,
            "mean_min": nan if rule.mean_min is None else float(rule.mean_min),
            "mean_max": nan if rule.mean_max is None else float(rule.mean_max),
            "max_range": nan if rule.max_range is None else float(rule.max_range),
            "hard": rule.hard, "weight": float(rule.weight), "enabled": rule.enabled}


def apply_bounds_row(rule: NumericBoundsRule, row: Dict[str, Any]) -> None:
    apply_common(rule, row)
    rule.mean_min = opt_float(row.get("mean_min"))
    rule.mean_max = opt_float(row.get("mean_max"))
    rule.max_range = opt_float(row.get("max_range"))


def shared_row(rule: SharedValueRule, kind: Optional[str] = None) -> Dict[str, Any]:
    return {"rule": describe_shared_rule(rule, kind), "label": rule.label,
            "min_shared": int(rule.min_shared), "hard": rule.hard,
            "weight": float(rule.weight), "enabled": rule.enabled}


def apply_shared_row(rule: SharedValueRule, row: Dict[str, Any]) -> None:
    apply_common(rule, row)
    rule.min_shared = max(1, opt_int(row.get("min_shared")) or 1)


with tab_numeric:
    st.markdown("#### Numeric bounds")
    st.caption(extensions.text(
        "rules.bounds_caption",
        "Per-team limits on a numeric attribute: the team average within a range "
        "(e.g. average coffees per day 1.5–2.5) and/or a maximum spread (e.g. coffees "
        "per day within 3)."))
    outcome = edit_rule_table(
        "bounds", policy.numeric_bounds, bounds_row, apply_bounds_row,
        {**COMMON_RULE_COLUMNS,
         "mean_min": st.column_config.NumberColumn("Mean ≥", format="%.2f"),
         "mean_max": st.column_config.NumberColumn("Mean ≤", format="%.2f"),
         "max_range": st.column_config.NumberColumn("Max range", min_value=0.0, format="%.2f")},
        disabled=["rule"],
    )
    if outcome == "deleted":
        commit(policy, "Numeric bounds rule deleted.")
    if not numeric_attrs:
        st.caption("The roster has no numeric attributes.")
    else:
        st.markdown("##### ➕ Add numeric bounds")
        attr = st.selectbox("Numeric attribute", numeric_attrs, key=k("nb_attr"))
        stats = infos[attr].stats
        if stats:
            st.caption(f"Cohort: mean {fmt_g(stats.get('mean'))}, sd {fmt_g(stats.get('std'))}, "
                       f"range {fmt_g(stats.get('min'))}–{fmt_g(stats.get('max'))}")
        c1, c2, c3 = st.columns(3)
        mean_min = c1.number_input("Team mean at least", value=None, key=k(f"nb_lo_{attr}"))
        mean_max = c2.number_input("Team mean at most", value=None, key=k(f"nb_hi_{attr}"))
        max_range = c3.number_input("Max range within a team", value=None, min_value=0.0,
                                    key=k(f"nb_range_{attr}"))
        c4, c5, c6 = st.columns([1, 1, 2])
        weight = c4.number_input("Weight", min_value=0.0, value=3.0, step=0.5, key=k("nb_weight"))
        hard = c5.toggle("Hard", value=False, key=k("nb_hard"))
        label = c6.text_input("Label (optional)", key=k("nb_label"))
        if st.button("➕ Add numeric bounds", key=k("nb_add"), type="primary"):
            new_rule = NumericBoundsRule(attribute=attr, mean_min=opt_float(mean_min),
                                         mean_max=opt_float(mean_max),
                                         max_range=opt_float(max_range), hard=bool(hard),
                                         weight=float(weight), label=label.strip())
            problems = rule_problems(Policy(numeric_bounds=[new_rule]), kinds)
            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                policy.numeric_bounds.append(new_rule)
                commit(policy, f"Added: {describe_numeric_bounds_rule(new_rule)}")

    st.markdown("---")
    st.markdown("#### Shared values")
    st.caption("Teammates share at least *k* values of an attribute — " + SHARED_EXAMPLE
               + " For a single-value attribute this means everyone in the team has the same value.")
    outcome = edit_rule_table(
        "shared", policy.shared, lambda r: shared_row(r, kinds.get(r.attribute)),
        apply_shared_row,
        {**COMMON_RULE_COLUMNS,
         "min_shared": st.column_config.NumberColumn("Shared values ≥", min_value=1, step=1,
                                                     format="%d")},
        disabled=["rule"],
    )
    if outcome == "deleted":
        commit(policy, "Shared-value rule deleted.")
    if categorical_attrs:
        st.markdown("##### ➕ Add a shared-value rule")
        multi_first = sorted(categorical_attrs, key=lambda n: infos[n].kind != "multi")
        attr = st.selectbox("Attribute", multi_first, key=k("sv_attr"),
                            format_func=lambda n: f"{n} ({infos[n].kind})")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        min_shared = c1.number_input("Shared values ≥", min_value=1, value=1, step=1,
                                     key=k("sv_min"))
        weight = c2.number_input("Weight", min_value=0.0, value=5.0, step=0.5, key=k("sv_weight"))
        hard = c3.toggle("Hard", value=False, key=k("sv_hard"))
        label = c4.text_input("Label (optional)", key=k("sv_label"))
        if st.button("➕ Add shared-value rule", key=k("sv_add"), type="primary"):
            new_rule = SharedValueRule(attribute=attr, min_shared=int(min_shared), hard=bool(hard),
                                       weight=float(weight), label=label.strip())
            problems = rule_problems(Policy(shared=[new_rule]), kinds)
            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                policy.shared.append(new_rule)
                commit(policy, f"Added: {describe_shared_rule(new_rule, kinds.get(new_rule.attribute))}")
    st.caption("Numeric-bounds and shared-value rules are optimised and reported like every "
               "other rule; hard ones cost 1000 points per unit of violation.")

labels = student_labels(frame)


def pair_row(rule: PairRule) -> Dict[str, Any]:
    return {"rule": describe_pair_rule(rule), "label": rule.label, "kind": rule.kind,
            "projects": ", ".join(rule.projects) if rule.projects else "all",
            "hard": rule.hard, "weight": float(rule.weight), "enabled": rule.enabled}


def apply_pair_row(rule: PairRule, row: Dict[str, Any]) -> None:
    apply_common(rule, row)
    rule.kind = str(row.get("kind") or rule.kind)


with tab_pairs:
    st.markdown("#### Keep together / keep apart")
    st.caption("Explicit groups: students who must work together (same team) or must not "
               "(all in different teams), optionally only in some projects.")
    outcome = edit_rule_table(
        "pairs", policy.pairs, pair_row, apply_pair_row,
        {**COMMON_RULE_COLUMNS,
         "kind": st.column_config.SelectboxColumn("Kind", options=["together", "apart"],
                                                  required=True),
         "projects": st.column_config.TextColumn("Projects")},
        disabled=["rule", "projects"],
    )
    if outcome == "deleted":
        commit(policy, "Pair rule deleted.")

    st.markdown("##### ➕ Add a group")
    chosen_students = st.multiselect("Students", list(labels), key=k("pr_students"),
                                     format_func=lambda sid: labels.get(sid, sid),
                                     placeholder="Search by id or attribute")
    c1, c2, c3 = st.columns([2, 1, 1])
    kind = c1.radio("Kind", ["together", "apart"], horizontal=True, key=k("pr_kind"),
                    format_func=lambda v: "Keep together" if v == "together" else "Keep apart")
    hard = c2.toggle("Hard", value=True, key=k("pr_hard"))
    weight = c3.number_input("Weight", min_value=0.0, value=10.0, step=0.5, key=k("pr_weight"))
    c4, c5 = st.columns(2)
    projects = c4.multiselect("Only in projects (empty = all)", generated_ids, key=k("pr_projects"))
    label = c5.text_input("Label (optional)", key=k("pr_label"))
    if st.button("➕ Add group", key=k("pr_add"), type="primary"):
        new_rule = PairRule(kind=kind, students=list(chosen_students), hard=bool(hard),
                            weight=float(weight), label=label.strip(),
                            projects=list(projects) or None)
        problems = rule_problems(Policy(pairs=[new_rule]), kinds)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            policy.pairs.append(new_rule)
            commit(policy, f"Added: {describe_pair_rule(new_rule)}")

with tab_history:
    h = policy.history
    st.markdown("#### Repeat pairings across projects")
    h_enabled = st.toggle("Use the project history (imported and earlier generated projects)",
                          value=h.enabled, key=k("h_enabled"))
    c1, c2 = st.columns(2)
    with c1:
        cap_on = st.checkbox("Cap how often a pair may share a team",
                             value=h.max_times_together is not None, key=k("h_cap_on"),
                             disabled=not h_enabled)
        max_times = st.number_input(
            "Max times together (including the project being formed)", min_value=1,
            max_value=20, value=int(h.max_times_together or 1), step=1, key=k("h_max"),
            disabled=not (h_enabled and cap_on), help="1 = never repeat a teammate")
        hard_cap = st.toggle("The cap is a hard rule", value=h.hard_cap, key=k("h_hard"),
                             disabled=not (h_enabled and cap_on))
        cap_weight = st.number_input("Points per over-cap pair (soft cap)", min_value=0.0,
                                     value=float(h.cap_weight), step=1.0, key=k("h_cap_weight"),
                                     disabled=not (h_enabled and cap_on) or hard_cap)
    with c2:
        repeat_weight = st.number_input(
            "Repeat penalty weight", min_value=0.0, max_value=100.0, value=float(h.repeat_weight),
            step=0.5, key=k("h_repeat"), disabled=not h_enabled,
            help="Soft pressure against meeting the same people again, below the cap")
        decay = st.slider("Weight of older projects (per project ago)", 0.0, 1.0,
                          value=float(h.decay), step=0.05, key=k("h_decay"),
                          disabled=not h_enabled,
                          help="1 = every earlier project counts fully; 0.5 halves each step back")
        contact_model = st.selectbox(
            "Contact model", ["team_size", "uniform"],
            index=0 if h.contact_model == "team_size" else 1, key=k("h_contact"),
            disabled=not h_enabled,
            format_func=lambda m: "Scale by group size" if m == "team_size" else "Every shared group counts 1",
            help="Meeting in a 20-person section counts less than in a 5-person team")
        reference_size = st.number_input("Reference team size", min_value=2, max_value=50,
                                         value=int(h.reference_team_size), step=1,
                                         key=k("h_ref_size"),
                                         disabled=not h_enabled or contact_model != "team_size")
    as_teams = st.toggle(
        "Treat large imported groups as teams", value=h.ignore_groups_larger_than is None,
        key=k("h_large_as_teams"), disabled=not h_enabled,
        help=extensions.text('rules.as_teams_help', 'On: every earlier group counts as a team — large sections too — so section-mates are kept apart as strictly as former teammates (the cap applies to them). Off: groups larger than the limit are sections; they only add a weak soft contact and do not count toward the cap, so some section-mates may be reunited (Team Formation shows how many).'))
    large_limit = st.number_input("Groups larger than this are sections", min_value=2,
                                  max_value=500, value=int(h.ignore_groups_larger_than or 12),
                                  step=1, key=k("h_large"), disabled=not h_enabled or as_teams)
    large_on = not as_teams
    h.enabled = bool(h_enabled)
    h.max_times_together = int(max_times) if cap_on else None
    h.hard_cap = bool(hard_cap)
    h.cap_weight = float(cap_weight)
    h.repeat_weight = float(repeat_weight)
    h.decay = float(decay)
    h.contact_model = contact_model
    h.reference_team_size = int(reference_size)
    h.ignore_groups_larger_than = int(large_limit) if large_on else None

    st.markdown("#### Student preferences")
    prefs = policy.preferences
    n_prefer = sum(1 for v in frame.get("prefer_teammates", []) if v)
    n_avoid = sum(1 for v in frame.get("avoid_teammates", []) if v)
    st.caption(f"{n_prefer} student(s) listed preferred teammates, {n_avoid} listed teammates "
               "to avoid (edit them on Data Overview).")
    use_prefs = st.toggle("Use student teammate requests", value=prefs.use_student_preferences,
                          key=k("p_use"))
    c1, c2, c3 = st.columns(3)
    prefer_weight = c1.number_input("Points per satisfied preference", min_value=0.0,
                                    value=float(prefs.prefer_weight), step=0.5,
                                    key=k("p_prefer"), disabled=not use_prefs)
    avoid_hard = c2.toggle("Avoid requests are hard", value=prefs.avoid_hard, key=k("p_avoid_hard"),
                           disabled=not use_prefs)
    avoid_weight = c3.number_input("Points per violated avoid request", min_value=0.0,
                                   value=float(prefs.avoid_weight), step=1.0,
                                   key=k("p_avoid"), disabled=not use_prefs or avoid_hard)
    prefs.use_student_preferences = bool(use_prefs)
    prefs.prefer_weight = float(prefer_weight)
    prefs.avoid_hard = bool(avoid_hard)
    prefs.avoid_weight = float(avoid_weight)

with tab_fair:
    st.markdown("#### Total vs worst team")
    worst = st.slider("Weight on the worst team", 0.0, 1.0,
                      value=float(policy.fairness.worst_team_weight), step=0.05,
                      key=k("fairness"))
    policy.fairness.worst_team_weight = float(worst)
    st.markdown(
        "The engine minimises **(1 − w) × sum of team penalties + w × number of teams × "
        "worst team penalty**.\n\n"
        "- **w = 0** — best total: most teams excellent, a few may be noticeably worse.\n"
        "- **w ≈ 0.25** (default) — good overall with a guard against outliers.\n"
        "- **w = 1** — pure maximin (CATME-style): the weakest team is as good as possible, "
        "even if the average suffers a little."
    )
    st.caption(f"Now: {worst:.0%} on the worst team, {1 - worst:.0%} on the total.")

with tab_solver:
    s = policy.solver
    st.markdown("#### Search")
    methods = ["anneal", "milp", "milp+anneal"]
    method = st.radio("Method", methods, index=methods.index(s.method) if s.method in methods else 0,
                      key=k("s_method"), horizontal=True)
    st.markdown(METHOD_HELP[method])
    c1, c2, c3 = st.columns(3)
    time_cap = hosting.max_time_limit()
    time_max, time_value = 3600.0, float(s.time_limit_s)
    if time_cap is not None:
        time_max = min(3600.0, float(time_cap))
        time_value = min(max(hosting.clamp_time_limit(s.time_limit_s), 1.0), time_max)
        if float(st.session_state.get(k("s_time"), time_value)) > time_max:
            st.session_state[k("s_time")] = time_max
    time_limit = c1.number_input("Time limit per project (s)", min_value=1.0, max_value=time_max,
                                 value=time_value, step=1.0, key=k("s_time"),
                                 help=(f"At most {time_max:g} s on this server."
                                       if time_cap is not None else None))
    scale_time = c1.checkbox("Scale the time limit with cohort size",
                             value=bool(s.scale_time_limit), key=k("s_scale_time"),
                             help="Above 200 students the cap grows as time limit × students "
                                  "/ 200 (10 s → 30 s for 600 students), so the automatic move "
                                  "budget is not cut short on large cohorts. Off = the time "
                                  "limit is the cap whatever the cohort size.")
    restarts = c2.number_input("Restarts", min_value=1, max_value=100, value=int(s.restarts),
                               step=1, key=k("s_restarts"),
                               help="Independent searches per project; the best one is kept")
    cpu_workers = os.cpu_count() or 1
    max_workers = hosting.clamp_workers(cpu_workers)
    if max_workers > 1:
        workers = c3.slider("Worker processes", 1, max_workers,
                            value=min(max(int(s.workers), 1), max_workers), key=k("s_workers"),
                            help="Run restarts in parallel; results stay identical for a "
                                 "fixed seed")
    else:
        workers = 1
        c3.caption("Worker processes: 1 (single-core machine)" if cpu_workers <= 1
                   else "Worker processes: 1 (the limit of this server)")
    c4, c5 = st.columns(2)
    random_seed = c4.checkbox("New random seed every run", value=s.seed is None,
                              key=k("s_seed_random"),
                              help="Off = the same inputs always give the same teams")
    seed = c4.number_input("Seed", min_value=0, value=int(s.seed if s.seed is not None else 42),
                           step=1, key=k("s_seed"), disabled=random_seed)
    auto_iterations = c5.checkbox("Automatic move budget", value=s.iterations is None,
                                  key=k("s_iter_auto"),
                                  help="Derived from the cohort and team count; untick to fix "
                                       "the number of annealing moves per restart")
    iterations = c5.number_input("Moves per restart", min_value=100, max_value=10_000_000,
                                 value=int(s.iterations or 20000), step=1000, key=k("s_iter"),
                                 disabled=auto_iterations)
    st.markdown("#### Options")
    c6, c7, c8 = st.columns(3)
    flexible = c6.toggle("Flexible team sizes", value=s.flexible_sizes, key=k("s_flex"),
                         help="Allow moves that change team sizes within each project's min/max")
    polish = c7.toggle("Polish with swaps", value=s.polish, key=k("s_polish"),
                       help="Finish every restart with a local descent: improving swaps "
                            "(and single moves with flexible team sizes) are applied as "
                            "soon as they are found, until none is left or the fixed "
                            "evaluation budget is spent")
    joint = c8.toggle("Optimise projects jointly", value=s.joint_projects, key=k("s_joint"),
                      help="Re-optimise generated projects against each other after the "
                           "sequential pass (slower; can trade early quality for fewer repeats)")
    mip_gap = st.number_input("MILP optimality gap", min_value=0.0, max_value=0.5,
                              value=float(s.mip_gap), step=0.005, format="%.3f",
                              key=k("s_gap"), disabled=method == "anneal",
                              help="Stop the MILP once the solution is within this relative gap")
    s.method = method
    s.time_limit_s = float(time_limit)
    s.scale_time_limit = bool(scale_time)
    s.restarts = int(restarts)
    s.workers = int(workers)
    s.seed = None if random_seed else int(seed)
    s.iterations = None if auto_iterations else int(iterations)
    s.flexible_sizes = bool(flexible)
    s.polish = bool(polish)
    s.joint_projects = bool(joint)
    s.mip_gap = float(mip_gap)
    n_projects = max(1, len(setup.specs))
    run_solver = es.capped_policy(policy).solver
    run_limit = float(run_solver.time_limit_s)
    caps = [effective_time_limit(run_solver, len(spec.student_ids)) for spec in setup.specs]
    worst = (sum(caps) or run_limit) * ((1 + JOINT_ROUNDS) if joint else 1)
    scaled = [cap for cap in caps if cap > run_limit + 1e-9]
    st.caption(f"Worst-case run time ≈ {worst:.0f} s for {n_projects} generated project(s)"
               + (f" including up to {JOINT_ROUNDS} joint rounds" if joint else "")
               + (f"; the cap grows to {max(scaled):.0f} s per project for this cohort size"
                  if scaled else "") + ".")

if policy.to_dict() != (config.policy or {}):
    es.save_policy(config, policy)
st.session_state[SEEN_KEY] = policy_fingerprint(config.policy)

problems = policy.validate(kinds)

with tab_feas:
    st.markdown("#### Feasibility check")
    st.caption("Counting arguments and a quick construction check whether the rules can hold "
               "for the configured projects — before you run anything.")
    thorough = st.button("🔍 Re-check now (thorough)", key="rules_feas_recheck",
                         help="Recompute with a longer repeat-pairing probe")
    report = (feasibility_report(setup, force=thorough, probe_time_s=1.5 if thorough else 0.3)
              if setup.specs else None)
    if report is None:
        st.warning("Every project is imported — there is nothing to generate. Raise the number "
                   "of projects on Configuration.")
    else:
        st.markdown(f"**{report.summary()}** · checked in {report.elapsed_s * 1000:.0f} ms")
        if not report.findings:
            st.success("✅ Every rule can be met for the configured projects.")
        areas: Dict[str, List[Any]] = {}
        for finding in report.findings:
            areas.setdefault(finding.area, []).append(finding)
        for area, findings in areas.items():
            st.markdown(f"**{AREA_LABELS.get(area, area.title())}**")
            for finding in findings:
                prefix = f"**{finding.rule_label}** — " if finding.rule_label else ""
                LEVEL_RENDER.get(finding.level, st.info)(prefix + finding.message)

with side_box:
    with st.container(border=True):
        st.markdown(f"### {policy.name}")
        st.caption(f"{active_rule_count(policy)} active rules · {policy.solver.method}, "
                   f"{policy.solver.time_limit_s:g} s per project")
        b1, b2 = st.columns(2)
        b1.download_button("⬇️ JSON", data=policy.to_json(), file_name=f"{slug(policy.name)}.json",
                           mime="application/json", key="rules_download", width="stretch",
                           help="Download these rules")
        if b2.button("↩️ Reset", key="rules_reset", width="stretch",
                     help="Replace the rules with the suggestion from the data"):
            commit(suggest_policy(frame, team_size=team_size, num_teams=num_teams),
                   "Rules reset to the suggestion from the data.")
        if problems:
            st.error("**Fix before running:**\n" + "\n".join(f"- {p}" for p in problems))
        else:
            st.success("Rules are valid.")
        if report is not None:
            if report.errors:
                st.error(f"**Feasibility:** {report.summary()}\n"
                         + "\n".join(f"- {f.message}" for f in report.errors[:3]))
            elif report.warnings:
                st.warning(f"**Feasibility:** {report.summary()}")
            else:
                st.caption(f"Feasibility: {report.summary()}")
    st.markdown("**In plain English**")
    st.markdown("\n".join(f"- {line}" for line in summary_lines(policy, kinds)))
    page_link("formation", "Run on Team Formation")
