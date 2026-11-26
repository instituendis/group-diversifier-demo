import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core import editor_ops as eo  # noqa: E402
from app.core import engine_session as es  # noqa: E402
from app.core import run_history as rh  # noqa: E402
from app.core.assignment_import import (  # noqa: E402
    CONTIGUITY_ERROR_PREFIX,
    parse_assignments_csv,
    validate_against_roster,
)
from app.core.ids import trailing_number_key  # noqa: E402
from app.core.data_session import (  # noqa: E402
    ensure_session_state,
    flash,
    get_data_loader,
    is_new_upload,
    mark_upload_processed,
    page_link,
    render_flashes,
)
from app.ui_kit import refuse_large_table, sample_file  # noqa: E402
from app.utils.serialization import json_safe  # noqa: E402
from app.utils.headline import SECTION_MATES_HELP  # noqa: E402

st.set_page_config(
    page_title="Compare Runs | Group Diversifier",
    page_icon="🆚",
    layout="wide",
)

config = ensure_session_state()

RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MAX_COMPARE = 4
NOTICE_NAMES = 5
IMPORT_SLOT = "compare_runs_import"
SELECTION_KEY = "compare_selection"
PENDING_KEY = "_compare_pending_select"
RESCORED_KEY = "_compare_rescored"
CONFIRM_DELETE_KEY = "compare_confirm_delete"
SOURCE_UPLOAD = "Upload a CSV"
SOURCE_REFERENCE = "Bundled sample reference groups"
SOURCE_LABELS = {"engine": "engine", "legacy": "legacy algorithm", "edited": "edited",
                 "imported": "imported", "evaluated": "evaluated (external)"}
NUMBERING_SAME = "same"
NUMBERING_CONTINUE = "continue"
NUMBERING_LABELS = {
    NUMBERING_SAME: "Real project numbers (e.g. a hand-made project 3; file projects that are "
                    "imported in this session are history only)",
    NUMBERING_CONTINUE: "The file continues after the imported projects (its project 1 is the "
                        "first new project)",
}
HEADLINE_METRICS = [
    ("quality", "Quality vs random", True),
    ("objective", "Objective", False),
    ("hard_units", "Hard violations", False),
    ("repeated_pairs", "Repeat pairs", False),
    ("section_reunions", "Section-mates reunited", False),
    ("worst_team_penalty", "Worst team penalty", False),
    ("prefer_satisfied", "Preferences satisfied", True),
]


@dataclass
class Ctx:
    config: Any
    students: List[Any]
    raw: Optional[pd.DataFrame]
    frame: Optional[pd.DataFrame]
    policy: Any


def external_allocation(raw: bytes, roster_ids: Iterable[str], *, imported: Any = None,
                        numbering: str = NUMBERING_SAME
                        ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    parsed, errors = parse_assignments_csv(raw)
    ids = sorted(parsed.teams_by_project, key=int)
    if errors and ids and ids != [str(i) for i in range(1, len(ids) + 1)]:
        errors = [e for e in errors if not e.startswith(CONTIGUITY_ERROR_PREFIX)]
    if errors:
        return [], [], errors
    validate_against_roster(parsed, {str(s) for s in roster_ids})
    warnings = list(parsed.warnings)
    imported_ids = list(imported.imported_project_ids) if imported is not None else []
    offset = len(imported_ids) if numbering == NUMBERING_CONTINUE else 0

    def entry(pid: str, teams: Dict[str, List[str]], is_history: bool) -> Dict[str, Any]:
        items = sorted(teams.items(), key=lambda kv: trailing_number_key(kv[0]))
        return {"project_id": pid, "name": f"Project {pid}", "imported": is_history,
                "teams": [list(m) for _, m in items], "team_names": [t for t, _ in items]}

    timeline: Dict[int, Dict[str, Any]] = {}
    for pid in imported_ids:
        timeline[int(pid)] = entry(str(pid), imported.teams_by_project.get(pid, {}), True)
    for pid in ids:
        teams = parsed.teams_by_project.get(pid, {})
        if not teams:
            warnings.append(f"Project {pid} of the file has no student in the roster; skipped")
            continue
        new_id = int(pid) + offset
        timeline[new_id] = entry(str(new_id), teams, str(new_id) in imported_ids)
    projects = [timeline[k] for k in sorted(timeline)]
    scored = [p for p in projects if not p["imported"]]
    if not scored:
        errors.append("Every project in the file is imported history in this session, so there "
                      "is nothing to score. Upload the next project(s) with their real numbers, "
                      "or choose 'the file continues after the imported projects'.")
    for project in scored:
        number = int(project["project_id"])
        known = sum(1 for k in timeline if 1 <= k < number)
        missing_n = (number - 1) - known
        if missing_n <= 0:
            continue
        first: List[str] = []
        for k in range(1, number):
            if len(first) >= NOTICE_NAMES:
                break
            if k not in timeline:
                first.append(str(k))
        if missing_n <= NOTICE_NAMES:
            warnings.append(f"Project {number} is scored without project(s) "
                            f"{', '.join(first)} as history: they are neither in the file "
                            f"nor imported on Team Formation.")
        else:
            warnings.append(f"Project {number} is scored without {missing_n} earlier project(s) "
                            f"as history (e.g. {', '.join(first)} …): they are neither in the "
                            f"file nor imported on Team Formation.")
    return projects, warnings, errors


def live_view(record: Optional[rh.RunRecord]) -> Optional[rh.RunRecord]:
    reports = es.active_reports()
    alloc = es.active_allocation()
    projects = alloc.projects if alloc is not None else []
    if record is None or not reports or not projects:
        return None
    summary = {**(record.summary or {}), **es.summarize_reports(reports)}
    return replace(record, projects=es.projects_to_ids(projects), reports=reports,
                   summary=summary, policy=es.rules_of_active_reports().to_dict())


def history_off(record: rh.RunRecord) -> bool:
    flags = [(r.get("history") or {}).get("enabled") for r in (record.reports or {}).values()]
    known = [bool(f) for f in flags if f is not None]
    if known:
        return not any(known)
    return ((record.policy or {}).get("history") or {}).get("enabled") is False


def headline(record: rh.RunRecord) -> Dict[str, Any]:
    m = dict(eo.run_metrics(record))
    m["history_off"] = history_off(record)
    if m["history_off"]:
        m["repeated_pairs"] = None
        m["section_reunions"] = None
    return m


def rescored(ctx: Ctx, record: rh.RunRecord) -> rh.RunRecord:
    cache: Dict[tuple, Dict[str, Any]] = st.session_state.setdefault(RESCORED_KEY, {})
    key = (record.run_id, eo.fingerprint(record.projects), eo.scoring_signature(ctx.policy),
           eo.roster_signature(ctx.frame))
    if key not in cache:
        cache[key] = eo.evaluate_plain(record.projects, ctx.config, ctx.students,
                                       policy=ctx.policy, raw_df=ctx.raw)
    reports = cache[key]
    summary = {**record.summary, **es.summarize_reports(reports)}
    return replace(record, reports=reports, summary=summary, policy=ctx.policy.to_dict())


def cb_activate(ctx: Ctx, run_id: str) -> None:
    record = rh.get_run(run_id)
    if record is None:
        flash("That run no longer exists.", "error")
        return
    if not record.projects:
        flash(f"Run {run_id} holds no projects, so there is nothing to activate.", "error")
        return
    try:
        warnings = eo.activate_run(record, ctx.config, ctx.students, raw_df=ctx.raw,
                                   policy=ctx.policy)
    except (ValueError, KeyError) as exc:
        flash(f"Could not activate run {run_id}: {exc}", "error")
        return
    flash(f"Run {rh.run_title(record)} is now the active allocation (Analytics, Export and the "
          f"Team Editor use it).", "success")
    for warning in warnings:
        flash(warning, "warning")


def cb_ask_delete(run_id: str) -> None:
    st.session_state[CONFIRM_DELETE_KEY] = run_id


def cb_confirm_delete(run_id: str) -> None:
    record = rh.get_run(run_id)
    st.session_state.pop(CONFIRM_DELETE_KEY, None)
    if record is not None and rh.delete_run(run_id):
        flash(f"Deleted run {rh.run_title(record)}.", "success")


def cb_cancel_delete() -> None:
    st.session_state.pop(CONFIRM_DELETE_KEY, None)


def runs_table(records: List[rh.RunRecord], active_id: Optional[str],
               live: Optional[rh.RunRecord] = None) -> None:
    unsaved = es.unsaved_edits()
    shown = [live if live is not None and r.run_id == live.run_id else r for r in records]
    metrics = [headline(record) for record in shown]
    sections = any(m["show_section_reunions"] for m in metrics)
    rows = []
    for record, m in zip(shown, metrics):
        active = ""
        if record.run_id == active_id:
            active = "● edited" if unsaved else "●"
        row = {
            "Run": record.run_id,
            "Label": rh.display_label(record.label),
            "Active": active,
            "Source": SOURCE_LABELS.get(record.source, record.source),
            "Created": (record.created_at or "").replace("T", " "),
            "Method": m["method"],
            "Quality vs random": m["quality"],
            "Objective": m["objective"],
            "Hard violations": m["hard_units"],
            "Repeat pairs": m["repeated_pairs"],
        }
        if sections:
            row["Section-mates reunited"] = (m["section_reunions"]
                                             if m["show_section_reunions"] else None)
        row.update({
            "Worst team": m["worst_team_penalty"],
            "Prefer satisfied": (f"{m['prefer_satisfied']}/{m['prefer_total']}"
                                 if m.get("prefer_total") else "–"),
            "Pinned": bool(record.pinned),
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    key = "compare_runs_table_" + hashlib.sha1(
        "|".join(r.run_id for r in records).encode()).hexdigest()[:10]
    edited = st.data_editor(
        table, key=key, hide_index=True, width="stretch",
        disabled=[c for c in table.columns if c not in ("Label", "Pinned")],
        column_config={
            "Label": st.column_config.TextColumn(help="Click to rename the run"),
            "Active": st.column_config.TextColumn(
                help="The allocation the other pages show; 'edited' = it has unsaved Team "
                     "Editor changes"),
            "Quality vs random": st.column_config.NumberColumn(
                format="%.1f%%", help="Soft-rule quality vs random teams (higher is better)"),
            "Objective": st.column_config.NumberColumn(
                format="%.2f", help="Total penalty, summed over generated projects (lower is "
                                    "better; runs scored with different rules are not "
                                    "comparable)"),
            "Repeat pairs": st.column_config.NumberColumn(
                help="Teammates in a generated project who already shared a team earlier "
                     "(imported history included) — the same number as on Home, Team "
                     "Formation, Analytics, Cross-Project and Export. Pairs that met twice "
                     "only within imported projects are not repeats. Empty for runs scored "
                     "with history switched off (nothing was counted)."),
            "Section-mates reunited": st.column_config.NumberColumn(
                help=SECTION_MATES_HELP + " Empty for runs without earlier projects or with "
                     "history switched off."),
            "Worst team": st.column_config.NumberColumn(format="%.2f",
                                                        help="Highest team penalty"),
            "Pinned": st.column_config.CheckboxColumn(
                help="Pinned runs are never dropped when the history is full"),
        },
    )
    if live is not None:
        saved = headline(next(r for r in records if r.run_id == live.run_id))
        now = next(m for r, m in zip(shown, metrics) if r.run_id == live.run_id)

        def pairs(m: Dict[str, Any]) -> str:
            return "history off" if m["history_off"] else f"{m['repeated_pairs']} repeat pairs"

        st.caption(
            f"✏️ {live.run_id} has unsaved Team Editor changes: its row shows the edited "
            f"allocation (saved: {pairs(saved)}, now: {pairs(now)}; objective saved "
            f"{eo.fmt_number(saved['objective'])}, now {eo.fmt_number(now['objective'])}; hard "
            f"violations saved {eo.fmt_number(saved['hard_units'])}, now "
            f"{eo.fmt_number(now['hard_units'])}), as on the other pages. Save it as a run in "
            f"the Team Editor to keep it.")
    off = [r.run_id for r, m in zip(shown, metrics) if m["history_off"]]
    if off:
        st.caption(f"History was switched off in the rules of run(s) {', '.join(off)}: no "
                   f"repeats were counted, so their **Repeat pairs** cell is empty (not 0, "
                   f"which would mean no repeats).")
    by_id = {r.run_id: r for r in records}
    for row in edited.to_dict("records"):
        record = by_id.get(row["Run"])
        if record is None:
            continue
        label = rh.display_label(str(row.get("Label") or "").strip())
        if label and label != rh.display_label(record.label):
            rh.rename_run(record.run_id, label)
        if bool(row.get("Pinned")) != bool(record.pinned):
            rh.set_pinned(record.run_id, bool(row.get("Pinned")))


def run_actions(ctx: Ctx, records: List[rh.RunRecord], active_id: Optional[str]) -> None:
    ids = [r.run_id for r in records]
    labels = {r.run_id: rh.run_title(r) for r in records}
    if st.session_state.get("compare_action_run") not in ids:
        st.session_state["compare_action_run"] = active_id if active_id in ids else ids[-1]
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    target = c1.selectbox("Run", ids, key="compare_action_run", format_func=labels.get)
    unsaved = es.unsaved_edits()
    c2.button("▶️ Activate", key="compare_activate", type="primary", width="stretch",
              disabled=target == active_id and not unsaved, on_click=cb_activate,
              args=(ctx, target), help="Load this run as the session's active allocation")
    c3.button("🗑️ Delete", key="compare_delete", width="stretch", on_click=cb_ask_delete,
              args=(target,))
    if unsaved:
        st.caption("⚠️ The Team Editor has unsaved edits of the active allocation; activating a "
                   "run replaces them. Save them as a run in the editor first to keep them.")
    pending = st.session_state.get(CONFIRM_DELETE_KEY)
    if pending in labels:
        st.warning(f"Delete run **{labels[pending]}**? This cannot be undone (export it first "
                   f"to keep a copy).")
        d1, d2, _ = st.columns([1, 1, 4])
        d1.button("Delete run", key="compare_confirm_delete_btn", type="primary",
                  on_click=cb_confirm_delete, args=(pending,))
        d2.button("Cancel", key="compare_cancel_delete", on_click=cb_cancel_delete)


def export_import(records: List[rh.RunRecord], selection: List[str]) -> None:
    by_id = {r.run_id: r for r in records}
    chosen = [by_id[r] for r in selection if r in by_id]
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    bundle = rh.export_runs(chosen) if chosen else {}
    c1.download_button(
        f"⬇️ Export {len(chosen)} selected run(s) (JSON)", key="compare_export",
        data=json.dumps(json_safe(bundle), indent=1),
        file_name="group_diversifier_runs.json", mime="application/json",
        disabled=not chosen, on_click="ignore",
        help="Selected runs with their allocations, rules and scores; import them in a later "
             "session to compare or roll back.")
    uploaded = c2.file_uploader("Import runs (JSON export)", type=["json"],
                                key="compare_import_upload")
    if uploaded is not None and is_new_upload(IMPORT_SLOT, uploaded):
        mark_upload_processed(IMPORT_SLOT, uploaded)
        try:
            imported = rh.import_runs(json.loads(uploaded.getvalue().decode("utf-8-sig")))
        except (ValueError, UnicodeDecodeError, TypeError) as exc:
            st.error(f"Could not import runs from {uploaded.name}: {exc}")
            return
        renamed: Dict[str, str] = {}
        for record in imported:
            old = record.run_id
            record.run_id = ""
            record.label = rh.display_label(record.label) or "Imported run"
            stored = rh.add_run(record, activate=False)
            renamed[old] = stored.run_id
        for record in imported:
            if record.parent_run_id in renamed:
                record.parent_run_id = renamed[record.parent_run_id]
        flash(f"Imported {len(imported)} run(s) from {uploaded.name}.", "success")
        st.rerun()


def metric_bars(records: List[rh.RunRecord]) -> None:
    metrics = [headline(r) for r in records]
    optional = {"prefer_satisfied": any(m.get("prefer_total") for m in metrics),
                "section_reunions": any(m.get("show_section_reunions") for m in metrics)}
    shown = [(key, title, up) for key, title, up in HEADLINE_METRICS
             if optional.get(key, True)]
    cols = 3
    rows = (len(shown) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.08, vertical_spacing=0.22,
                        subplot_titles=[f"{title} {'↑' if up else '↓'}"
                                        for _, title, up in shown])
    for i, (record, m) in enumerate(zip(records, metrics)):
        color = RUN_COLORS[i % len(RUN_COLORS)]
        for k, (key, title, _) in enumerate(shown):
            value = m.get(key)
            off = value is None and m["history_off"] and key in ("repeated_pairs",
                                                                 "section_reunions")
            label = "off" if off else eo.fmt_number(value, 1)
            hover = "history switched off" if off else eo.fmt_number(value)
            fig.add_trace(
                go.Bar(x=[record.run_id], y=[value if value is not None else 0.0],
                       name=rh.run_title(record), legendgroup=record.run_id,
                       showlegend=k == 0, marker_color=color,
                       text=[label], textposition="outside", cliponaxis=False,
                       hovertemplate=f"{rh.run_title(record)}<br>{title}: "
                                     f"{hover}<extra></extra>"),
                row=k // cols + 1, col=k % cols + 1)
    fig.update_layout(height=260 * rows + 60, barmode="group", bargap=0.35, barcornerradius=4,
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
    fig.update_yaxes(rangemode="tozero", showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                     zeroline=False)
    st.plotly_chart(fig, width="stretch", key="compare_metric_bars")
    note = (" Repeat pairs \"off\": history was switched off for that run, so nothing was "
            "counted." if any(m["history_off"] for m in metrics) else "")
    st.caption("↑ higher is better · ↓ lower is better. Objectives add up over the generated "
               "projects of a run." + note)


def comparison_tables(records: List[rh.RunRecord]) -> None:
    columns = [eo.run_column(r) for r in records]
    rules = pd.DataFrame(eo.compare_rules(records))
    st.markdown("**Rule compliance** — share of teams meeting each count / isolation / pair "
                "rule (rules matched by label)")
    if rules.empty:
        st.caption("None of these runs has team-level rules.")
    else:
        rules["Spread"] = rules[columns].max(axis=1) - rules[columns].min(axis=1)
        rules = rules.rename(columns={"rule": "Rule", "kind": "Kind", "hard": "Hard/soft"})
        st.dataframe(rules.sort_values("Spread", ascending=False), hide_index=True,
                     width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.0f%%")
                                    for c in columns + ["Spread"]})
    attrs = pd.DataFrame(eo.compare_attributes(records))
    st.markdown("**Attribute quality** — 100% = perfect for the goal, 0% = no better than "
                "random teams")
    if attrs.empty:
        st.caption("None of these runs has attribute rules.")
    else:
        attrs["Spread"] = attrs[columns].max(axis=1) - attrs[columns].min(axis=1)
        attrs = attrs.rename(columns={"rule": "Rule", "attribute": "Attribute", "goal": "Goal"})
        st.dataframe(attrs.sort_values("Spread", ascending=False), hide_index=True,
                     width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in columns + ["Spread"]})


def policy_differences(records: List[rh.RunRecord]) -> None:
    base = records[0]
    st.markdown(f"**Rule differences** (from {rh.run_title(base)})")
    for other in records[1:]:
        diffs = eo.policy_differences(base, other)
        if diffs is None:
            st.caption(f"{rh.run_title(other)}: the rules of one of the runs were not recorded.")
        elif not diffs:
            st.caption(f"{rh.run_title(other)}: same rules.")
        else:
            with st.expander(f"{rh.run_title(other)}: {len(diffs)} difference(s)"):
                st.markdown("\n".join(f"- {d}" for d in diffs))


def similarity(ctx: Ctx, records: List[rh.RunRecord]) -> None:
    base = records[0]
    common = eo.common_project_ids(records)
    st.markdown(f"**How alike are the allocations?** Each run against {rh.run_title(base)}, per "
                f"project both runs contain.")
    if not common:
        st.caption("These runs have no project in common.")
        return
    rows, details = [], {}
    for other in records[1:]:
        for pid in common:
            a, b = base.project(pid), other.project(pid)
            if a.get("imported") and b.get("imported"):
                continue
            sim = eo.compare_allocations(base, other, pid)
            choice = f"{other.run_id} · project {pid}"
            details[choice] = (other, pid, sim)
            rows.append({"Project": pid, "Run": rh.run_title(other),
                         "Teammate pairs shared": 100.0 * sim["pair_overlap"],
                         "Adjusted Rand index": sim["ari"],
                         "Students who changed team": sim["n_moved"],
                         f"Only in {base.run_id}": len(sim["only_a"]),
                         "Only in this run": len(sim["only_b"])})
    if not rows:
        st.caption("The common projects are imported history in every run.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Teammate pairs shared": st.column_config.NumberColumn(
            format="%.0f%%", help="Share of teammate pairs the two allocations have in common"),
        "Adjusted Rand index": st.column_config.NumberColumn(
            format="%.2f", help="1 = identical teams, about 0 = as alike as two random "
                                "allocations"),
        "Students who changed team": st.column_config.NumberColumn(
            help="Students not in the team matched to their team in the first run (teams are "
                 "matched by overlap, so renumbering alone does not count)"),
    })
    choices = list(details)
    if st.session_state.get("compare_affected_choice") not in choices:
        st.session_state["compare_affected_choice"] = next(
            (c for c in choices if details[c][2]["n_moved"]), choices[0])
    choice = st.selectbox("Students affected", choices, key="compare_affected_choice")
    other, pid, sim = details[choice]
    if not sim["moved"] and not sim["only_a"] and not sim["only_b"]:
        st.caption("The same teams in both runs.")
        return
    attributes = eo.default_chip_attributes(ctx.frame, ctx.policy) if ctx.frame is not None \
        else []
    chips = eo.chip_values(ctx.frame, attributes) if ctx.frame is not None else {}
    affected = []
    for item in sim["moved"]:
        row = {"Student": item["student"], f"Team in {base.run_id}": item["team_a"],
               f"Team in {other.run_id}": item["team_b"]}
        for attr, value in zip(attributes, chips.get(item["student"], [])):
            row[attr.replace("_", " ").title()] = value
        affected.append(row)
    if affected:
        st.dataframe(pd.DataFrame(affected), hide_index=True, width="stretch")
    if sim["only_a"] or sim["only_b"]:
        st.caption(f"Only in {base.run_id}: {', '.join(sim['only_a']) or '—'} · only in "
                   f"{other.run_id}: {', '.join(sim['only_b']) or '—'}")


def external_evaluation(ctx: Ctx) -> None:
    st.subheader("Evaluate an external allocation")
    st.markdown(
        "Score an allocation made elsewhere (by hand, last year's groups, another tool) with "
        "the **current rules**, so it sits next to the engine's runs on the same yardstick. "
        "Use the long format of the Team Formation import: one row per student with "
        "`team_id, project_id, student_id`. Project numbers can be the real ones — a file "
        "with only project 3 is fine.")
    with st.expander("How history is counted"):
        st.markdown(
            "- Each project is scored against **every project before it**: the projects "
            "imported on Team Formation plus the file's own earlier projects (repeat pairs, "
            "history caps), exactly as the engine scores a run.\n"
            "- Project numbers are kept, so the evaluated run lines up with the same projects "
            "of the session's other runs (e.g. a hand-made project 3 next to the engine's "
            "project 3).\n"
            "- With **real project numbers**, a file project whose number is imported in this "
            "session stands in for it (history only, not scored). When the file **continues "
            "after** the imported projects, its project 1 becomes the first new project.\n"
            "- An earlier project that is neither in the file nor imported cannot count as "
            "history; the page says so.\n"
            "- Students missing from the loaded roster are left out (with a warning).")
    if ctx.frame is None:
        st.warning("Load student data first: the allocation is scored against the roster.")
        return
    reference = sample_file("reference_groups")
    sources = [SOURCE_UPLOAD] + ([SOURCE_REFERENCE] if reference is not None else [])
    source = st.radio("Allocation", sources, key="compare_ext_source", horizontal=True)
    raw_bytes: Optional[bytes] = None
    default_label = "External allocation"
    if source == SOURCE_UPLOAD:
        uploaded = st.file_uploader("Allocation CSV (team_id, project_id, student_id)",
                                    type=["csv", "txt"], key="compare_ext_upload")
        if uploaded is not None and not refuse_large_table(uploaded):
            raw_bytes = uploaded.getvalue()
            default_label = Path(uploaded.name).stem
    else:
        raw_bytes = reference[0].read_bytes()
        default_label = "Sample reference groups"
    imported = st.session_state.get("imported_assignments")
    numbering = NUMBERING_SAME
    if imported is not None and imported.project_count:
        numbering = st.radio(
            f"Project numbers in the file ({imported.project_count} project(s) are imported in "
            f"this session)", list(NUMBERING_LABELS), key="compare_ext_numbering",
            format_func=NUMBERING_LABELS.get)
    label = st.text_input("Run label", key="compare_ext_label", placeholder=default_label)
    if raw_bytes is None:
        return
    plain, warnings, errors = external_allocation(
        raw_bytes, ctx.frame["student_id"].tolist(), imported=imported, numbering=numbering)
    for error in errors:
        st.error(error)
    if errors:
        return
    overview = [{"Project": p["project_id"], "Role": "history" if p["imported"] else "scored",
                 "Teams": len(p["teams"]), "Students": sum(len(t) for t in p["teams"])}
                for p in plain]
    st.dataframe(pd.DataFrame(overview), hide_index=True, width="content")
    for warning in warnings:
        st.caption(f"⚠️ {warning}")
    if st.button("📏 Evaluate with the current rules and add as a run", key="compare_ext_evaluate",
                 type="primary"):
        try:
            with st.spinner("Scoring the allocation…"):
                record = eo.evaluated_record(plain, ctx.config, ctx.students, policy=ctx.policy,
                                             label=label.strip() or default_label,
                                             raw_df=ctx.raw)
        except (eo.EditError, ValueError) as exc:
            st.error(f"Could not evaluate the allocation: {exc}")
            return
        stored = rh.add_run(record, activate=False)
        st.session_state[PENDING_KEY] = stored.run_id
        flash(f"Added run {rh.run_title(stored)}: quality vs random "
              f"{eo.fmt_number(stored.summary.get('quality_index_soft'), 1)}%, objective "
              f"{eo.fmt_number(stored.summary.get('objective'))}. Activate it to open it in the "
              f"Team Editor, Analytics or Export.", "success")
        st.rerun()


st.title("🆚 Compare Runs")
st.caption("Every allocation of this session — formation runs, saved edits, evaluated external "
           "allocations and imported runs — side by side. Activate one to make it the "
           "allocation the other pages use.")
render_flashes()

loader = get_data_loader()
students = loader.get_students()
raw_df = loader.get_dataframe()
frame = es.roster_frame(students, raw_df) if students else None
policy = es.current_policy(config, frame) if frame is not None else None
ctx = Ctx(config=config, students=students, raw=raw_df, frame=frame, policy=policy)

records = rh.runs()
active_run = rh.active_run()
active_id = active_run.run_id if active_run is not None else None

if not records:
    st.info("No runs yet. A run is recorded whenever teams are formed on **Team Formation**, "
            "an edit is saved in the **Team Editor**, an external allocation is evaluated "
            "below, or runs are imported from a JSON export.")
    c1, c2 = st.columns(2)
    with c1:
        page_link("formation", "Form teams", icon="🎯", button_key="compare_go_formation")
    with c2:
        page_link("editor", "Open the Team Editor", icon="✏️", button_key="compare_go_editor")
    export_import(records, [])
else:
    st.subheader("Runs")
    live = None
    if es.unsaved_edits():
        live = live_view(next((r for r in records if r.run_id == active_id), None))
    runs_table(records, active_id, live)
    records = rh.runs()
    run_actions(ctx, records, active_id)

    ids = [r.run_id for r in records]
    labels = {r.run_id: rh.run_title(r) for r in records}
    if SELECTION_KEY not in st.session_state:
        default = [ids[-1]]
        if active_id in ids and active_id != ids[-1]:
            default.insert(0, active_id)
        elif len(ids) > 1:
            default.insert(0, ids[-2])
        st.session_state[SELECTION_KEY] = default
    selection = [r for r in st.session_state[SELECTION_KEY] if r in ids]
    pending = st.session_state.pop(PENDING_KEY, None)
    if pending in ids and pending not in selection:
        selection = (selection + [pending])[-MAX_COMPARE:]
    st.session_state[SELECTION_KEY] = selection
    selection = st.multiselect("Selected runs (compare 2–4, or export any number)", ids,
                               key=SELECTION_KEY, format_func=labels.get)
    export_import(records, selection)

    st.subheader("Comparison")
    by_id = {r.run_id: r for r in records}
    if live is not None and live.run_id in by_id:
        by_id[live.run_id] = replace(live, label=by_id[live.run_id].label,
                                     pinned=by_id[live.run_id].pinned)
    chosen = [by_id[r] for r in selection][:MAX_COMPARE]
    if len(selection) > MAX_COMPARE:
        st.warning(f"Comparing the first {MAX_COMPARE} selected runs.")
    if len(chosen) < 2:
        st.info("Select two to four runs above to compare them.")
    else:
        recorded = list(chosen)
        if live is not None and live.run_id in selection[:MAX_COMPARE]:
            st.caption(f"✏️ {live.run_id} is compared as the edited allocation (its unsaved Team "
                       f"Editor changes), like its row in the runs table.")
        same_yardstick = st.toggle(
            "Score every compared run with the current rules", key="compare_rescore",
            disabled=policy is None,
            help="Runs keep the scores of the rules they were made with. Re-scoring them all "
                 "with today's rules compares the allocations themselves.")
        if same_yardstick and policy is not None:
            try:
                with st.spinner("Scoring the runs with the current rules…"):
                    chosen = [rescored(ctx, r) for r in chosen]
            except ValueError as exc:
                st.error(f"Could not re-score the runs: {exc}")
        else:
            signatures = {eo.scoring_signature(r.policy) if r.policy else None for r in chosen}
            if len(signatures) > 1:
                st.warning("These runs were scored with different rules, so their scores are "
                           "not directly comparable. Turn on scoring with the current rules "
                           "to compare like with like.")
        metric_bars(chosen)
        comparison_tables(chosen)
        policy_differences(recorded)
        similarity(ctx, chosen)

st.divider()
external_evaluation(ctx)
