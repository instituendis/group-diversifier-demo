import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app import hosting  # noqa: E402
from app.core import editor_ops as eo  # noqa: E402
from app.core import engine_session as es  # noqa: E402
from app.core import import_sync as isync  # noqa: E402
from app.core import run_activation as ra  # noqa: E402
from app.core import run_history as rh  # noqa: E402
from app.core.data_session import (  # noqa: E402
    chip,
    ensure_session_state,
    flash,
    get_data_loader,
    page_link,
    refresh_imports,
    render_flashes,
)
from app.core.ids import id_sort_key  # noqa: E402
from app.core.job_runner import WAITING_TEXT  # noqa: E402
from app.utils.headline import SECTION_MATES_HELP  # noqa: E402

st.set_page_config(
    page_title="Team Editor | Group Diversifier",
    page_icon="✏️",
    layout="wide",
)

config = ensure_session_state()

EDITOR_KEY = "team_editor"
SUGGESTIONS_KEY = "editor_suggestions"
SUGGEST_AUTO_KEY = "editor_sugg_auto"
CARDS_PER_ROW = 4
MAX_ISSUE_CHIPS = 4
SUGGESTION_COUNT = 10
_EPS = 1e-9

FOCUS_ALL = "Whole project"
FOCUS_WORST = "Worst team"
FOCUS_STUDENT = "One student"
OP_MOVE = "Move a student"
OP_SWAP = "Swap two students"
EDITED_SUFFIX = " (edited)"

@dataclass
class Ctx:
    config: Any
    students: List[Any]
    raw: Optional[pd.DataFrame]
    frame: pd.DataFrame
    policy: Any


def overall(report: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    value = (report or {}).get("overall", {}).get(key)
    return None if value is None else float(value)


def history_value(report: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    value = (report or {}).get("history", {}).get(key)
    return None if value is None else float(value)


def delta_or_none(current: Optional[float], original: Optional[float]) -> Optional[float]:
    if current is None or original is None or abs(current - original) < 1e-9:
        return None
    return current - original


def size_bounds(ctx: Ctx, editor: eo.EditSession, pid: str):
    eff = es.project_config(ctx.config, editor.project_index(pid))
    return int(eff.team_size_min), int(eff.team_size_max), int(eff.team_size_target)


def sanitize_choice(key: str, options: Sequence[Any]) -> None:
    if key in st.session_state and st.session_state[key] is not None \
            and st.session_state[key] not in options:
        st.session_state[key] = None


def default_run_label(editor: eo.EditSession) -> str:
    base = rh.get_run(editor.base_run_id)
    if base is None:
        return "Edited allocation"
    label = rh.display_label(base.label)
    label = label[:-len(EDITED_SUFFIX)] if label.endswith(EDITED_SUFFIX) else label
    return label + EDITED_SUFFIX


def run_totals(editor: eo.EditSession) -> Optional[Tuple[int, int]]:
    editable = editor.editable_project_ids()
    if not editable or not all(p in editor.reports for p in editable):
        return None
    summary = es.summarize_reports({p: editor.reports[p] for p in editable})
    return int(summary["hard_units"]), int(summary["repeated_pairs"])


def change(after: int, before: int) -> str:
    return f"{before} → {after}" if after != before else f"{after} (unchanged)"


def flash_edit(message: Optional[str], editor: eo.EditSession,
               before: Optional[Tuple[int, int]]) -> None:
    after = run_totals(editor)
    if message:
        if before is not None and after is not None and len(editor.editable_project_ids()) > 1:
            message += (f" · Run-wide: hard violations {change(after[0], before[0])}, "
                        f"repeat pairs {change(after[1], before[1])}.")
        flash(message, "success")
    if before is not None and after is not None and after[0] > before[0]:
        flash(f"⚠️ This change raised the hard-rule violations across the run from {before[0]} "
              f"to {after[0]} (for example repeat pairs in a later project). Undo it, or check "
              "the later projects.", "warning")


def stored_reports(record: rh.RunRecord, scoring: str,
                   roster: str) -> Optional[Dict[str, Dict[str, Any]]]:
    if record.reports and record.policy and eo.scoring_signature(record.policy) == scoring \
            and record.roster_signature == roster:
        return record.reports
    return None


def start_editor(ctx: Ctx, plain: List[Dict[str, Any]], fp: str, scoring: str,
                 project_id: Optional[str], roster: str) -> eo.EditSession:
    record = rh.active_run()
    if record is not None and eo.fingerprint(record.projects) == fp:
        return eo.EditSession.start(plain, project_id=project_id, base_run_id=record.run_id,
                                    reports=stored_reports(record, scoring, roster))
    alloc = es.active_allocation()
    stats = (alloc.formation_result.algorithm_stats or {}) if alloc is not None else {}
    base = rh.get_run(stats.get("base_run_id")) if stats.get("source") == "edited" else None
    if base is not None and base.projects:
        reports = stored_reports(base, scoring, roster)
        editor = eo.EditSession.start(base.projects, project_id=project_id,
                                      base_run_id=base.run_id, reports=reports)
        try:
            for entry in plain:
                pid = entry["project_id"]
                if not entry["imported"] and \
                        eo.fingerprint([entry]) != eo.fingerprint([editor.project(pid)]):
                    editor.set_teams(entry["teams"], project_id=pid,
                                     team_names=entry["team_names"],
                                     label="Unsaved edits restored from the session file")
        except eo.EditError:
            editor = None
        if editor is not None and refresh_imported_history(ctx, editor):
            reports = None
        if editor is not None and eo.fingerprint(editor.export_projects()) == fp:
            if reports is None:
                eo.rescore(editor, ctx.config, ctx.students, policy=ctx.policy, raw_df=ctx.raw)
            return editor
    return eo.EditSession.start(plain, project_id=project_id)


def refresh_imported_history(ctx: Ctx, editor: eo.EditSession) -> List[str]:
    refresh_imports({s.student_id for s in ctx.students}, int(ctx.config.num_projects))
    fresh = isync.refreshed_imports(editor.projects,
                                    st.session_state.get("imported_assignments"), ctx.students)
    editor.refresh_imported(fresh)
    return list(fresh)


def adopt_imported_history(ctx: Ctx, editor: eo.EditSession, refreshed: List[str]) -> None:
    commit(ctx, editor)
    st.info(f"{isync.changed_notice(refreshed)}, so the editor and the active allocation now "
            "use it: repairs keep the newcomers away from their earlier teammates.")


def sync_editor(ctx: Ctx, active_projects: Sequence[Any]) -> eo.EditSession:
    plain = es.projects_to_ids(active_projects)
    fp = eo.fingerprint(plain)
    scoring = eo.scoring_signature(ctx.policy)
    roster = eo.roster_signature(ctx.frame)
    editor: Optional[eo.EditSession] = st.session_state.get(EDITOR_KEY)
    refreshed: List[str] = []
    if editor is not None and editor.meta.get("roster") != roster:
        refreshed = refresh_imported_history(ctx, editor)
        if refreshed and editor.meta.get("fingerprint") != fp \
                and eo.fingerprint(editor.export_projects()) == fp:
            editor.meta["fingerprint"] = fp
    if editor is None or editor.meta.get("fingerprint") != fp:
        restarted = editor is not None
        previous = editor.project_id if editor is not None else None
        editor = start_editor(ctx, plain, fp, scoring, previous, roster)
        editor.meta.update(fingerprint=fp, scoring=scoring, roster=roster)
        st.session_state[EDITOR_KEY] = editor
        st.session_state.pop(SUGGESTIONS_KEY, None)
        st.session_state["editor_project"] = editor.project_id
        if restarted:
            st.info("The active allocation changed outside the editor (a new formation run or "
                    "a run activated on Compare Runs), so the editor restarted from it and "
                    "its undo history was cleared.")
        refreshed = refresh_imported_history(ctx, editor)
        if refreshed:
            with st.spinner("The imported history changed: scoring the allocation again…"):
                eo.rescore(editor, ctx.config, ctx.students, policy=ctx.policy, raw_df=ctx.raw)
            adopt_imported_history(ctx, editor, refreshed)
        return refresh_stale(ctx, editor)
    record = rh.active_run()
    if editor.dirty and record is not None and record.run_id != editor.base_run_id \
            and eo.fingerprint(record.projects) == fp:
        editor.mark_saved(record.run_id)
    if editor.meta.get("scoring") != scoring or editor.meta.get("roster") != roster:
        with st.spinner("The rules or the roster changed: scoring the allocation again…"):
            eo.rescore(editor, ctx.config, ctx.students, policy=ctx.policy, raw_df=ctx.raw)
        editor.meta.update(scoring=scoring, roster=roster)
        st.info("The allocation rules or the roster changed, so every score here (the "
                "original run's included) was recomputed with the current rules.")
        if refreshed:
            adopt_imported_history(ctx, editor, refreshed)
    return refresh_stale(ctx, editor)


def refresh_stale(ctx: Ctx, editor: eo.EditSession) -> eo.EditSession:
    if editor.stale:
        with st.spinner("Scoring the allocation…"):
            eo.refresh_reports(editor, ctx.config, ctx.students, policy=ctx.policy,
                               raw_df=ctx.raw)
    return editor


def commit(ctx: Ctx, editor: eo.EditSession) -> None:
    eo.refresh_reports(editor, ctx.config, ctx.students, policy=ctx.policy, raw_df=ctx.raw)
    legacy = es.projects_from_ids(editor.export_projects(), ctx.students)
    fp = eo.fingerprint(es.projects_to_ids(legacy))
    if fp == editor.meta.get("fingerprint"):
        return
    reports = {pid: editor.reports[pid] for pid in editor.editable_project_ids()
               if pid in editor.reports}
    ra.activate_edited_allocation(legacy, ctx.config, ctx.students, reports,
                                  policy=ctx.policy, base_run_id=editor.base_run_id,
                                  edits=editor.edit_count, unsaved=editor.dirty)
    editor.meta["fingerprint"] = fp


def run_edit(ctx: Ctx, action: Callable[[eo.EditSession], Any],
             success: Optional[Callable[[Any], Optional[str]]] = None) -> None:
    editor: Optional[eo.EditSession] = st.session_state.get(EDITOR_KEY)
    if editor is None:
        return
    before = run_totals(editor)
    try:
        result = action(editor)
    except eo.EditError as exc:
        flash(str(exc), "error")
        return
    try:
        commit(ctx, editor)
    except ValueError as exc:
        flash(f"The edit was applied but could not be scored: {exc}", "warning")
    flash_edit(success(result) if success is not None else None, editor, before)


def cb_apply_op(ctx: Ctx, op: Dict[str, Any], reset_key: str) -> None:
    def action(editor):
        editor.apply(op)
        st.session_state[reset_key] = None
        return editor.log[-1]["text"]
    run_edit(ctx, action, lambda text: text)


def cb_apply_suggestion(ctx: Ctx, suggestion: Dict[str, Any]) -> None:
    def action(editor):
        editor.apply_suggestion(suggestion)
        return suggestion
    run_edit(ctx, action, lambda s: f"Applied: {s['description']} ({s['delta']:+.2f})")


def cb_undo(ctx: Ctx) -> None:
    def action(editor):
        label = editor.undo()
        st.session_state["editor_project"] = editor.project_id
        return label
    run_edit(ctx, action, lambda label: f"Undid: {label}")


def cb_redo(ctx: Ctx) -> None:
    def action(editor):
        label = editor.redo()
        st.session_state["editor_project"] = editor.project_id
        return label
    run_edit(ctx, action, lambda label: f"Redid: {label}")


def cb_revert(ctx: Ctx) -> None:
    run_edit(ctx, lambda editor: editor.revert_all(),
             lambda changed: "Reverted every project to the original allocation (undo brings "
                             "your edits back)." if changed else None)


def cb_toggle_team_lock(ctx: Ctx, pid: str, team: int, key: str) -> None:
    def action(editor):
        if st.session_state.get(key):
            editor.lock_team(team, pid)
        else:
            editor.unlock_team(team, pid)
    run_edit(ctx, action)


def cb_set_student_locks(ctx: Ctx, pid: str, key: str) -> None:
    run_edit(ctx, lambda editor: editor.set_locked_students(st.session_state.get(key) or [], pid))


def cb_unlock_all(ctx: Ctx, pid: str) -> None:
    def action(editor):
        editor.set_locked_students([], pid)
        for t in editor.locked_team_indices(pid):
            editor.unlock_team(t, pid)
    run_edit(ctx, action, lambda _: "Every lock in this project was released.")


def cb_add_team(ctx: Ctx, pid: str) -> None:
    def action(editor):
        name = (st.session_state.get("editor_new_team_name") or "").strip() or None
        t = editor.add_team(name, pid)
        st.session_state["editor_new_team_name"] = ""
        return editor.display_names(pid)[t]
    run_edit(ctx, action, lambda name: f"Added empty team {name}; move students into it.")


def cb_remove_team(ctx: Ctx, pid: str) -> None:
    def action(editor):
        team = st.session_state.get("editor_remove_team")
        if team is None:
            raise eo.EditError("Choose an empty team to remove")
        name = editor.display_names(pid)[int(team)]
        editor.remove_team(int(team), pid)
        st.session_state["editor_remove_team"] = None
        return name
    run_edit(ctx, action, lambda name: f"Removed empty team {name}.")


def cb_place_student(ctx: Ctx, pid: str) -> None:
    def action(editor):
        sid = st.session_state.get("editor_place_student")
        team = st.session_state.get("editor_place_team")
        if sid is None or team is None:
            raise eo.EditError("Choose a new student and a team")
        editor.place_student(sid, team, pid)
        st.session_state["editor_place_student"] = None
        return editor.log[-1]["text"]
    run_edit(ctx, action, lambda text: text)


def cb_remove_departed(ctx: Ctx, pid: str, departed: List[str]) -> None:
    run_edit(ctx, lambda editor: editor.remove_students(departed, pid),
             lambda gone: f"Removed {len(gone)} departed student(s) from the project.")


def cb_save(ctx: Ctx) -> None:
    editor: Optional[eo.EditSession] = st.session_state.get(EDITOR_KEY)
    if editor is None:
        return
    label = (st.session_state.get("editor_save_label") or "").strip() or default_run_label(editor)
    try:
        eo.refresh_reports(editor, ctx.config, ctx.students, policy=ctx.policy, raw_df=ctx.raw)
        record = eo.edited_record(editor, label=label, config=ctx.config, policy=ctx.policy,
                                  notes=(st.session_state.get("editor_save_notes") or "").strip(),
                                  roster=eo.roster_signature(ctx.frame))
    except (eo.EditError, ValueError) as exc:
        flash(str(exc), "error")
        return
    stored, legacy = ra.save_edited_run(record, config=ctx.config, students=ctx.students,
                                        policy=ctx.policy)
    editor.mark_saved(stored.run_id)
    editor.meta["fingerprint"] = eo.fingerprint(es.projects_to_ids(legacy))
    st.session_state["editor_save_label"] = ""
    st.session_state["editor_save_notes"] = ""
    flash(f"Saved as run {stored.run_id} · {rh.display_label(stored.label)}. It is now the active run; "
          f"compare it with others on Compare Runs.", "success")


def render_headline(editor: eo.EditSession, pid: str, pm: eo.ProjectModel) -> None:
    report = editor.reports.get(pid)
    original = editor.original_reports.get(pid)
    objective = overall(report, "objective")
    objective = pm.objective if objective is None else objective
    quality = overall(report, "quality_index_soft")
    hard = overall(report, "hard_units")
    repeats = history_value(report, "repeated_pairs")
    before = next((p for p in editor.original if p["project_id"] == pid), None)
    moved = eo.allocation_similarity(before["teams"], editor.teams(pid))["n_moved"] \
        if before is not None else 0
    sections = report is not None and es.show_section_reunions({}, {pid: report})

    cols = st.columns(6 if sections else 5)
    d = delta_or_none(objective, overall(original, "objective"))
    cols[0].metric("Objective", eo.fmt_number(objective),
                   delta=None if d is None else f"{d:+.2f}", delta_color="inverse",
                   help="Total penalty of this project under the current rules (lower is "
                        "better). Deltas compare with the allocation the editor started from.")
    d = delta_or_none(quality, overall(original, "quality_index_soft"))
    cols[1].metric("Quality vs random", "–" if quality is None else f"{quality:.1f}%",
                   delta=None if d is None else f"{d:+.1f} pts",
                   help="How much better than random teams the soft rules are met (100% = "
                        "perfect, 0% = no better than chance).")
    d = delta_or_none(hard, overall(original, "hard_units"))
    cols[2].metric("Hard violations", eo.fmt_number(hard),
                   delta=None if d is None else f"{d:+.0f}", delta_color="inverse")
    d = delta_or_none(repeats, history_value(original, "repeated_pairs"))
    cols[3].metric("Repeat pairs", eo.fmt_number(repeats),
                   delta=None if d is None else f"{d:+.0f}", delta_color="inverse",
                   help="Teammate pairs who already worked together in an earlier project.")
    if sections:
        reunited = es.section_reunions({}, {pid: report})
        d = delta_or_none(reunited, None if original is None
                          else es.section_reunions({}, {pid: original}))
        cols[4].metric("Section-mates reunited", reunited,
                       delta=None if d is None else f"{d:+.0f}", delta_color="inverse",
                       help=SECTION_MATES_HELP)
    cols[-1].metric("Students moved", moved,
                   help="Students not in the team that corresponds to their original one.")

    editable = editor.editable_project_ids()
    if len(editable) > 1 and all(p in editor.reports for p in editable):
        now = es.summarize_reports({p: editor.reports[p] for p in editable})
        then = (es.summarize_reports({p: editor.original_reports[p] for p in editable})
                if all(p in editor.original_reports for p in editable) else None)
        st.markdown("**All generated projects together**")
        run_cols = st.columns(5)
        run_sections = es.show_section_reunions(now)
        d = None if then is None else delta_or_none(now["objective"], then["objective"])
        run_cols[0].metric("Run objective", eo.fmt_number(now["objective"]),
                           delta=None if d is None else f"{d:+.2f}", delta_color="inverse")
        d = None if then is None else delta_or_none(now["hard_units"], then["hard_units"])
        run_cols[1].metric("Run hard violations", now["hard_units"],
                           delta=None if d is None else f"{d:+.0f}", delta_color="inverse")
        d = None if then is None else delta_or_none(now["repeated_pairs"], then["repeated_pairs"])
        run_cols[2].metric("Run repeat pairs", now["repeated_pairs"],
                           delta=None if d is None else f"{d:+.0f}", delta_color="inverse",
                           help="Teammate pairs who already shared a team in an earlier project, "
                                "over every generated project.")
        if run_sections:
            d = None if then is None else delta_or_none(now["section_reunions"],
                                                        then["section_reunions"])
            run_cols[3].metric("Run section-mates reunited", now["section_reunions"],
                               delta=None if d is None else f"{d:+.0f}", delta_color="inverse",
                               help=SECTION_MATES_HELP)
        if then is not None and now["hard_units"] > then["hard_units"]:
            st.warning(f"⚠️ The edits raised the hard-rule violations across the run from "
                       f"{then['hard_units']} to {now['hard_units']} — an earlier project's "
                       "change can create repeat pairs in a later one. See Team Formation's "
                       "rule compliance, or undo.")


def render_preview(pv: Dict[str, Any]) -> None:
    delta = pv["delta"]
    if delta < -_EPS:
        st.success(f"**{pv['description']}** improves the project by {-delta:.2f} points.")
    elif delta > _EPS:
        st.warning(f"**{pv['description']}** makes the project worse by {delta:.2f} points.")
    else:
        st.info(f"**{pv['description']}** does not change the score.")
    c1, c2 = st.columns(2)
    c1.metric("Objective after", eo.fmt_number(pv["objective_after"]),
              delta=f"{delta:+.2f}" if abs(delta) > _EPS else None, delta_color="inverse")
    hard_delta = pv["hard_after"] - pv["hard_before"]
    c2.metric("Hard violations after", pv["hard_after"],
              delta=f"{hard_delta:+d}" if hard_delta else None, delta_color="inverse")
    run_wide = pv.get("run_wide")
    if run_wide:
        render_run_wide_preview(run_wide)
    rows = [{"Team": r["name"],
             "Penalty before": r["penalty_before"], "Penalty after": r["penalty_after"],
             "Change": r["change"], "Size": f"{r['size_before']} → {r['size_after']}"}
            for r in pv["teams"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.2f")
                                for k in ("Penalty before", "Penalty after", "Change")})
    if pv["reasons"]:
        st.markdown("**Why:** " + " · ".join(pv["reasons"]))
    for item in pv["violations_resolved"]:
        st.markdown(f"✅ resolves {'hard ' if item['hard'] else ''}issue — {item['text']}")
    for item in pv["violations_added"]:
        st.markdown(f"{'⛔' if item['hard'] else '⚠️'} new {'hard ' if item['hard'] else ''}"
                    f"issue — {item['text']}")
    for warning in pv["warnings"]:
        st.caption(f"⚠️ {warning}")


def render_run_wide_preview(run_wide: Dict[str, Any]) -> None:
    hard_before = int(run_wide.get("hard_units_before") or 0)
    hard_after = int(run_wide.get("hard_units_after") or 0)
    rep_before = int(run_wide.get("repeated_pairs_before") or 0)
    rep_after = int(run_wide.get("repeated_pairs_after") or 0)
    c1, c2 = st.columns(2)
    c1.metric("Run-wide hard violations after", hard_after,
              delta=f"{hard_after - hard_before:+d}" if hard_after != hard_before else None,
              delta_color="inverse", help="Over every generated project of the run.")
    c2.metric("Run-wide repeat pairs after", rep_after,
              delta=f"{rep_after - rep_before:+d}" if rep_after != rep_before else None,
              delta_color="inverse", help="Over every generated project of the run.")
    if hard_after > hard_before:
        st.warning(f"⚠️ This edit raises the hard-rule violations across the run from "
                   f"{hard_before} to {hard_after} (e.g. repeat pairs in a later project).")


def student_labels(editor: eo.EditSession, pid: str) -> Dict[str, str]:
    names = editor.display_names(pid)
    locked = editor.effective_locks(pid)
    return {sid: f"{sid} · {names[t]}{' 🔒' if sid in locked else ''}"
            for t, team in enumerate(editor.teams(pid)) for sid in team}


def render_move_swap(ctx: Ctx, editor: eo.EditSession, pid: str, pm: eo.ProjectModel) -> None:
    lo, hi, _ = size_bounds(ctx, editor, pid)
    labels = student_labels(editor, pid)
    sids = sorted(labels, key=id_sort_key)
    names = editor.display_names(pid)
    teams = editor.teams(pid)
    kind = st.radio("Edit", [OP_MOVE, OP_SWAP], horizontal=True, key="editor_op_kind",
                    label_visibility="collapsed")
    op, reset_key = None, ""
    if kind == OP_MOVE:
        team_ids = list(range(len(teams)))
        sanitize_choice("editor_move_student", sids)
        sanitize_choice("editor_move_target", team_ids)
        c1, c2 = st.columns(2)
        sid = c1.selectbox("Student", sids, index=None, placeholder="Choose a student",
                           key="editor_move_student", format_func=lambda s: labels.get(s, s))
        target = c2.selectbox("Move to", team_ids, index=None, placeholder="Choose a team",
                              key="editor_move_target",
                              format_func=lambda t: f"{names[t]} ({len(teams[t])} members)")
        if sid is not None and target is not None:
            op, reset_key = {"kind": "move", "student": sid, "to_team": int(target)}, \
                "editor_move_target"
        button_label = "Apply move"
    else:
        sanitize_choice("editor_swap_a", sids)
        sanitize_choice("editor_swap_b", sids)
        c1, c2 = st.columns(2)
        a = c1.selectbox("Student", sids, index=None, placeholder="Choose a student",
                         key="editor_swap_a", format_func=lambda s: labels.get(s, s))
        b = c2.selectbox("Swap with", sids, index=None, placeholder="Choose a student",
                         key="editor_swap_b", format_func=lambda s: labels.get(s, s))
        if a is not None and b is not None:
            op, reset_key = {"kind": "swap", "students": [a, b]}, "editor_swap_b"
        button_label = "Apply swap"

    preview = None
    if op is not None:
        try:
            preview = eo.preview(editor, ctx.frame, ctx.policy, op, size_bounds=(lo, hi), pm=pm)
        except eo.EditError as exc:
            st.warning(str(exc))
    if preview is not None:
        render_preview(preview)
    elif op is None:
        st.caption("Choose the student(s) and the preview shows the effect before anything "
                   "changes.")
    st.button(button_label, key="editor_apply_op", type="primary", disabled=preview is None,
              on_click=cb_apply_op, args=(ctx, op, reset_key))


def render_suggestions(ctx: Ctx, editor: eo.EditSession, pid: str, pm: eo.ProjectModel) -> None:
    lo, hi, _ = size_bounds(ctx, editor, pid)
    labels = student_labels(editor, pid)
    locked = editor.effective_locks(pid)
    free = sorted((s for s in labels if s not in locked and s in pm.index), key=id_sort_key)
    c1, c2, c3 = st.columns([2, 2, 1.4], vertical_alignment="bottom")
    focus = c1.radio("Focus", [FOCUS_ALL, FOCUS_WORST, FOCUS_STUDENT], horizontal=True,
                     key="editor_sugg_focus")
    student = None
    if focus == FOCUS_STUDENT:
        sanitize_choice("editor_sugg_student", free)
        student = c2.selectbox("Student", free, index=None, placeholder="Choose a student",
                               key="editor_sugg_student", format_func=lambda s: labels[s])
    allow_moves = c3.checkbox(f"Allow size changes ({lo}–{hi})", key="editor_sugg_moves",
                              help="Also propose moving one student to another team when both "
                                   "teams stay within the size limits of this project.")
    worst = pm.worst_team()
    signature = (pid, eo.fingerprint(editor.projects), tuple(sorted(locked.items())), focus,
                 student, allow_moves, editor.meta.get("scoring"))
    requested = st.button("💡 Suggest improvements", key="editor_suggest")
    if requested:
        st.session_state[SUGGEST_AUTO_KEY] = True
    cached = st.session_state.get(SUGGESTIONS_KEY)
    wanted = requested or (st.session_state.get(SUGGEST_AUTO_KEY)
                           and (cached is None or cached.get("signature") != signature))
    if focus == FOCUS_STUDENT and student is None:
        st.caption("Choose an unlocked student to see the best swaps involving them.")
        return
    if wanted:
        try:
            with st.spinner("Searching swaps and moves…"):
                items = eo.suggestions(
                    editor, ctx.frame, ctx.policy, k=SUGGESTION_COUNT,
                    focus_student=student,
                    focus_team=worst if focus == FOCUS_WORST else None,
                    allow_moves=allow_moves, size_bounds=(lo, hi))
        except eo.EditError as exc:
            st.warning(str(exc))
            return
        cached = {"signature": signature, "items": items}
        st.session_state[SUGGESTIONS_KEY] = cached
    if cached is None or cached.get("signature") != signature:
        st.caption("Suggestions are the best single swaps (and optional moves) by exact score "
                   "change; locked students and teams are never touched. They refresh "
                   "automatically after each edit once requested.")
        return
    items = cached["items"]
    if focus == FOCUS_WORST and worst is not None:
        st.caption(f"Focused on the worst team: **{pm.team_names[worst]}** "
                   f"(penalty {eo.fmt_number(pm.penalties[worst])}).")
    if not items:
        st.success("No single swap or move improves this project any further (with the current "
                   "locks and focus).")
        return
    for k, item in enumerate(items):
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 1.2, 1], vertical_alignment="center")
            c1.markdown(f"**{item['description']}**")
            if item["reasons"]:
                c1.caption(" · ".join(item["reasons"][:4]))
            c2.markdown(f"**{item['delta']:+.2f}**")
            c3.button("Apply", key=f"editor_sugg_apply_{k}", on_click=cb_apply_suggestion,
                      args=(ctx, item))


def render_reoptimise(ctx: Ctx, editor: eo.EditSession, pid: str) -> None:
    lo, hi, _ = size_bounds(ctx, editor, pid)
    locks = editor.effective_locks(pid)
    explicit = editor.explicit_locks(pid)
    locked_teams = editor.locked_team_indices(pid)
    st.markdown("**Polish** — apply the best improving swap again and again: few changes, "
                "nothing locked moves.")
    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    steps = c1.number_input("Up to this many changes", min_value=1, max_value=50, value=10,
                            key="editor_polish_steps")
    if c2.button("✨ Polish", key="editor_polish"):
        before = run_totals(editor)
        try:
            with st.spinner("Polishing…"):
                applied = eo.polish(editor, ctx.frame, ctx.policy, max_steps=int(steps),
                                    project_id=pid)
        except eo.EditError as exc:
            st.error(str(exc))
        else:
            if applied:
                commit(ctx, editor)
                flash_edit(f"Polished: {len(applied)} change(s), objective "
                           f"{sum(s['delta'] for s in applied):+.2f}.", editor, before)
                st.rerun()
            st.info("Nothing to polish: no single swap improves this project.")
    st.divider()
    st.markdown("**Re-optimise unlocked students** — a full engine search for this project. "
                "Locked students and every member of a locked team stay where they are; the "
                "number of teams and their sizes stay as they are now; anyone else may move. "
                "The result is applied only if it scores better.")
    st.caption(f"{len(locks)} student(s) pinned: {len(explicit)} locked individually, "
               f"{len(locked_teams)} locked team(s).")
    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    budget_max = min(300, hosting.max_time_limit() or 300)
    if int(st.session_state.get("editor_reopt_time", 0)) > budget_max:
        st.session_state["editor_reopt_time"] = budget_max
    budget = c1.number_input("Time budget (seconds)", min_value=1, max_value=budget_max,
                             value=min(10, budget_max), key="editor_reopt_time")
    if c2.button("⚙️ Re-optimise unlocked students", key="editor_reopt"):
        before = run_totals(editor)
        bar = st.progress(0.0, text="Starting…")

        def progress(fraction: float, best: float, message: str) -> None:
            best_text = f" · best {best:.2f}" if math.isfinite(float(best)) else ""
            bar.progress(min(1.0, max(0.0, float(fraction))), text=f"{message}{best_text}")

        try:
            outcome = eo.reoptimize_unlocked(editor, ctx.frame, ctx.policy,
                                             time_limit_s=float(budget), size_min=lo,
                                             size_max=hi, project_id=pid, progress=progress)
        except eo.EditError as exc:
            bar.empty()
            st.error(str(exc))
        else:
            bar.empty()
            if outcome.applied:
                commit(ctx, editor)
                flash_edit(f"Re-optimised: objective {outcome.delta:+.2f}, {len(outcome.moved)} "
                           f"student(s) changed team, {outcome.locked} stayed locked.", editor,
                           before)
                st.rerun()
            st.info(f"The search found nothing better than the current allocation "
                    f"({eo.fmt_number(outcome.objective_before)}); nothing was changed.")


def render_locks_and_teams(ctx: Ctx, editor: eo.EditSession, pid: str) -> None:
    labels = student_labels(editor, pid)
    sids = sorted(labels, key=id_sort_key)
    key = f"editor_locks_{pid}"
    st.session_state[key] = sorted(editor.explicit_locks(pid), key=id_sort_key)
    st.multiselect("Locked students", sids, key=key, format_func=lambda s: labels.get(s, s),
                   on_change=cb_set_student_locks, args=(ctx, pid, key),
                   help="Locked students keep their team: edits, suggestions and "
                        "re-optimisation never move them. Lock whole teams on their cards.")
    names = editor.display_names(pid)
    locked_teams = editor.locked_team_indices(pid)
    if locked_teams:
        st.caption("Locked teams: " + ", ".join(names[t] for t in locked_teams))
    st.button("Release every lock in this project", key="editor_unlock_all",
              disabled=not (locked_teams or editor.explicit_locks(pid)),
              on_click=cb_unlock_all, args=(ctx, pid))
    st.divider()
    lo, hi, _ = size_bounds(ctx, editor, pid)
    problems = editor.size_report(lo, hi, pid)
    if problems:
        for problem in problems:
            st.warning(problem)
    else:
        st.caption(f"Every team has {lo}–{hi} members, as configured for this project.")
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    c1.text_input("New team name (optional)", key="editor_new_team_name")
    c2.button("➕ Add empty team", key="editor_add_team", on_click=cb_add_team, args=(ctx, pid))
    empty = [t for t, team in enumerate(editor.teams(pid)) if not team]
    if empty:
        sanitize_choice("editor_remove_team", empty)
        c1, c2 = st.columns(2, vertical_alignment="bottom")
        c1.selectbox("Empty team", empty, index=None, placeholder="Choose a team",
                     key="editor_remove_team", format_func=lambda t: names[t])
        c2.button("🗑️ Remove empty team", key="editor_remove_team_btn",
                  on_click=cb_remove_team, args=(ctx, pid))


def render_roster_changes(ctx: Ctx, editor: eo.EditSession, pid: str,
                          diff: eo.RosterDiff, pool: Optional[List[str]]) -> None:
    if not diff.changed:
        st.success("Everyone who should take part in this project is placed, and nobody in it "
                   "has left the loaded roster.")
        return
    st.warning(f"The loaded roster and this project differ: {len(diff.added)} student(s) to "
               f"place, {len(diff.removed)} no longer in the roster or participant list.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**New (not placed yet)**")
        st.markdown(", ".join(diff.added) if diff.added else "—")
    with c2:
        st.markdown("**Departed (still in a team)**")
        st.markdown(", ".join(diff.removed) if diff.removed else "—")
    lo, hi, target = size_bounds(ctx, editor, pid)
    keep = st.checkbox("Keep the number of teams", value=True, key="editor_repair_keep")
    if st.button("🩹 Repair automatically", key="editor_repair", type="primary"):
        before = run_totals(editor)
        status = st.empty()

        def repair_progress(fraction: float, best: float, message: str) -> None:
            if message == WAITING_TEXT:
                status.caption(f"⏳ {message}")

        try:
            with st.spinner("Repairing with as few changes as possible…"):
                outcome = eo.repair(editor, ctx.frame, ctx.policy, pool=pool, size_min=lo,
                                    size_max=hi, size_target=target, keep_team_count=keep,
                                    project_id=pid, progress=repair_progress)
        except eo.RepairUnavailable as exc:
            st.warning(str(exc))
        except eo.EditError as exc:
            st.error(str(exc))
        else:
            commit(ctx, editor)
            flash_edit(f"Repaired: {len(outcome.added)} placed, {len(outcome.removed)} removed, "
                       f"{len(outcome.moved)} other student(s) moved.", editor, before)
            st.rerun()
    st.markdown("**Or fix it by hand**")
    names = editor.display_names(pid)
    if diff.added:
        team_ids = list(range(len(editor.teams(pid))))
        sanitize_choice("editor_place_student", diff.added)
        sanitize_choice("editor_place_team", team_ids)
        c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        c1.selectbox("New student", diff.added, index=None, placeholder="Choose a student",
                     key="editor_place_student")
        c2.selectbox("Place in", team_ids, index=None, placeholder="Choose a team",
                     key="editor_place_team",
                     format_func=lambda t: f"{names[t]} ({len(editor.teams(pid)[t])} members)")
        c3.button("Place", key="editor_place_btn", on_click=cb_place_student, args=(ctx, pid))
    departed = [s for s in diff.removed if editor.team_of(s, pid) is not None]
    if departed:
        st.button(f"Remove the {len(departed)} departed student(s) from their teams",
                  key="editor_remove_departed", on_click=cb_remove_departed,
                  args=(ctx, pid, departed))


def render_log(editor: eo.EditSession) -> None:
    if not editor.log:
        st.caption("No edits yet.")
        return
    rows = [{"Time": e["at"][11:], "Project": e["project_id"], "Action": e["action"],
             "Change": e["text"]} for e in reversed(editor.log)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_team_grid(ctx: Ctx, editor: eo.EditSession, pid: str, cards: List[Dict[str, Any]],
                     worst: Optional[int], attributes: List[str], editable: bool) -> None:
    chips = eo.chip_values(ctx.frame, attributes)
    locked = editor.effective_locks(pid) if editable else {}
    for start in range(0, len(cards), CARDS_PER_ROW):
        cols = st.columns(CARDS_PER_ROW)
        for col, card in zip(cols, cards[start:start + CARDS_PER_ROW]):
            t = card["index"]
            with col, st.container(border=True):
                plural = "s" if card["size"] != 1 else ""
                title = f"**{card['name']}** · {card['size']} member{plural}"
                if editable and editor.is_team_locked(t, pid):
                    title = "🔒 " + title
                st.markdown(title)
                penalty = card["penalty"]
                badges = []
                if penalty is None:
                    badges.append(chip("empty", "gray"))
                else:
                    color = "red" if card["hard_units"] else ("orange" if t == worst else "blue")
                    badges.append(chip(f"penalty {penalty:.2f}", color))
                    if t == worst:
                        badges.append(chip("worst team", "orange"))
                for issue in card["issues"][:MAX_ISSUE_CHIPS]:
                    badges.append(chip(("⛔ " if issue["hard"] else "") + issue["text"],
                                       "red" if issue["hard"] else "orange"))
                if len(card["issues"]) > MAX_ISSUE_CHIPS:
                    badges.append(chip(f"+{len(card['issues']) - MAX_ISSUE_CHIPS} more", "gray"))
                st.markdown(" ".join(badges))
                lines = []
                for sid in card["members"]:
                    prefix = "🔒 " if sid in locked else ""
                    if sid in chips:
                        values = " ".join(chip(v) for v in chips[sid])
                        lines.append(f"{prefix}`{sid}` {values}")
                    else:
                        lines.append(f"{prefix}`{sid}` {chip('not in roster', 'red')}")
                st.markdown("  \n".join(lines) if lines else "_no members_")
                if editable:
                    key = f"editor_lock_team_{pid}_{t}"
                    st.session_state[key] = editor.is_team_locked(t, pid)
                    st.toggle("Lock team", key=key, on_change=cb_toggle_team_lock,
                              args=(ctx, pid, t, key))


st.title("✏️ Team Editor")
st.caption("Fine-tune an allocation by hand: move and swap students with a live preview of the "
           "effect, lock what must stay, apply improvement suggestions, and save the result as "
           "a new run.")
render_flashes()

loader = get_data_loader()
students = loader.get_students()
raw_df = loader.get_dataframe()
if not students:
    st.warning("No student data is loaded.")
    page_link("data", "Load data on Data Overview", icon="📊", button_key="editor_go_data")
    st.stop()

alloc = es.active_allocation()
active_projects = alloc.projects if alloc is not None else []
if not active_projects:
    st.info("There is no active allocation to edit yet. Form teams on **Team Formation**, or "
            "load an allocation made elsewhere: on **Compare Runs**, evaluate an external "
            "allocation (CSV) and activate it.")
    c1, c2 = st.columns(2)
    with c1:
        page_link("formation", "Form teams", icon="🎯", button_key="editor_go_formation")
    with c2:
        page_link("compare", "Compare Runs → evaluate an external allocation", icon="🆚",
                  button_key="editor_go_compare")
    st.stop()

frame = es.roster_frame(students, raw_df)
policy = es.current_policy(config, frame)
rule_problems = policy.validate(es.attribute_kinds(frame))
if rule_problems:
    st.error("The allocation rules do not fit the loaded data, so edits cannot be scored: "
             + "; ".join(rule_problems))
    page_link("rules", "Fix the rules", icon="🧭", button_key="editor_go_rules")
    st.stop()

ctx = Ctx(config=config, students=students, raw=raw_df, frame=frame, policy=policy)
editor = sync_editor(ctx, active_projects)
if es.active_reports() and not es.same_scoring(es.rules_of_active_reports(), ctx.policy):
    st.info(f"Scores here use the current allocation rules; the other pages show "
            f"{es.rules_source_text()}, so their numbers can differ until you save an edit.")

project_ids = editor.project_ids()
if st.session_state.get("editor_project") not in project_ids:
    st.session_state["editor_project"] = editor.project_id
project_names = {p["project_id"]: p["name"] for p in editor.projects}


def project_label(pid: str) -> str:
    return f"{project_names[pid]} (imported, read-only)" if not editor.is_editable(pid) \
        else project_names[pid]


top = st.columns([2, 3], vertical_alignment="bottom")
pid = top[0].selectbox("Project", project_ids, key="editor_project", format_func=project_label)
editor.select(pid)
base = rh.get_run(editor.base_run_id)
with top[1]:
    if base is not None:
        status = f"Editing run **{base.run_id} · {rh.display_label(base.label)}**"
    else:
        status = "Editing the active allocation (not saved as a run yet)"
    if editor.dirty:
        count = f" ({editor.edit_count} so far)" if editor.edit_count else ""
        status += f" · **unsaved edits**{count}: Analytics and Export already show them; " \
                  f"save them as a run to compare or roll back"
    st.markdown(status)

editable = editor.is_editable(pid)
try:
    pm = eo.project_model(editor, frame, policy, pid)
except eo.EditError as exc:
    st.error(str(exc))
    st.stop()

if editable:
    render_headline(editor, pid, pm)
    bar = st.columns([1, 1, 1.4, 3, 1.6], vertical_alignment="bottom")
    bar[0].button("↶ Undo", key="editor_undo", disabled=not editor.can_undo,
                  on_click=cb_undo, args=(ctx,), width="stretch",
                  help=f"Undo: {editor.undo_stack[-1]['label']}" if editor.can_undo else None)
    bar[1].button("↷ Redo", key="editor_redo", disabled=not editor.can_redo,
                  on_click=cb_redo, args=(ctx,), width="stretch",
                  help=f"Redo: {editor.redo_stack[-1]['label']}" if editor.can_redo else None)
    bar[2].button("⟲ Revert all", key="editor_revert", on_click=cb_revert, args=(ctx,),
                  disabled=not editor.has_changes, width="stretch",
                  help="Back to the allocation the editor started from (undoable)")
    bar[3].text_input("Run label", key="editor_save_label",
                      placeholder=default_run_label(editor))
    bar[4].button("💾 Save as new run", key="editor_save", type="primary", on_click=cb_save,
                  args=(ctx,), width="stretch",
                  disabled=not editor.dirty and editor.base_run_id is not None,
                  help="Store the edited allocation (all projects, re-scored) in the run "
                       "history; it becomes the active run.")
    st.text_input("Notes for the saved run (optional)", key="editor_save_notes")

    participants = st.session_state.get("project_participants")
    pool = participants.pool_for(pid) if participants is not None else None
    diff = eo.roster_diff(editor, frame["student_id"].tolist(),
                          pool=list(pool) if pool is not None else None, project_id=pid)
    if diff.changed:
        st.warning(f"Roster change: {len(diff.added)} student(s) to place and "
                   f"{len(diff.removed)} to remove in this project — see **Roster changes**.")

    tabs = st.tabs(["🔀 Move / swap", "💡 Suggestions", "⚙️ Re-optimise", "🔒 Locks & teams",
                    "🩹 Roster changes", "📜 Edit log"])
    with tabs[0]:
        render_move_swap(ctx, editor, pid, pm)
    with tabs[1]:
        render_suggestions(ctx, editor, pid, pm)
    with tabs[2]:
        render_reoptimise(ctx, editor, pid)
    with tabs[3]:
        render_locks_and_teams(ctx, editor, pid)
    with tabs[4]:
        render_roster_changes(ctx, editor, pid, diff, list(pool) if pool is not None else None)
    with tabs[5]:
        render_log(editor)
else:
    st.info("This project is imported history: it cannot be edited, but it counts as pair "
            "history for every later project.")

st.subheader("Teams")
chip_options = eo.chip_attributes(frame)
if "editor_chip_attrs" not in st.session_state:
    st.session_state["editor_chip_attrs"] = eo.default_chip_attributes(frame, policy)
st.session_state["editor_chip_attrs"] = [a for a in st.session_state["editor_chip_attrs"]
                                         if a in chip_options]
chip_attrs = st.multiselect("Attributes shown next to each student (up to 4)", chip_options,
                            key="editor_chip_attrs", max_selections=4,
                            format_func=lambda a: a.replace("_", " ").title())
render_team_grid(ctx, editor, pid, eo.team_cards(pm), pm.worst_team(), chip_attrs, editable)
