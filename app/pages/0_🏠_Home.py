import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core import engine_session as es  # noqa: E402
from app.core import run_history as rh  # noqa: E402
from app.core.data_session import (  # noqa: E402
    active_rule_count,
    engine_setup,
    ensure_session_state,
    feasibility_report,
    get_active_source,
    page_exists,
    page_link,
    planned_sizes,
    render_flashes,
)
from app.engine.roster import infer_attributes  # noqa: E402
from app.engine.sizing import describe_sizes  # noqa: E402
from app.ui_kit import demo_notice  # noqa: E402
from app.pages._views import render_headline_row  # noqa: E402
from app.utils.headline import headline_rows  # noqa: E402

st.set_page_config(page_title="Home | Group Diversifier", page_icon="🏠", layout="wide")

config = ensure_session_state()

STATUS_ICONS = {"done": "✅", "todo": "⬜", "warning": "⚠️", "error": "❌", "info": "ℹ️"}
SOURCE_LABELS = {"engine": "Rules engine", "legacy": "Classic", "edited": "Edited",
                 "imported": "Imported", "evaluated": "Evaluated"}


def step(number: int, title: str, status: str, detail: str, links: List[tuple],
         extra: Optional[List[str]] = None) -> None:
    with st.container(border=True):
        icon_col, text_col, link_col = st.columns([0.5, 5, 2.2], vertical_alignment="center")
        icon_col.markdown(f"### {STATUS_ICONS.get(status, '⬜')}")
        with text_col:
            st.markdown(f"**{number}. {title}**  \n{detail}")
            for line in extra or []:
                st.caption(line)
        with link_col:
            for key, label in links:
                page_link(key, label)


def ago(timestamp: str) -> str:
    try:
        seconds = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
    except (TypeError, ValueError):
        return ""
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f} h ago"
    return timestamp[:10]


st.title("🎯 Group Diversifier")
st.markdown("*Diverse, balanced project teams across a series of projects — "
            "without repeat pairings.*")
render_flashes()
demo_notice()

setup = engine_setup(config)
history = rh.runs()
active = rh.active_run()
reports = es.active_reports()
alloc = es.active_allocation()
unsaved = alloc is not None and alloc.unsaved_edits

st.subheader("🧭 Workflow")

if setup is None:
    step(1, "Load student data", "todo",
         "No students loaded yet — upload a CSV or Excel roster.",
         [("data", "Data Overview")])
    st.stop()

infos = infer_attributes(setup.frame)
usable = [name for name, info in infos.items() if info.usable]
n_students = len(setup.students)
step(1, "Load student data", "done",
     f"**{n_students} students** · {len(usable)} usable attributes "
     f"({', '.join(usable[:6])}{' …' if len(usable) > 6 else ''})",
     [("data", "Data Overview")],
     [f"Source: {get_active_source()}"])

size_lines: List[str] = []
size_status = "done"
for spec in setup.specs:
    sizes = planned_sizes(spec)
    line = f"Project {spec.project_id}: {len(spec.student_ids)} students → {describe_sizes(sizes)}"
    if sizes and (min(sizes) < spec.size_min or max(sizes) > spec.size_max):
        size_status = "warning"
        line += f" (outside {spec.size_min}–{spec.size_max})"
    size_lines.append(line)
if setup.imported_count:
    size_lines.insert(0, f"Projects 1–{setup.imported_count} imported (locked history)")
if not setup.specs:
    size_status = "error"
    size_lines.append(f"Every project is imported — raise the number of projects above "
                      f"{setup.imported_count} to generate new ones.")
if config.per_project_team_sizes:
    size_text = "team sizes set per project"
elif int(config.num_teams or 0) > 0:
    size_text = f"{int(config.num_teams)} teams per project"
else:
    size_text = (f"teams of {config.team_size_min}–{config.team_size_max} "
                 f"(target {config.team_size_target})")
step(2, "Team sizes & projects", size_status,
     f"**{int(config.num_projects)} project(s)** · {size_text}",
     [("configuration", "Configuration")], size_lines)

policy = setup.policy
kinds = {name: info.kind for name, info in infos.items()}
problems = policy.validate(kinds)
report = feasibility_report(setup) if setup.specs else None
rules_status = "done"
rule_lines: List[str] = []
if problems:
    rules_status = "error"
    rule_lines += [f"Invalid: {p}" for p in problems[:3]]
if report is not None:
    if report.errors:
        rules_status = "error"
    elif report.warnings and rules_status == "done":
        rules_status = "warning"
    rule_lines.append(f"Feasibility check: {report.summary()}")
    rule_lines += [f"{f.rule_label + ': ' if f.rule_label else ''}{f.message}"
                   for f in report.errors[:3]]
if config.algorithm != "engine":
    rule_lines.append("A classic algorithm is selected: the rules score its runs "
                      "but do not drive them.")
step(3, "Allocation rules", rules_status,
     f"**{policy.name}** · {active_rule_count(policy)} active rules · solver "
     f"{policy.solver.method}, {policy.solver.time_limit_s:g} s per project",
     [("rules", "Allocation Rules")], rule_lines)

if active is not None:
    form_status = "done"
    form_detail = (f"Active run **{rh.run_title(active)}** · "
                   f"{SOURCE_LABELS.get(active.source, active.source)}"
                   + (f" · {ago(active.created_at)}" if active.created_at else "")
                   + (" · **unsaved edits** (Team Editor)" if unsaved else ""))
elif alloc is not None:
    form_status = "done"
    form_detail = "Teams formed (run not recorded in the history)."
else:
    form_status = "todo"
    form_detail = "No teams formed yet."
step(4, "Form teams", form_status, form_detail, [("formation", "Team Formation")])

review_links = [(key, label) for key, label in (
    ("editor", "Team Editor"), ("compare", "Compare Runs"), ("analytics", "Analytics"),
    ("cross_project", "Cross-Project"),
) if page_exists(key)]
edited = sum(1 for r in history if r.source == "edited")
step(5, "Review & fine-tune", "done" if edited or len(history) > 1 else
     ("info" if alloc is not None else "todo"),
     f"{len(history)} run(s) in the history" + (f" · {edited} edited" if edited else ""),
     review_links)

step(6, "Share & export", "info" if alloc is not None else "todo",
     "Download team lists, reports and the configuration."
     if alloc is not None else "Available once teams are formed.",
     [("export", "Export")])

st.subheader("📊 Active run")
summary: Dict[str, Any] = {}
if unsaved and reports:
    summary = es.summarize_reports(reports)
    st.caption(f"{rh.run_title(active)} **+ unsaved edits** — the numbers describe the edited "
               "allocation that Analytics, Export and Team Formation show. Save it as a run in "
               "the Team Editor to keep it.")
elif active is not None and not reports and active.reports:
    st.caption(f"{rh.run_title(active)} has no scores for the loaded roster, so Analytics and "
               "Export show none either; the run's stored scores are in the history below.")
elif active is not None and reports and not es.run_holds_reports(active, reports):
    summary = {**{k: active.summary[k] for k in ("method", "elapsed_s")
                  if k in (active.summary or {})},
               **es.summarize_reports(reports)}
    st.caption(f"{rh.run_title(active)} **scored again** with {es.rules_source_text()} — the "
               "numbers describe the active reports that Analytics, Export and Team Formation "
               "show; the run's stored scores in the history below differ.")
elif active is not None and active.summary:
    summary = dict(active.summary)
elif reports:
    summary = es.summarize_reports(reports)
if not summary and active is not None and not reports and active.reports:
    st.info("No scores for the active allocation with this roster. Load the dataset the teams "
            "were formed with, or form new teams on **Team Formation**.")
elif not summary:
    st.info("No rule-based results yet. Run the rules engine on **Team Formation** to "
            "see quality, rule compliance and repeat pairings here.")
else:
    render_headline_row(summary, reports, preferences=True)
    extras = []
    if summary.get("method"):
        extras.append(f"method {summary['method']}")
    if summary.get("elapsed_s") is not None:
        extras.append(f"{float(summary['elapsed_s']):.1f} s")
    if summary.get("teams"):
        extras.append(f"{int(summary['teams'])} teams in {int(summary.get('projects') or 0)} "
                      "generated project(s)")
    if extras:
        st.caption(" · ".join(extras))

st.subheader(f"🗂️ Run history ({len(history)})")
if not history:
    st.caption("Every formation run, edit and evaluation is kept here for comparison.")
else:
    rows = []
    shown = list(reversed(history[-8:]))
    for record in shown:
        s = record.summary or {}
        hl = {r.key: r for r in headline_rows(s, record.reports)}
        row = {
            "Run": ("📌 " if record.pinned else "") + record.run_id
            + ((" (active, unsaved edits)" if unsaved else " (active)")
               if active is not None and record.run_id == active.run_id else ""),
            "Label": rh.display_label(record.label),
            "Source": SOURCE_LABELS.get(record.source, record.source),
            "Created": ago(record.created_at),
            "Quality vs random": hl["quality"].value,
            "Hard violations": hl["hard"].value,
            "Repeat pairs": hl["repeats"].value,
        }
        if "section_mates" in hl:
            row["Section-mates reunited"] = hl["section_mates"].value
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
        "Quality vs random": st.column_config.NumberColumn(format="%.1f%%"),
    })
    if page_exists("compare"):
        page_link("compare", "Compare runs side by side")
