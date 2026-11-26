import io
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import extensions
from app.core import engine_session as es
from app.core import run_history
from app.core.data_session import (
    departed_warning,
    ensure_session_state,
    get_data_loader,
    page_exists,
    require_results,
    switch_to,
)
from app.core.engine_session import (
    active_reports,
    history_rules_text,
    roster_frame,
    rules_of_active_reports,
    rules_source_text,
    shared_team_counts,
    stale_report_ids,
    summarize_reports,
)
from app.core.models import Team
from app.pages._views import render_headline_row, render_scored_rules_caption  # noqa: E402
from app.utils import lms_export
from app.utils.exporters import (
    DataExporter,
    arrow_safe,
    composition_frame,
    engine_report_json,
    export_configuration_to_json,
    plain_projects,
    project_label,
    rule_compliance_frame,
    safe_csv,
    wide_roster_frame,
)
from app.utils.roster_view import RosterView
from app.utils.headline import headline_rows
from app.utils.report_html import default_report_attributes, render_allocation_html

st.set_page_config(
    page_title="Export | Group Diversifier",
    page_icon="📤",
    layout="wide",
)

ensure_session_state()

_JSON_PREVIEW_TEAMS = 3
_COMPOSITION_DEFAULT = 4


def teams_with_project(projects) -> List[Team]:
    return [
        replace(team, project_id=project.project_id)
        for project in projects
        for team in project.teams
    ]


def json_preview(json_text: str) -> Dict[str, Any]:
    data = json.loads(json_text)
    preview = {k: v for k, v in data.items() if k != "projects"}
    preview["projects"] = []
    for project in data.get("projects", []):
        teams = project.get("teams", [])
        trimmed = dict(project, teams=teams[:_JSON_PREVIEW_TEAMS])
        if len(teams) > _JSON_PREVIEW_TEAMS:
            trimmed["teams_not_shown"] = len(teams) - _JSON_PREVIEW_TEAMS
        preview["projects"].append(trimmed)
    return preview


def csv_preview(text: str, header: bool = True, rows: int = 15) -> pd.DataFrame:
    if not text.strip():
        return pd.DataFrame()
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                        header=0 if header else None)
    return frame.head(rows)


st.title("📤 Export Results")
st.markdown("Download team formation results in various formats.")

with st.expander("ℹ️ What is this page?", expanded=False):
    st.markdown("""
    This page lets you **download your team formation results** in different formats.

    **Export formats:**
    - **CSV**: long format (team_id, project_id, student_id, …) — re-importable as prior projects
    - **Printable report**: one self-contained HTML file per allocation — headline numbers, rule
      compliance and every team with its members, composition and rule violations; print it or
      save it as PDF. Also as one-team-per-page *team sheets*, optionally anonymised.
    - **LMS groups**: group-import files for Canvas, Brightspace (D2L) and Moodle
    - **Roster & composition**: one row per student with their team in every project, and
      head counts per value for each team
    - **Rule report**: the allocation engine's rule compliance (CSV) and full report (JSON)
    - **Excel**: workbook with summary, one sheet per project, diversity scores and repeat pairings
    - **JSON**: machine-readable results including the configuration used
    - **Markdown report**: human-readable summary
    """)

st.subheader("⚙️ Configuration")
st.markdown("Download the full configuration snapshot as JSON (the current, live settings).")
live_config = st.session_state.config
config_export_json = export_configuration_to_json(
    live_config,
    name="Group Diversifier Configuration",
)
st.download_button(
    "📥 Download Configuration (JSON)",
    config_export_json,
    f"group_diversifier_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    "application/json",
    width="stretch",
    on_click="ignore",
    help=(
        "Captures diversity weights, dimension and proportionality settings, "
        "constraints, algorithm parameters, allocation rules and team sizes incl. "
        "per-project overrides — re-importable from the Configuration page."
    ),
)

st.markdown("---")
require_results()

alloc = es.active_allocation()
result = alloc.formation_result
projects = alloc.projects
result_config = result.config
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

loader = get_data_loader()
students = loader.get_students()
frame = roster_frame(students, loader.get_dataframe())
view = RosterView(frame)
plain = plain_projects(projects)
project_names = {p["project_id"]: p["name"] for p in plain}
labels = {p["project_id"]: project_label(p) for p in plain}
all_project_ids = [p["project_id"] for p in plain]
generated_ids = [p["project_id"] for p in plain if not p.get("imported")]
reports = active_reports()
stale = stale_report_ids(plain, reports, frame)
scored_rules = rules_of_active_reports()
exporter = DataExporter(roster=frame, reports=reports, raw_df=loader.get_dataframe())

st.subheader("📊 Export Summary")
summary_cols = st.columns(4)
summary_cols[0].metric("Projects", len(projects))
summary_cols[1].metric("Total Teams", sum(len(p.teams) for p in projects))
summary_cols[2].metric("Students", len({sid for p in projects for t in p.teams for sid in t.member_ids}))
if reports:
    quality = summarize_reports(reports).get("quality_index_soft")
    summary_cols[3].metric("Quality vs random", "–" if quality is None else f"{quality:.1f}%",
                           help="Mean over scored projects (see the Rule Report tab).")
else:
    summary_cols[3].metric("Classic diversity score", f"{result.total_score:.2f}",
                           help="Legacy display index of the classic algorithms. Score the "
                                "allocation with the rules on Analytics for quality vs random.")
departed_warning(projects, view.ids)

st.markdown("---")
(tab_csv, tab_html, tab_lms, tab_tables, tab_rules,
 tab_excel, tab_json, tab_report) = st.tabs([
    "📄 CSV Export",
    "🖨️ Printable Report",
    "🎓 LMS Groups",
    "🧾 Roster & Composition",
    "✅ Rule Report",
    "📊 Excel Export",
    "📋 JSON Export",
    "📝 Report",
])

with tab_csv:
    st.subheader("CSV Export Options")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Team Assignments")
        st.markdown("Export team assignments with student details.")
        include_details = st.checkbox("Include student details", value=True, key="csv_details")
        csv_data = exporter.export_teams_to_csv(teams_with_project(projects), include_details=include_details)
        st.download_button(
            "📥 Download Team Assignments (CSV)",
            csv_data,
            f"team_assignments_{stamp}.csv",
            "text/csv",
            width="stretch",
            on_click="ignore",
        )
    with col2:
        st.markdown("### Per-Project Export")
        st.markdown("Export each project separately.")
        selected_project = st.selectbox(
            "Select Project",
            range(len(projects)),
            format_func=lambda x: projects[x].name + (" (imported)" if projects[x].imported else ""),
        )
        project = projects[selected_project]
        project_csv = exporter.export_teams_to_csv(teams_with_project([project]), include_details=include_details)
        st.download_button(
            f"📥 Download {project.name} (CSV)",
            project_csv,
            f"project_{project.project_id}_{stamp}.csv",
            "text/csv",
            width="stretch",
            on_click="ignore",
        )

    st.markdown("---")
    st.markdown("### Preview")
    preview_df = pd.read_csv(io.StringIO(csv_data), dtype=str, keep_default_na=False)
    st.dataframe(preview_df.head(20), width="stretch", hide_index=True)

with tab_html:
    st.subheader("Printable Report")
    st.markdown("One self-contained HTML file (no internet needed): open it in a browser and "
                "print or *Save as PDF*. Every project starts on a new page and team cards never "
                "split across pages.")
    c1, c2 = st.columns([2, 1])
    html_projects = c1.multiselect("Projects", all_project_ids, default=all_project_ids,
                                   format_func=lambda pid: labels[pid], key="html_projects")
    html_layout = c2.radio(
        "Layout", ["report", "sheets"], horizontal=True, key="html_layout",
        format_func=lambda v: "Full report" if v == "report" else "Team sheets",
        help="Full report: headline numbers and rule compliance per project, then all teams. "
             "Team sheets: one team per printed page (hand-outs for students or tutors).")
    attribute_options = view.usable_attributes()
    html_default = [a for a in default_report_attributes(view, reports) if a in attribute_options]
    html_attributes = st.multiselect(
        "Attributes shown per student", attribute_options, default=html_default,
        key="html_attributes",
        help="Also summarised per team. Defaults to the attributes your allocation rules use.")
    c3, c4 = st.columns([2, 1])
    html_title = c3.text_input("Title", value="Team allocation", key="html_title")
    html_anonymize = c4.checkbox(
        "Anonymize student ids", key="html_anonymize",
        help="Replaces ids with pseudonyms (S-3FA2C1; same id → same pseudonym within "
             "this report, and across reports only when you give a key) "
             "everywhere, including violation texts, and leaves out columns that look "
             "identifying (name, e-mail, login, number…).")
    html_salt = ""
    if html_anonymize:
        html_salt = c4.text_input(
            "Pseudonym key (optional)", key="html_salt", type="password",
            help="Any secret text: the same key always gives the same pseudonyms; another key "
                 "makes them unlinkable to earlier reports. Left empty, each report gets a fresh "
                 "random key: pseudonyms stay stable inside the report but cannot be worked out "
                 "from the student numbers.")
    chosen_plain = [p for p in plain if p["project_id"] in html_projects]
    if not chosen_plain:
        st.info("Select at least one project.")
    else:
        if stale:
            st.caption("Teams edited after scoring are shown without their (outdated) penalties.")
        report_html = render_allocation_html(
            chosen_plain, frame, reports, title=html_title.strip() or "Team allocation",
            attributes=html_attributes, anonymize=html_anonymize,
            team_sheets=html_layout == "sheets", salt=html_salt,
        )
        suffix = "team_sheets" if html_layout == "sheets" else "team_report"
        st.download_button(
            "📥 Download Printable Report (HTML)",
            report_html,
            f"{suffix}{'_anonymized' if html_anonymize else ''}_{stamp}.html",
            "text/html",
            key="html_download",
            type="primary",
            width="stretch",
            on_click="ignore",
        )
        if st.toggle("Show preview", key="html_preview"):
            st.iframe(report_html, height=720)

with tab_lms:
    st.subheader("LMS Group Import")
    st.markdown("Files to create the teams as groups in your learning platform — one group set "
                "(Canvas) / group category (Brightspace) per project. Students must already be "
                "enrolled in the course.")
    lms_key = st.radio("Learning platform", list(lms_export.FORMATS), horizontal=True,
                       format_func=lambda k: lms_export.FORMATS[k].title, key="lms_target")
    lms_format = lms_export.FORMATS[lms_key]
    st.caption(lms_format.help)
    lms_projects = st.multiselect(
        "Projects", all_project_ids, default=generated_ids or all_project_ids,
        format_func=lambda pid: labels[pid], key="lms_projects",
        help="Imported projects are left out by default (they usually exist in the LMS already).")
    candidates = lms_export.identifier_candidates(frame)
    c1, c2 = st.columns(2)
    id_source = c1.selectbox(
        "Student identifier (roster column)", candidates, key="lms_id_source",
        help="Roster columns with one unique value per student. Use the column that holds what "
             "the LMS knows students by (SIS id, login, username).")
    id_column = c2.selectbox("Write it as", lms_format.id_options,
                             index=lms_format.id_options.index(lms_format.default_id),
                             key=f"lms_id_column_{lms_key}")
    group_pattern = st.text_input(
        "Group name pattern", value=lms_export.DEFAULT_GROUP_PATTERN, key="lms_group_pattern",
        help="Placeholders: {project_id}, {project} (project name), {team} (e.g. Team 3), "
             "{n} (team number), {nn} (zero-padded team number). Names must be unique in the "
             "course, so keep the project in them.")
    options: Dict[str, Any] = {"pattern": group_pattern or lms_export.DEFAULT_GROUP_PATTERN}
    identifiers = (lms_export.identifiers_from_roster(frame, id_source)
                   if id_source != "student_id" else None)
    header = True
    ready = True
    if lms_key == "canvas":
        options["id_column"] = id_column
    elif lms_key == "brightspace":
        b1, b2, b3 = st.columns(3)
        options["code_pattern"] = b1.text_input(
            "Group code pattern", value=lms_export.DEFAULT_CODE_PATTERN, key="lms_code_pattern",
            help="Must match the codes of the groups you created in Brightspace (codes are "
                 "required, groups are not created by the import).") or lms_export.DEFAULT_CODE_PATTERN
        org_options = ["(leave empty)"] + candidates
        if id_column == "Username":
            org_source = b2.selectbox("OrgDefinedId from", org_options, key="lms_org_source")
            if org_source != "(leave empty)":
                options["org_identifiers"] = lms_export.identifiers_from_roster(frame, org_source)
        header = b3.checkbox("Header row", value=True, key="lms_header")
        options.update(id_field=id_column, header=header)
    else:
        course = st.text_input("Course short name (course1)", key="lms_course",
                               help="The Moodle course short name, e.g. PROJ-2026.")
        options["course_shortname"] = course
        ready = bool(course.strip())
    if identifiers is not None:
        options["identifiers"] = identifiers
        missing = lms_export.unmapped_students(
            [p for p in plain if p["project_id"] in lms_projects], identifiers)
        if missing:
            st.warning(f"{len(missing)} student(s) have no '{id_source}' value and are written "
                       f"with their student id: {', '.join(missing[:12])}"
                       f"{' …' if len(missing) > 12 else ''}.")
    if not lms_projects:
        st.info("Select at least one project.")
    elif not ready:
        st.info("Enter the Moodle course short name to build the file.")
    else:
        lms_csv = lms_export.lms_group_csv(lms_key, plain, project_ids=lms_projects, **options)
        if len(lms_projects) == 1:
            st.download_button(
                f"📥 Download {lms_format.title} groups (CSV)",
                lms_csv,
                f"{lms_key}_groups_project_{lms_projects[0]}_{stamp}.csv",
                "text/csv", key="lms_download", type="primary", width="stretch",
                on_click="ignore",
            )
        else:
            st.download_button(
                f"📥 Download {lms_format.title} groups (ZIP, one CSV per project)",
                lms_export.lms_zip(lms_key, plain, lms_projects, **options),
                f"{lms_key}_groups_{stamp}.zip",
                "application/zip", key="lms_download", type="primary", width="stretch",
                on_click="ignore",
            )
        if lms_key == "moodle" and len(lms_projects) > 1:
            st.download_button(
                "📥 All projects in one Moodle file (CSV)", lms_csv,
                f"moodle_groups_all_{stamp}.csv", "text/csv", key="lms_download_combined",
                width="stretch", on_click="ignore",
            )
        st.markdown("**Preview**")
        st.dataframe(csv_preview(lms_csv, header=header), width="stretch", hide_index=True)
    with st.expander("Format details & assumptions"):
        st.markdown("""
        - **Canvas** — *People › Groups › (group set) › Import*. Columns `user_id` (the student's
          **SIS id**), `login_id` or `canvas_user_id`, plus `group_name`; unknown group names are
          created in the set. Canvas does not read a `sis_user_id` column in this import.
        - **Brightspace (D2L)** — *Groups › (category) › Import* (bulk enrolment, July 2025
          release). Columns `Action,OrgDefinedId,Username,Group Name,Group Code` with `ENROLL`;
          `Username` and an **existing** group code are required by default, so create the
          category with the right number of groups first and match the code pattern. Some
          institutions' guides import without the header row — untick *Header row* for that.
        - **Moodle** — *Site administration › Users › Upload users* with
          `username,course1,group1` (upload type *Update existing users only*): users are enrolled
          in the course if needed and added to the group, which is created when missing. Needs
          site-administrator rights.
        - Compare the file once with a template exported from your own course: institutions
          configure identifiers differently.
        """)

with tab_tables:
    st.subheader("Wide Roster")
    st.markdown("One row per student: the roster columns, then the team in every project "
                "(`team_project_<id>`; empty = not taking part). The team columns are ignored "
                "if the file is uploaded again as student data.")
    wide_attrs = st.multiselect("Roster columns", view.attributes, default=view.attributes,
                                key="wide_attributes")
    wide = wide_roster_frame(plain, frame, attributes=wide_attrs)
    st.download_button(
        "📥 Download Wide Roster (CSV)", safe_csv(wide),
        f"wide_roster_{stamp}.csv", "text/csv", key="wide_download", width="stretch",
        on_click="ignore",
    )
    st.dataframe(arrow_safe(wide.head(20)), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Team Composition")
    st.markdown("One row per team with head counts per value (categorical attributes) or the "
                "team mean (numeric attributes).")
    rule_attrs = [a for a in default_report_attributes(view, reports) if a in attribute_options]
    comp_attrs = st.multiselect(
        "Attributes", attribute_options,
        default=(rule_attrs or attribute_options)[:_COMPOSITION_DEFAULT],
        key="composition_attributes")
    comp_projects = st.multiselect("Projects", all_project_ids, default=all_project_ids,
                                   format_func=lambda pid: labels[pid], key="composition_projects")
    composition = composition_frame(plain, frame, comp_attrs, project_ids=comp_projects)
    st.download_button(
        "📥 Download Team Composition (CSV)", safe_csv(composition),
        f"team_composition_{stamp}.csv", "text/csv", key="composition_download",
        width="stretch", on_click="ignore",
    )
    st.dataframe(arrow_safe(composition.head(30)), width="stretch", hide_index=True)

with tab_rules:
    st.subheader("Rule Report")
    if not reports:
        st.info("The active allocation has no rule report (it was formed by a legacy algorithm, "
                "imported or restored). Score it with the current allocation rules on the "
                "Analytics page, then come back here.")
        if st.button("📈 Open Analytics", key="export_open_analytics"):
            switch_to("analytics")
    else:
        if stale:
            st.warning(f"{', '.join(labels.get(pid, pid) for pid in stale)}: teams changed after "
                       "scoring — the report describes the earlier teams. Re-score on the "
                       "Analytics page for current numbers.")
        summary = summarize_reports(reports)
        render_headline_row(summary, reports)
        render_scored_rules_caption(
            extra="The Engine Report (JSON) carries these rules as its policy.")
        rules_df = rule_compliance_frame(reports, project_names)
        record = run_history.active_run()
        metadata = {
            "stale_projects": stale,
            "run_id": record.run_id if record is not None else None,
            "run_label": record.label if record is not None else None,
        }
        r1, r2 = st.columns(2)
        r1.download_button(
            "📥 Rule Compliance (CSV)", safe_csv(rules_df),
            f"rule_compliance_{stamp}.csv", "text/csv", key="rules_csv_download",
            width="stretch", on_click="ignore",
        )
        r2.download_button(
            "📥 Engine Report (JSON)",
            engine_report_json(reports, scored_rules.to_dict(), summary, projects=plain,
                               metadata=metadata),
            f"engine_report_{stamp}.json", "application/json", key="engine_json_download",
            width="stretch", on_click="ignore",
            help="Every project's full report (teams, penalties, breakdowns, violations, rule "
                 "compliance, history, preferences), the policy and the summary.",
        )
        st.dataframe(rules_df, width="stretch", hide_index=True, column_config={
            "compliance_pct": st.column_config.NumberColumn("compliance %", format="%.0f"),
            "quality_pct": st.column_config.NumberColumn("quality %", format="%.0f"),
            "points": st.column_config.NumberColumn(format="%.2f"),
        })

with tab_excel:
    st.subheader("Excel Export")
    st.markdown("Export complete results to Excel with multiple sheets.")
    st.markdown("""
    The Excel file includes:
    - **Summary** - Overview metrics
    - **Project sheets** - Team assignments per project
    - **Diversity Scores** - Detailed diversity metrics
    - **Repeat Pairings** - Students who worked together multiple times
    """)
    excel_data = exporter.export_teams_to_excel(result)
    st.download_button(
        "📥 Download Complete Results (Excel)",
        excel_data,
        f"group_diversifier_results_{stamp}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        type="primary",
        on_click="ignore",
    )

    st.markdown("---")
    st.markdown("### Sheet Previews")
    preview_tab = st.selectbox(
        "Select sheet to preview",
        ["Summary", "Team Assignments", "Diversity Scores"]
    )
    if preview_tab == "Summary":
        metric_rows = [(row.label, row.text)
                       for row in headline_rows(summarize_reports(reports), reports)] \
            if reports else []
        metric_rows += [
            ("Classic diversity score", f"{result.total_score:.2f}"),
            ("Execution Time", f"{result.execution_time:.2f}s"),
            ("Projects", str(len(projects))),
            ("Teams", str(sum(len(p.teams) for p in projects))),
            ("Algorithm", str(result_config.algorithm)),
        ]
        summary_df = pd.DataFrame({"Metric": [m for m, _ in metric_rows],
                                   "Value": [v for _, v in metric_rows]})
        st.dataframe(summary_df, width="stretch", hide_index=True)
    elif preview_tab == "Team Assignments":
        rows = []
        member_columns = extensions.hook("ui.export.member_columns") if extensions.active() else None
        if member_columns is None:
            preview_view = exporter.roster_view()
            preview_attrs = preview_view.usable_attributes()[:2] if preview_view is not None else []

            def member_columns(member):
                if preview_view is None or member.student_id not in preview_view:
                    return {a.replace("_", " ").title(): "" for a in preview_attrs}
                return {a.replace("_", " ").title(): preview_view.raw(member.student_id, a) or ""
                        for a in preview_attrs}
        for proj in projects[:1]:
            for team in proj.teams:
                for member in team.members:
                    rows.append({
                        "Project": proj.name,
                        "Team": team.team_id,
                        "Student ID": member.student_id,
                        **member_columns(member),
                    })
        st.dataframe(pd.DataFrame(rows).head(20), width="stretch", hide_index=True)
    elif extensions.without_inactive_metrics(result.diversity_scores):
        st.dataframe(pd.DataFrame([extensions.without_inactive_metrics(result.diversity_scores)]),
                     width="stretch", hide_index=True)

with tab_json:
    st.subheader("JSON Export")
    st.markdown("Export results in JSON format for programmatic use.")
    json_data = exporter.export_to_json(result)
    st.download_button(
        "📥 Download Results (JSON)",
        json_data,
        f"group_diversifier_results_{stamp}.json",
        "application/json",
        width="stretch",
        on_click="ignore",
    )
    st.markdown("---")
    st.markdown("### JSON Preview")
    with st.expander("View JSON Structure"):
        st.caption(f"Showing the first {_JSON_PREVIEW_TEAMS} teams of each project; the download is complete.")
        st.json(json_preview(json_data), expanded=1)

with tab_report:
    st.subheader("Generate Report")
    st.markdown("Generate a formatted markdown report of the results.")
    report = exporter.generate_report(result)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            "📥 Download Report (Markdown)",
            report,
            f"team_formation_report_{stamp}.md",
            "text/markdown",
            width="stretch",
            on_click="ignore",
        )
    with col2:
        st.download_button(
            "📥 Download Report (Text)",
            report,
            f"team_formation_report_{stamp}.txt",
            "text/plain",
            width="stretch",
            on_click="ignore",
        )
    st.markdown("---")
    st.markdown("### Report Preview")
    st.markdown(report)

st.markdown("---")
st.subheader("🔧 Additional Exports")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Team Summary")
    summary_df = exporter.export_team_summary(projects)
    if "Project" not in summary_df.columns:
        summary_df.insert(0, "Project", [t.project_id for t in teams_with_project(projects)])
    st.download_button(
        "📥 Team Summary (CSV)",
        safe_csv(summary_df),
        "team_summary.csv",
        "text/csv",
        width="stretch",
        on_click="ignore",
    )
with col2:
    st.markdown("### Overlap Matrix")
    if result.overlap_matrix:
        st.caption(f"Teams_Shared counts with {rules_source_text()}: "
                   f"{history_rules_text(scored_rules)}. Groups_Shared counts every shared group.")
        counted = shared_team_counts(projects, scored_rules, frame)
        overlap_df = pd.DataFrame([
            {"Student_1": p[0], "Student_2": p[1], "Groups_Shared": c,
             "Teams_Shared": counted.get(p, 0)}
            for p, c in result.overlap_matrix.items()
        ])
        st.download_button(
            "📥 Overlap Matrix (CSV)",
            safe_csv(overlap_df),
            "overlap_matrix.csv",
            "text/csv",
            width="stretch",
            on_click="ignore",
        )
    else:
        st.info("No overlap data (single project)")

st.markdown("---")
st.subheader("📚 Export Updated Student Data")
st.markdown("Export the student data with any updates made in this session (e.g. edited preferences).")
updated_df = loader.get_dataframe()
if not updated_df.empty:
    st.download_button(
        "📥 Download Updated Student Data (CSV)",
        safe_csv(updated_df),
        f"students_updated_{stamp}.csv",
        "text/csv",
        width="stretch",
        type="primary",
        on_click="ignore",
    )

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Cross-Project Analysis", width="stretch"):
        switch_to("cross_project")
with col2:
    if page_exists("home") and st.button("🏠 Home", width="stretch"):
        switch_to("home")
