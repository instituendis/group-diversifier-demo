import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Set

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core import engine_session as es
from app.core.data_session import (
    ensure_session_state,
    get_data_loader,
    require_results,
    switch_to,
)
from app.core.diversity_metrics import DiversityMetrics
from app.core.engine_session import roster_frame
from app.core.exposure import (
    ANY_DIFFERENCE,
    MAX_EXPOSURE_VALUES,
    category_pair_ratio,
    cross_group_exposure,
    group_of,
    pair_efficiency,
    pair_shared_projects,
    per_student_repeat_summary,
    project_label,
    project_teams,
    repeat_explanation_table,
    repeat_reasons,
    rotation_matrix,
    rotation_team_columns,
    teammate_reach,
    teammates_by_student,
    teammates_per_student,
    teams_shared_distribution,
    weighted_repeats,
)
from app.core.ids import counts_as_team, id_sort_key
from app.engine.repeats import summarize_explanations
from app.engine.roster import infer_attributes
from app import extensions  # noqa: E402

GROUP_EXAMPLE = extensions.text("cross.group_example", "Red with anyone else")


def _extension_ui(name: str):
    return extensions.hook(f"ui.cross_project.{name}") if extensions.active() else None



from app.pages._views import classic_team_score, render_scored_rules_caption  # noqa: E402
from app.utils.charts import (
    ALL_PAIRS_LABEL,
    ALL_RATE_LABEL,
    HEATMAP_AXIS,
    HEATMAP_COLOR_MODES,
    HEATMAP_GRADIENTS,
    HEATMAP_LAYOUT,
    HISTORY_OFF_NOTE,
    REPEAT_HEATMAP_COLOR_MODES,
    REPEAT_HEATMAP_MAX_STUDENTS,
    category_pair_figure,
    cross_group_figure,
    heatmap_color_groups,
    reach_timeline_figure,
    repeat_heatmap_trace,
    rotation_heatmap_traces,
)
from app.utils.exporters import safe_csv
from app.utils.headline import headline_rows  # noqa: E402
from app.utils.roster_view import MISSING_LABEL, RosterView  # noqa: E402

st.set_page_config(
    page_title="Cross-Project | Group Diversifier",
    page_icon="🔄",
    layout="wide",
)

ensure_session_state()


def render_exposure_tab(projects, students, raw_df, history, source: str,
                        per_project: pd.DataFrame, pairs: pd.DataFrame) -> None:
    view = RosterView(roster_frame(students, raw_df))
    cohort_ids = list(dict.fromkeys(view.ids + [sid for p in projects for t in project_teams(p)
                                                for sid in t]))

    st.subheader("Teammate Reach")
    st.caption("How many different classmates each student has worked with, against the "
               "maximum their team sizes allow (every repeat teammate costs one new contact).")
    per_student, timeline = teammate_reach(projects, cohort_ids)
    if per_student.empty:
        st.info("No team members to analyse.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean distinct teammates",
              f"{per_student['Distinct teammates'].mean():.1f}",
              delta=f"of {per_student['Maximum possible'].mean():.1f} possible", delta_color="off")
    c2.metric("Students meeting a classmate again",
              int((per_student["Repeat teammates"] > 0).sum()),
              help="Any shared group counts here, sections included — unlike the repeat pairs "
                   "of the allocation rules (Repeat Pairs tab).")
    c3.metric("Lowest reach", f"{per_student['Reach'].min():.0%}",
              help="Distinct teammates ÷ maximum possible for the least-exposed student.")
    c4.metric("Cohort met (mean)", f"{per_student['Share of cohort met'].mean():.0%}",
              help="Distinct teammates ÷ (cohort size − 1).")
    if len(timeline) > 1:
        st.plotly_chart(reach_timeline_figure(timeline), width="stretch",
                        key="exposure_reach_chart")
    with st.expander("Per-student reach table"):
        st.dataframe(per_student, width="stretch", hide_index=True, column_config={
            "Reach": st.column_config.ProgressColumn("Reach", min_value=0.0, max_value=1.0,
                                                     format="percent"),
            "Share of cohort met": st.column_config.NumberColumn(format="percent"),
        })

    st.markdown("---")
    st.subheader("Cross-Group Exposure")
    st.caption("Within-team pairings between students of different groups, per project and "
               "cumulatively, against what random teams of the same participants would give.")
    options = [a for a in view.usable_attributes()
               if view.kind(a) == "categorical"
               and 2 <= len([v for v in view.cohort_counts(a) if v != MISSING_LABEL])
               <= MAX_EXPOSURE_VALUES]
    if not options:
        st.info("No categorical attribute with 2–40 values to analyse.")
    else:
        default_attr = next((a for a in (_extension_ui("exposure_defaults") or ())
                             if a in options), options[0])
        c1, c2 = st.columns(2)
        attribute = c1.selectbox("Attribute", options, index=options.index(default_attr),
                                 key="exposure_attribute")
        values = [v for v in view.cohort_counts(attribute) if v != MISSING_LABEL]
        focus_hook = _extension_ui("default_focus")
        default_focus = (focus_hook(attribute, values, ANY_DIFFERENCE) if focus_hook is not None
                         else ANY_DIFFERENCE)
        focus_options = [ANY_DIFFERENCE] + values
        focus = c2.selectbox(
            "Count as cross-group", focus_options, index=focus_options.index(default_focus),
            format_func=lambda v: "any two different values" if v == ANY_DIFFERENCE
            else f"{v} with anyone else", key=f"exposure_focus_{attribute}",
            help="Pick a value to count only pairings between that group and everyone else "
                 f"(e.g. {GROUP_EXAMPLE}).")
        groups = group_of(view, attribute, focus)
        exposure = cross_group_exposure(projects, groups)
        last = exposure.iloc[-1]
        m1, m2, m3 = st.columns(3)
        share = exposure["Cross-group share"].dropna()
        m1.metric("Cross-group share (mean)", f"{share.mean():.0%}" if len(share) else "–",
                  delta=(f"random {exposure['Random expectation'].dropna().mean():.0%}"
                         if exposure["Random expectation"].notna().any() else None),
                  delta_color="off")
        m2.metric("Distinct cross-group pairs", int(last["Distinct cross-group pairs so far"]))
        m3.metric("Students never exposed", int(last["Students not yet exposed"]),
                  help="Students (with a value) who have not had a teammate from another group "
                       "in any project yet.")
        st.plotly_chart(cross_group_figure(exposure), width="stretch", key="exposure_cross_chart")
        st.dataframe(exposure, width="stretch", hide_index=True, column_config={
            "Cross-group share": st.column_config.NumberColumn(format="percent"),
            "Random expectation": st.column_config.NumberColumn(format="percent"),
        })
        if focus == ANY_DIFFERENCE and len(values) >= 2:
            labels, ratio, observed = category_pair_ratio(projects, groups)
            st.markdown("**Who is paired with whom** (all projects, observed ÷ random)")
            st.plotly_chart(category_pair_figure(labels, ratio, observed), width="stretch",
                            key="exposure_pair_heatmap")
            st.caption("Below 1× on the diagonal = fewer same-group pairings than random teams "
                       "(more mixing); above 1× off the diagonal = groups meet more often than "
                       "chance.")

    st.markdown("---")
    st.subheader("Repeat Pairings under the Contact Model")
    cap_text = ("no cap" if history.max_times_together is None
                else f"at most {history.max_times_together} shared team(s) per pair")
    model_text = ("team-size contact model (a shared group of size s counts "
                  f"min(1, {history.reference_team_size - 1}/(s − 1)))"
                  if history.contact_model == "team_size" else "uniform contact model")
    st.caption(f"History settings of {source}: "
               f"{model_text}, decay {history.decay:g} per project, {cap_text}"
               + (f"; groups over {history.ignore_groups_larger_than} are sections and do not "
                  "count toward the cap" if history.ignore_groups_larger_than else "")
               + f"; repeat weight {history.repeat_weight:g}.")
    st.dataframe(per_project, width="stretch", hide_index=True)
    st.caption("*Pairs met before*: team-mates who shared any earlier group (sections included) · "
               "*Repeat team pairs*: shared an earlier team, which counts toward the cap · "
               "*Contact weight*: their summed contact weights, × repeat weight = *Penalty points* "
               "the allocation rules charge for that project.")
    if pairs.empty:
        st.success("No pair shared a team more than once.")
    else:
        st.markdown(f"**{len(pairs)} pair(s) met more than once** — contact weight as the next "
                    "project would see it (higher = stronger penalty for meeting again).")
        st.dataframe(pairs, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Who Has Never Worked with Whom")
    together = teammates_per_student(projects)
    lookup_ids = sorted(cohort_ids, key=id_sort_key)
    c1, c2 = st.columns([1, 2])
    student = c1.selectbox("Student", lookup_ids, key="exposure_student")
    attrs = [a for a in view.usable_attributes() if view.kind(a) != "numeric"][:12]
    default_attrs = [a for a in (_extension_ui("lookup_defaults") or ()) if a in attrs] or attrs[:3]
    shown = c2.multiselect("Show attributes", attrs, default=default_attrs,
                           key="exposure_lookup_attrs")
    met = together.get(student, Counter())
    never = [sid for sid in lookup_ids if sid != student and sid not in met]
    m1, m2 = st.columns(2)
    m1.metric("Worked with", len(met))
    m2.metric("Never worked with", len(never))
    left, right = st.columns(2)
    with left:
        st.markdown("**Never in a team together**")
        never_df = pd.DataFrame([{"Student": sid, **{a: view.text(sid, a) for a in shown}}
                                 for sid in never])
        st.dataframe(never_df, width="stretch", hide_index=True, height=320)
        st.download_button("📥 Download list (CSV)", safe_csv(never_df),
                           f"never_worked_with_{student}.csv", "text/csv",
                           key="exposure_never_download", on_click="ignore")
    with right:
        st.markdown("**Worked together**")
        met_df = pd.DataFrame([{"Student": sid, "Times": count,
                                **{a: view.text(sid, a) for a in shown}}
                               for sid, count in sorted(met.items(),
                                                        key=lambda kv: (-kv[1], id_sort_key(kv[0])))])
        if met_df.empty:
            st.info("This student has not been in a team yet.")
        else:
            st.dataframe(met_df, width="stretch", hide_index=True, height=320)


st.title("🔄 Cross-Project Analysis")
st.markdown("Analyze team rotations and collaboration patterns across multiple projects.")

with st.expander("ℹ️ What is this page?", expanded=False):
    st.markdown("""
    This page analyzes **how students rotate across multiple projects**.

    - **Exposure**: how many different classmates each student has met, how much the groups of
      an attribute mix (e.g. GROUP_EXAMPLE_SLOT pairings) compared with random teams, repeat
      pairings scored with the allocation rules' contact model, and who has never worked with whom
    - **Project Comparison**: two projects side by side and how many teammates carry over
    - **Rotation Matrix**: which team each student joined in every project
    - **Student Journey**: one student's teammates across all projects
    - **Rotation Quality**: how often pairs repeat, separated into imported history and generated projects
    - **Repeat Pairs**: every pair that met more than once

    **Key terms:**
    - **Repeat pairing**: the same two students in one team in more than one project
    - **Carried-over teammates**: of a student's teammates in one project, how many are teammates again in the other
    """.replace("GROUP_EXAMPLE_SLOT", GROUP_EXAMPLE))

require_results()

alloc = es.active_allocation()
result = alloc.formation_result
projects = alloc.projects
result_config = result.config
scored_rules = es.rules_of_active_reports()
history_rules = scored_rules.history
rules_source = es.rules_source_text()
cap = es.repeat_cap(scored_rules)

if len(projects) < 2:
    st.info("Cross-project analysis requires at least 2 projects. Currently you have 1 project.")
    st.markdown("To generate multiple projects:")
    st.markdown("1. Go to **Configuration**")
    st.markdown("2. Set **Number of Projects** to 2 or more")
    st.markdown("3. Run **Team Formation** again")
    if st.button("Go to Configuration"):
        switch_to("configuration")
    st.stop()

loader = get_data_loader()
students = sorted(loader.get_students(), key=lambda s: id_sort_key(s.student_id))
metrics = DiversityMetrics(result_config, cohort_students=students)
cohort_ids = es.history_ids(roster_frame(loader.get_students(), loader.get_dataframe()))
counts = es.allocation_pair_history(projects, scored_rules, cohort_ids, even_when_off=True)
per_project_repeats, repeat_pairs_df = weighted_repeats(projects, history_rules, cohort_ids)
generated_rows = [not p.imported for p in projects]
repeats_generated = int(per_project_repeats.loc[generated_rows, "Repeat team pairs"].sum())
over_generated = int(per_project_repeats.loc[generated_rows, "Over the cap"].sum())
over_imported = int(per_project_repeats.loc[[not g for g in generated_rows], "Over the cap"].sum())
shared_teams = {pair: c for pair, c in counts.cap_counts.items() if c > 0}
limit = history_rules.ignore_groups_larger_than
repeats_counted = history_rules.enabled

st.subheader("📊 Multi-Project Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Projects", len(projects))
team_counts = sorted({len(p.teams) for p in projects})
col2.metric(
    "Teams per Project",
    team_counts[0] if len(team_counts) == 1 else f"{team_counts[0]}–{team_counts[-1]}",
)
col3.metric("Total Students", len(students))
col4.metric("Repeat cap", es.cap_label(history_rules),
            help=f"Most teams a pair may share — the cap of {rules_source} (Allocation Rules › "
                 "🔁 History & preferences). The classic algorithms' 'Max Repeat Pairings' is "
                 "not used here.")
render_scored_rules_caption("Repeat numbers on this page are counted with",
                            ", the rules the rule reports were scored with")

st.markdown("---")

tab_exposure, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Exposure",
    "📋 Project Comparison",
    "🔗 Rotation Matrix",
    "👥 Student Journey",
    "📈 Rotation Quality",
    "🔁 Repeat Pairs"
])

with tab_exposure:
    render_exposure_tab(projects, students, loader.get_dataframe(), history_rules,
                        rules_source, per_project_repeats, repeat_pairs_df)

with tab1:
    st.subheader("Side-by-Side Project Comparison")
    col1, col2 = st.columns(2)
    with col1:
        project1_idx = st.selectbox(
            "Project 1",
            range(len(projects)),
            format_func=lambda x: project_label(projects[x]),
            key="compare_p1"
        )
    with col2:
        project2_idx = st.selectbox(
            "Project 2",
            range(len(projects)),
            format_func=lambda x: project_label(projects[x]),
            index=min(1, len(projects) - 1),
            key="compare_p2"
        )
    p1 = projects[project1_idx]
    p2 = projects[project2_idx]

    st.markdown("### Team Compositions")
    member_label = _extension_ui("member_label") or (lambda m: f"• {m.student_id}")
    col1, col2 = st.columns(2)
    for column, proj in ((col1, p1), (col2, p2)):
        with column:
            st.markdown(f"#### {project_label(proj)}")
            for team in proj.teams:
                with st.expander(f"{team.team_id} ({team.size} members)"):
                    for member in sorted(team.members, key=lambda m: id_sort_key(m.student_id)):
                        st.write(member_label(member))

    st.markdown("### Teammates Carried Over")
    mates1 = teammates_by_student(p1)
    mates2 = teammates_by_student(p2)
    common_students = sorted(set(mates1) & set(mates2), key=id_sort_key)
    carried = {sid: sorted(mates1[sid] & mates2[sid], key=id_sort_key)
               for sid in common_students}
    if p1 is p2:
        st.info("Select two different projects to compare.")
    elif not common_students:
        st.info("No student takes part in both projects.")
    else:
        fresh_students = sum(1 for sid in common_students if not carried[sid])
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Students in both", len(common_students))
        mcol2.metric("With only new teammates", fresh_students)
        mcol3.metric("Max teammates carried over", max(len(v) for v in carried.values()))
        rows = [
            {
                "Student": sid,
                "Teammates carried over": len(carried[sid]),
                "Who": ", ".join(carried[sid]),
                f"Teammates in {p1.name}": len(mates1[sid]),
                f"Teammates in {p2.name}": len(mates2[sid]),
            }
            for sid in common_students if carried[sid]
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.success("Every student has completely new teammates in the second project.")

    st.markdown("### Pair Overlap")
    p1_pairs = p1.get_all_pairs()
    p2_pairs = p2.get_all_pairs()
    common_pairs = sorted(p1_pairs & p2_pairs, key=lambda p: (id_sort_key(p[0]), id_sort_key(p[1])))
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{p1.name} Pairs", len(p1_pairs))
    col2.metric(f"{p2.name} Pairs", len(p2_pairs))
    col3.metric("Common Pairs", len(common_pairs) if p1 is not p2 else 0)
    if common_pairs and p1 is not p2:
        with st.expander(f"View {len(common_pairs)} Common Pairs"):
            for pair in common_pairs[:50]:
                st.write(f"• {pair[0]} ↔ {pair[1]}")
            if len(common_pairs) > 50:
                st.info(f"... and {len(common_pairs) - 50} more")

with tab2:
    st.subheader("Student Rotation Matrix")
    st.markdown("Track which team each student is assigned to across all projects.")
    rotation_df = rotation_matrix(projects, [s.student_id for s in students])

    search = st.text_input("🔍 Search Student ID")
    if search:
        filtered_df = rotation_df[rotation_df["Student"].astype(str).str.contains(search, case=False, regex=False)]
    else:
        filtered_df = rotation_df
    st.dataframe(filtered_df, width="stretch", height=400, hide_index=True)
    st.download_button(
        "📥 Download Rotation Matrix (CSV)",
        safe_csv(rotation_df),
        "rotation_matrix.csv",
        "text/csv",
        on_click="ignore",
    )

    st.markdown("### Rotation Heatmap")
    st.caption("Each column is a project; colour = team number within that project, light = low "
               "team numbers, dark = high (blank = not participating).")
    color_mode = st.radio(
        "Colours",
        list(HEATMAP_COLOR_MODES),
        format_func=HEATMAP_COLOR_MODES.get,
        horizontal=True,
        key="rotation_heatmap_colors",
        help="One gradient for every project; a different gradient for each project; or one "
             "gradient for the uploaded (imported) projects and another for the upcoming "
             "projects formed here.",
    )

    labels = [project_label(p) for p in projects]
    imported_flags = [p.imported for p in projects]
    z_columns = rotation_team_columns(projects, [s.student_id for s in students])
    if color_mode == "per_project":
        st.caption(" · ".join(
            f"{label}: {HEATMAP_GRADIENTS[g][0]}"
            for label, (_, g) in zip(labels, heatmap_color_groups(imported_flags, color_mode))
        ))
    elif color_mode == "uploaded_vs_upcoming":
        if not any(imported_flags):
            st.caption("No uploaded projects: every column is upcoming (orange).")
        elif all(imported_flags):
            st.caption("No upcoming projects yet: every column is uploaded (blue).")
        else:
            st.caption("Blue = uploaded (imported) projects · orange = upcoming projects formed here.")
    fig = go.Figure(data=rotation_heatmap_traces(
        z_columns, labels, [s.student_id for s in students], imported_flags, color_mode,
    ))
    fig.update_layout(
        title="Student Team Assignments Across Projects",
        height=max(400, len(students) * 15),
        xaxis=dict(type="category", categoryorder="array", categoryarray=labels,
                   **HEATMAP_AXIS),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed", type="category",
                   **HEATMAP_AXIS),
        **HEATMAP_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader("Individual Student Journey")
    student_ids = [s.student_id for s in students]
    selected_student = st.selectbox("Select Student", student_ids)
    if selected_student:
        student = next((s for s in students if s.student_id == selected_student), None)
        if student:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Student Profile")
                st.write(f"- **ID:** {student.student_id}")
                profile_hook = _extension_ui("student_profile")
                if profile_hook is not None:
                    profile_hook(student)
                else:
                    _frame = es.roster_frame(students, loader.get_dataframe()).set_index("student_id")
                    if student.student_id in _frame.index:
                        for _attr, _info in infer_attributes(_frame.reset_index()).items():
                            if _info.usable:
                                st.write(f"- **{_attr.replace('_', ' ').title()}:** "
                                         f"{_frame.at[student.student_id, _attr]}")

            st.markdown("---")
            st.markdown("### Project Journey")
            journey_data = []
            collaborator_counts: Dict[str, int] = {}
            section_mates: Set[str] = set()
            for project in projects:
                for team in project.teams:
                    if selected_student in team.member_ids:
                        teammates = sorted(
                            (m.student_id for m in team.members if m.student_id != selected_student),
                            key=id_sort_key,
                        )
                        if counts_as_team(len(set(team.member_ids)), limit):
                            for mate in teammates:
                                collaborator_counts[mate] = collaborator_counts.get(mate, 0) + 1
                        else:
                            section_mates.update(teammates)
                        team_score = classic_team_score(metrics, team)
                        journey_data.append({
                            "Project": project_label(project),
                            "Team": team.team_id,
                            "Team Size": team.size,
                            "Teammates": ", ".join(teammates),
                            "Diversity Score": round(team_score, 3),
                        })
                        break
            if journey_data:
                st.dataframe(pd.DataFrame(journey_data), width="stretch", hide_index=True)
                st.markdown("### Collaboration Summary")
                st.metric("Unique Teammates", len(collaborator_counts),
                          help="Across groups that count as teams (sections excluded).")
                repeat_collabs = {k: v for k, v in collaborator_counts.items() if v > 1}
                if repeat_collabs:
                    st.markdown("#### Repeat Teammates" if repeats_counted
                                else "#### Teammates in more than one team")
                    if not repeats_counted:
                        st.caption(HISTORY_OFF_NOTE)
                    for collab_id, count in sorted(repeat_collabs.items(), key=lambda x: (-x[1], id_sort_key(x[0]))):
                        over = " — over the cap" if cap is not None and count > cap else ""
                        st.write(f"• **{collab_id}**: {count} teams together{over}")
                else:
                    st.success("No repeat teammates: every team-mate is new.")
                reunited = sorted(section_mates & set(collaborator_counts), key=id_sort_key)
                if reunited:
                    st.caption(f"Section-mates reunited in a team: {', '.join(reunited)} (a weak "
                               "contact, not a repeat).")
            else:
                st.info("This student is not in any team.")

with tab4:
    st.subheader("Rotation Quality Metrics")
    overlap = result.overlap_matrix or {}
    total_possible_pairs = len(students) * (len(students) - 1) // 2
    met_pairs = len(overlap)
    repeated_team_pairs = sum(1 for c in shared_teams.values() if c > 1)
    max_seen = max(shared_teams.values()) if shared_teams else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        coverage = met_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
        st.metric("Pair Coverage", f"{coverage:.1%}",
                  help="Share of all possible pairs that shared any group (sections included) "
                       "at least once")
    with col2:
        repeat_rate = repeated_team_pairs / len(shared_teams) if shared_teams else 0
        st.metric(ALL_RATE_LABEL, f"{repeat_rate:.1%}",
                  help="Of the pairs who shared a team in any project (imported history "
                       "included; sections do not count), the share that shared more than one. "
                       "Team Formation's 'Repeat pairs' counts the generated projects only."
                       + ("" if repeats_counted else " " + HISTORY_OFF_NOTE))
    with col3:
        st.metric(
            "Most teams shared by a pair", max_seen,
            delta=f"Cap: {es.cap_label(history_rules)}",
            delta_color="inverse" if cap is not None and max_seen > cap else "off",
        )

    st.markdown("---")
    st.subheader("Pairing Frequency Distribution")
    if shared_teams:
        count_dist: Dict[int, int] = {}
        for c in shared_teams.values():
            count_dist[c] = count_dist.get(c, 0) + 1
        fig = px.bar(
            x=list(count_dist.keys()),
            y=list(count_dist.values()),
            title="How Many Teams Pairs Have Shared (sections excluded)",
            labels={"x": "Teams shared", "y": "Number of Pairs"},
            color=list(count_dist.keys()),
            color_continuous_scale=["#5D8972", "#E1AE67", "#BD0226"]
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Rotation Efficiency by Project")
    st.caption("New pairs = teammates who had not shared any earlier group (sections included).")
    eff_df = pair_efficiency(projects)
    earlier = eff_df["Cumulative Unique"] - eff_df["New Pairs"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=eff_df["Project"], y=earlier, name="Unique pairs from earlier projects",
                         marker_color="#B8C2D6"))
    fig.add_trace(go.Bar(x=eff_df["Project"], y=eff_df["New Pairs"], name="New pairs",
                         marker_color="#5D8972", text=eff_df["New Pairs"], textposition="inside"))
    fig.update_layout(title="Cumulative Unique Pairs (new pairs per project on top)",
                      barmode="stack", yaxis_title="Pairs",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(eff_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("💡 Recommendations")
    st.caption(f"Repeat numbers below are counted with {rules_source}, the rules the reports "
               "were scored with (the same numbers as Team Formation's headline).")
    if over_generated:
        st.error(
            f"❌ {over_generated} pair(s) in generated projects share more teams than the cap "
            f"({es.cap_label(history_rules)}) allows (see 🔁 Repeat Pairs)."
        )
    elif repeats_generated:
        st.warning(f"⚠️ {repeats_generated} repeat pair(s) in generated projects (within the "
                   "cap). Smaller teams (more teams per project) leave more fresh partners.")
    elif any(generated_rows):
        st.success("✅ No repeat pairs in the generated projects.")
    if over_imported:
        st.info(
            f"ℹ️ {over_imported} pair(s) already exceed the cap within the imported history "
            "alone; locked projects cannot be changed."
        )
    if not history_rules.enabled:
        st.info("History is switched off in the allocation rules, so repeats are neither "
                "counted nor penalised (Allocation Rules › 🔁 History & preferences).")

with tab5:
    st.subheader("Repeat Pair Analytics")
    st.caption(f"Shared teams per pair, counted with {rules_source} (groups they treat as "
               "sections excluded); the table lists every pair that met more than once in any "
               "group.")
    ids = cohort_ids
    on_roster = set(ids)
    in_teams = {sid for p in projects for t in project_teams(p) for sid in t}
    all_student_ids = sorted(in_teams & on_roster, key=id_sort_key)
    left = len(in_teams - on_roster)

    active_summary = es.summarize_reports(es.active_reports()) if es.active_reports() else None
    repeats_row = None
    if active_summary:
        repeats_row = next((r for r in headline_rows(active_summary, es.active_reports())
                            if r.key == "repeats"), None)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", len(all_student_ids),
                help="Roster students in a team of any project"
                     + (f"; {left} student(s) who left the roster are not counted." if left
                        else "."))
    col2.metric("Repeat pairs",
                repeats_row.text if repeats_row is not None else str(repeats_generated),
                help=repeats_row.help if repeats_row is not None and repeats_row.help else
                     "Teammates in a generated project who already shared a team before: the "
                     "same number as on Home, Team Formation, Analytics and Export.")
    col3.metric(ALL_PAIRS_LABEL, repeated_team_pairs,
                help="Pairs who shared more than one team in any project, imported history "
                     "included (sections excluded)."
                     + ("" if repeats_counted else " " + HISTORY_OFF_NOTE))
    if not repeats_counted:
        st.info(HISTORY_OFF_NOTE + " The figures below only describe who shared teams.")
    col4.metric(
        "Pairs that can't meet again",
        sum(1 for c in shared_teams.values() if cap is not None and c >= cap),
        help=f"Pairs that already shared {es.cap_label(history_rules)} team(s) — the repeat cap — so "
             "a future project may not place them together again.",
    )

    explanations = (es.explain_repeat_pairs(projects, result_config, loader.get_students(),
                                            policy=scored_rules,
                                            raw_df=loader.get_dataframe())
                    if repeats_counted else [])
    st.subheader("Why these repeat pairs?")
    if not repeats_counted:
        st.caption("History is switched off, so there are no repeat pairs to explain.")
    elif not explanations:
        st.success("No repeat pairs in the generated projects — nothing to explain.")
    else:
        st.markdown(summarize_explanations(explanations))
        st.dataframe(repeat_explanation_table(explanations, projects), width="stretch",
                     hide_index=True,
                     column_config={"Reason": st.column_config.TextColumn(width="large"),
                                    "Suggested fix": st.column_config.TextColumn(width="large")})
        st.caption("Checked in this order: requested together (pair rule or preference), "
                   "pinned in the Team Editor, placed by hand, then every swap (and move, with "
                   "flexible team sizes) of either student, also followed by a second swap, "
                   "scored exactly like the reports. 'Unavoidable' means no such change "
                   "separates them without another repeat or a broken hard rule.")

    repeat_mode = st.radio(
        "Colours",
        list(REPEAT_HEATMAP_COLOR_MODES),
        format_func=REPEAT_HEATMAP_COLOR_MODES.get,
        horizontal=True,
        key="repeat_heatmap_colors",
        help="One gradient for every pair; a gradient per project, by the project in which the "
             "pair most recently shared a team; or blue for pairs who only met in uploaded "
             "(imported) projects and orange for pairs who met in an upcoming project.",
    )
    labels = [project_label(p) for p in projects]
    imported_flags = [p.imported for p in projects]
    if len(all_student_ids) > REPEAT_HEATMAP_MAX_STUDENTS:
        st.info(f"The student × student heatmap covers at most {REPEAT_HEATMAP_MAX_STUDENTS} "
                f"students; this allocation has {len(all_student_ids)}. The tables below "
                "list the same pairs.")
    else:
        met_in_projects = pair_shared_projects(projects, scored_rules, ids)
        trace, _groups = repeat_heatmap_trace(
            all_student_ids, ids, shared_teams, met_in_projects, labels, imported_flags,
            repeat_mode, why=repeat_reasons(explanations, ids),
        )
        if repeat_mode == "single":
            st.caption("Colour = teams the pair shared, light = one, dark = most (blank = never in "
                       "a team together). Hover a tile for the projects (and, for a repeat pair, "
                       "why).")
        elif repeat_mode == "per_project":
            st.caption("Colour = the project in which the pair most recently shared a team, darker "
                       "= more shared teams: " + " · ".join(
                           f"{label}: {HEATMAP_GRADIENTS[k % len(HEATMAP_GRADIENTS)][0]}"
                           for k, label in enumerate(labels)))
        elif not any(imported_flags):
            st.caption("No uploaded projects: every pair met in an upcoming project (orange), "
                       "darker = more shared teams.")
        elif all(imported_flags):
            st.caption("No upcoming projects yet: every pair met in uploaded projects (blue), "
                       "darker = more shared teams.")
        else:
            st.caption("Blue = pairs who only met in uploaded (imported) projects · orange = pairs "
                       "who met in at least one upcoming project formed here; darker = more "
                       "shared teams.")
        student_axis = [str(s) for s in all_student_ids]
        fig = go.Figure(trace)
        fig.update_layout(
            height=900,
            xaxis_title="Student",
            yaxis_title="Student",
            yaxis=dict(autorange="reversed", type="category", categoryorder="array",
                       categoryarray=student_axis, **HEATMAP_AXIS),
            xaxis=dict(showticklabels=False, type="category", categoryorder="array",
                       categoryarray=student_axis, **HEATMAP_AXIS),
            **HEATMAP_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Pairs Meeting More Than Once")
    if repeat_pairs_df.empty:
        st.success("No pair was placed together more than once.")
    else:
        table = repeat_pairs_df.drop(columns=["Contact weight"]).copy()
        imported_labels = {project_label(p) for p in projects if p.imported}
        table["Origin"] = [
            "imported only" if all(label in imported_labels for label in names.split(", "))
            else "includes generated"
            for names in table["Projects"]
        ]
        st.dataframe(table, width="stretch", hide_index=True)

    st.subheader("Per-Student Repeat Summary")
    st.dataframe(per_student_repeat_summary(projects, ids, shared_teams),
                 width="stretch", hide_index=True)

    st.subheader("Distribution of Teams Shared")
    dist_df = teams_shared_distribution(shared_teams)
    if not dist_df.empty:
        fig_dist = px.bar(dist_df, x="Teams together", y="Number of pairs", text="Number of pairs")
        fig_dist.update_layout(height=350)
        st.plotly_chart(fig_dist, width="stretch")
    else:
        st.info("No pairs to distribute.")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Analytics", width="stretch"):
        switch_to("analytics")
with col2:
    if st.button("🎯 Team Formation", width="stretch"):
        switch_to("formation")
with col3:
    if st.button("📤 Export →", width="stretch"):
        switch_to("export")
