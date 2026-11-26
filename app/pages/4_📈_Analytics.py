import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core import run_activation
from app.core import run_history
from app.core.analysis_views import (  # noqa: E402
    MAX_DEEP_DIVE_VALUES,
    available_attributes,
    attribute_value,
    bucket_value,
    capped_values,
    deep_dive_team_entries,
    distribution_frame,
    diversity_score_frame,
    fairness_view,
    network_view,
)
from app.core.classic import parameter_echo
from app import extensions  # noqa: E402
from app.core.data_session import (
    departed_warning,
    ensure_session_state,
    get_data_loader,
    require_results,
    switch_to,
)
from app.core.diversity_metrics import DiversityMetrics
from app.core.engine_session import (
    active_allocation,
    active_reports,
    cap_label,
    evaluate_projects,
    report_matches_teams,
    roster_frame,
    rules_of_active_reports,
    rules_source_text,
    section_reunions,
    show_section_reunions,
    summarize_reports,
)
from app.engine.report_views import (  # noqa: E402
    TeamView,
    attribute_rules,
    history_section,
    overall,
    preferences_section,
    rule_entries,
    teams,
    warnings,
)
from app.pages._views import (  # noqa: E402
    classic_score_bar,
    classic_team_score,
    render_headline_row,
    render_scored_rules_caption,
)
from app.utils.charts import category_color_map
from app.utils.exporters import pretty_team_name
from app.utils.fmt import fmt_thousands
from app.utils.headline import SECTION_MATES_HELP, repeat_cap_text  # noqa: E402
from app.utils.roster_view import MISSING_LABEL, RosterView  # noqa: E402

st.set_page_config(
    page_title="Analytics | Group Diversifier",
    page_icon="📈",
    layout="wide",
)

ensure_session_state()


IMPORT_PATTERNS = {"imported": "/", "generated": ""}
AUDIT_TEAMS = 5
MAX_CHART_VALUES = 8
QUALITY_COLORS = ((33.0, "#BD0226"), (66.0, "#E1AE67"), (101.0, "#5D8972"))


def team_label_map(project) -> Dict[frozenset, str]:
    if project is None:
        return {}
    return {frozenset(str(m.student_id) for m in t.members): pretty_team_name(t.team_id)
            for t in project.teams}


def team_axis(fig, team_order: List[str], axis: str = "x") -> None:
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(categoryorder="array", categoryarray=list(team_order))


def quality_color(pct: float) -> str:
    return next(color for limit, color in QUALITY_COLORS if pct < limit)


def render_project_compliance(pid: str, report: Dict[str, Any], project, view: RosterView) -> None:
    ova = overall(report)
    history = history_section(report)
    prefs = preferences_section(report)
    team_views = teams(report)
    labels = team_label_map(project)

    def team_name(team: TeamView) -> str:
        return labels.get(frozenset(str(s) for s in team.members or []),
                          team.name or f"Team {(team.index or 0) + 1}")

    quality = ova.quality_index_soft
    hard_units = int(ova.hard_units or 0)
    sections = show_section_reunions({}, {pid: report})
    c1, c2, c3, c4, c5, *c6 = st.columns(6 if sections else 5)
    c1.metric("Quality vs random", "–" if quality is None else f"{float(quality):.1f}%",
              help="How much lower the soft penalty is than the average random allocation with "
                   "the same team sizes (100% = no imbalance at all, 0% = random level). Hard "
                   "rules are left out so a single broken cap cannot hide everything else.")
    c2.metric("Objective", fmt_thousands(ova.objective),
              help="Total penalty points the optimiser minimised (lower is better).")
    c3.metric("Worst team", fmt_thousands(ova.worst_team_penalty, 2),
              delta=f"mean {fmt_thousands(ova.mean_team_penalty, 2)}", delta_color="off")
    c4.metric("Hard violations", hard_units,
              help="Units of hard-rule violation (each costs 1000 points); 0 = every hard "
                   "rule holds.")
    repeat_text = repeat_cap_text(int(history.repeated_pairs or 0),
                                  int(history.over_cap_pairs or 0))
    c5.metric("Repeat pairs", repeat_text, help="Pairs who already shared a team before.")
    if sections:
        c6[0].metric("Section-mates reunited", section_reunions({}, {pid: report}),
                     help=SECTION_MATES_HELP)
    baseline = ova.random_baseline_soft
    if baseline.samples:
        st.caption(
            f"Random allocations with the same team sizes: soft penalty {fmt_thousands(baseline.mean)} "
            f"± {fmt_thousands(baseline.std)} (best of {baseline.samples}: {fmt_thousands(baseline.best)}); "
            f"{fmt_thousands(baseline.percentile_of_solution, 0)}% of them beat this allocation.")
    if prefs.prefer_total or prefs.avoid_total:
        st.caption(f"Student preferences: {prefs.prefer_satisfied or 0}/{prefs.prefer_total or 0} "
                   f"'prefer' wishes met · {prefs.avoid_violated or 0} 'avoid' wishes broken.")
    for warning in warnings(report):
        st.warning(warning)

    rule_views = rule_entries(report)
    group_rules = [r for r in rule_views if r.kind != "attribute"]
    attribute_rule_views = [r for r in rule_views if r.kind == "attribute"]
    if group_rules:
        st.markdown("##### Rules")
        st.dataframe(pd.DataFrame([{
            "Rule": r.label or "",
            "Type": r.kind or "",
            "Hard": "hard" if r.hard else "soft",
            "Weight": r.weight,
            "Teams compliant": f"{r.teams_compliant or 0}/{r.teams_total or 0}",
            "Compliance": float(r.compliance_pct or 0.0),
            "Violation units": float(r.units or 0),
            "Points": float(r.points or 0.0),
        } for r in group_rules]), width="stretch", hide_index=True, column_config={
            "Compliance": st.column_config.ProgressColumn(
                "Compliance", min_value=0.0, max_value=100.0, format="%.0f%%"),
            "Points": st.column_config.NumberColumn("Points", format="%.2f"),
        })
    if attribute_rule_views:
        st.markdown("##### Attribute goals")
        labels_q = [r.label or "" for r in attribute_rule_views]
        values_q = [float(r.quality_pct or 0.0) for r in attribute_rule_views]
        fig = go.Figure(go.Bar(
            x=values_q, y=labels_q, orientation="h",
            marker_color=[quality_color(v) for v in values_q],
            text=[f"{v:.0f}%" for v in values_q], textposition="auto",
            customdata=[[r.weight, r.points] for r in attribute_rule_views],
            hovertemplate="%{y}: %{x:.1f}%<br>weight %{customdata[0]} · %{customdata[1]:.2f} "
                          "points<extra></extra>",
        ))
        fig.update_layout(xaxis=dict(range=[0, 100], title="Quality (100% = perfect mix, "
                                                           "0% = random level)"),
                          yaxis=dict(autorange="reversed"), margin=dict(t=10),
                          height=max(220, 34 * len(attribute_rule_views) + 80))
        st.plotly_chart(fig, width="stretch", key=f"compliance_quality_{pid}")

    if team_views:
        st.markdown("##### Team penalties")
        penalties = [float(t.penalty or 0.0) for t in team_views]
        fig = px.histogram(x=penalties, nbins=min(30, max(8, len(penalties))),
                           labels={"x": "Team penalty (points)", "y": "Teams"},
                           color_discrete_sequence=["#2871AD"])
        n_teams = max(1, len(team_views))
        fig.add_vline(x=float(ova.mean_team_penalty or 0.0), line_dash="dash",
                      line_color="#171C54", annotation_text="mean team",
                      annotation_position="top left")
        soft_baseline = ova.random_baseline_soft
        random_mean = soft_baseline.mean
        if random_mean is not None:
            fig.add_vline(x=float(random_mean) / n_teams, line_dash="dot", line_color="#BD0226",
                          annotation_text="random allocation (per team)",
                          annotation_position="top right")
            soft_objective = ova.soft_objective if ova.soft_objective is not None else ova.objective
            soft_objective = soft_objective or 0.0
            fig.add_vline(x=float(soft_objective) / n_teams, line_dash="dot",
                          line_color="#5D8972", annotation_text="this allocation (per team)",
                          annotation_position="bottom right")
        fig.update_layout(yaxis_title="Teams", margin=dict(t=30), bargap=0.05)
        st.plotly_chart(fig, width="stretch", key=f"compliance_penalties_{pid}")
        percentile = soft_baseline.percentile_of_solution
        if percentile is not None:
            st.caption(f"Per-team markers are the soft objective ÷ number of teams (the objective "
                       f"also weighs the worst team, so they can sit right of the mean); "
                       f"{fmt_thousands(percentile, 0)}% of {soft_baseline.samples or 0} random "
                       f"allocations score better than this one.")

        st.markdown(f"##### Worst {min(AUDIT_TEAMS, len(team_views))} teams")
        worst = sorted(team_views, key=lambda t: -float(t.penalty or 0.0))[:AUDIT_TEAMS]
        audit_attrs = [a for a in (worst[0].composition or {}) if a in view.frame.columns][:4]
        for rank, team in enumerate(worst):
            issues = team.issues or []
            title = (f"{team_name(team)} · penalty {fmt_thousands(team.penalty, 2)} · "
                     f"{len(issues)} issue{'s' if len(issues) != 1 else ''}")
            with st.expander(title, expanded=rank == 0):
                left, right = st.columns([3, 2])
                with left:
                    breakdown = {k: float(v) for k, v in (team.breakdown or {}).items()
                                 if float(v) > 1e-9}
                    if breakdown:
                        items = sorted(breakdown.items(), key=lambda kv: kv[1])
                        bar = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items],
                                               orientation="h", marker_color="#2871AD"))
                        bar.update_layout(height=max(160, 26 * len(items) + 60),
                                          margin=dict(t=10, b=10), xaxis_title="Points")
                        st.plotly_chart(bar, width="stretch",
                                        key=f"compliance_breakdown_{pid}_{team.index}")
                    else:
                        st.success("No penalty points.")
                with right:
                    members = [str(s) for s in team.members or []]
                    st.dataframe(pd.DataFrame([
                        {"Student": sid, **{a: view.text(sid, a) for a in audit_attrs}}
                        for sid in members
                    ]), width="stretch", hide_index=True)
                for issue in issues:
                    (st.error if issue.get("hard") else st.warning)(issue.get("text", ""))


def render_composition_views(pid: str, report: Dict[str, Any], project, view: RosterView) -> None:
    team_views = teams(report)
    if not team_views:
        return
    labels = team_label_map(project)
    team_names = [labels.get(frozenset(str(s) for s in t.members or []),
                             t.name or f"Team {k + 1}") for k, t in enumerate(team_views)]
    members_of = [[str(s) for s in t.members or []] for t in team_views]
    cohort_ids = [sid for members in members_of for sid in members]
    attrs = view.usable_attributes()
    rule_attrs = [r.attribute for r in attribute_rules(report)
                  if r.attribute in attrs]
    categorical = [a for a in attrs if view.kind(a) in ("categorical", "multi")]
    numeric = [a for a in attrs if view.kind(a) == "numeric"]

    st.markdown("##### Team composition cards")
    default_cards = [a for a in rule_attrs if a in categorical or a in numeric][:2] or attrs[:2]
    chosen = st.multiselect("Attributes on the cards", attrs, default=default_cards,
                            key=f"compliance_card_attrs_{pid}")
    if chosen:
        per_row = 4
        for start in range(0, len(team_views), per_row):
            columns = st.columns(per_row)
            for column, k in zip(columns, range(start, min(start + per_row, len(team_views)))):
                team = team_views[k]
                with column.container(border=True):
                    flag = " ⚠️" if team.violations else ""
                    st.markdown(f"**{team_names[k]}**{flag} · {len(members_of[k])} · "
                                f"penalty {fmt_thousands(team.penalty, 2)}")
                    lines = [f"**{a}**: {view.summary(members_of[k], a)}" for a in chosen]
                    st.markdown("  \n".join(lines))

    if categorical:
        st.markdown("##### Team mix vs cohort")
        default_dist = next((a for a in rule_attrs if a in categorical), categorical[0])
        attr = st.selectbox("Attribute", categorical, index=categorical.index(default_dist),
                            key=f"compliance_dist_attr_{pid}")
        cohort = view.cohort_counts(attr, cohort_ids)
        shown = [v for v in cohort][:MAX_CHART_VALUES]
        rows = []
        for name, members in [("Cohort", cohort_ids)] + list(zip(team_names, members_of)):
            counts = view.counts(members, attr)
            total = sum(counts.values()) or 1
            other = sum(c for v, c in counts.items() if v not in shown)
            for value in shown:
                rows.append({"Team": name, "Value": value, "Share": counts.get(value, 0) / total,
                             "Count": counts.get(value, 0)})
            if other:
                rows.append({"Team": name, "Value": "other", "Share": other / total, "Count": other})
        fig = px.bar(pd.DataFrame(rows), x="Team", y="Share", color="Value", hover_data=["Count"],
                     barmode="stack", category_orders={"Team": ["Cohort"] + team_names})
        total = sum(cohort.values()) or 1
        cumulative = 0.0
        for value in shown[:-1]:
            cumulative += cohort[value] / total
            fig.add_hline(y=cumulative, line_dash="dot", line_color="#555", line_width=1)
        fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]), margin=dict(t=10),
                          legend_title_text=attr)
        st.plotly_chart(fig, width="stretch", key=f"compliance_dist_{pid}")
        st.caption("Dotted lines mark the cohort's cumulative shares: a team that mirrors the "
                   "cohort has its colour boundaries on the lines.")

    if numeric:
        st.markdown("##### Team averages vs cohort")
        default_num = next((a for a in rule_attrs if a in numeric), numeric[0])
        attr = st.selectbox("Numeric attribute", numeric, index=numeric.index(default_num),
                            key=f"compliance_numeric_attr_{pid}")
        values = [x for x in (view.number(sid, attr) for sid in cohort_ids) if x is not None]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values))
            means = [view.mean(m, attr) for m in members_of]
            fig = go.Figure(go.Bar(x=team_names, y=means, marker_color="#2871AD",
                                   hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
            typical = float(np.mean([len(m) for m in members_of])) or 1.0
            band = std / np.sqrt(typical)
            fig.add_hrect(y0=mean - band, y1=mean + band, fillcolor="#5D8972", opacity=0.15,
                          line_width=0, annotation_text="±1 s.e. of a random team",
                          annotation_position="top left")
            fig.add_hline(y=mean, line_dash="dash", line_color="#171C54",
                          annotation_text=f"cohort mean {mean:.2f}")
            low = min([m for m in means if m is not None] + [mean - band])
            high = max([m for m in means if m is not None] + [mean + band])
            pad = (high - low) * 0.15 or 1.0
            fig.update_layout(yaxis=dict(range=[low - pad, high + pad], title=attr),
                              margin=dict(t=10))
            st.plotly_chart(fig, width="stretch", key=f"compliance_numeric_{pid}")


def render_rule_compliance(reports: Dict[str, Dict[str, Any]], projects, students, raw_df,
                           selected=None) -> None:
    by_id = {str(p.project_id): p for p in projects}
    pids = [str(p.project_id) for p in projects if str(p.project_id) in reports]
    pids += [pid for pid in reports if pid not in pids]
    names = {pid: (by_id[pid].name if pid in by_id else f"Project {pid}") for pid in pids}
    view = RosterView(roster_frame(students, raw_df))
    render_scored_rules_caption()

    if len(pids) > 1:
        summary = summarize_reports(reports)
        sections = show_section_reunions(summary, reports)
        render_headline_row(summary, reports)
        rows = []
        for pid in pids:
            scored = overall(reports[pid])
            row = {
                "Project": names[pid],
                "Quality vs random": scored.quality_index_soft,
                "Objective": scored.objective,
                "Worst team": scored.worst_team_penalty,
                "Hard violations": scored.hard_units,
                "Repeat pairs": history_section(reports[pid]).repeated_pairs,
            }
            if sections:
                one = {pid: reports[pid]}
                row["Section-mates reunited"] = (section_reunions({}, one)
                                                 if show_section_reunions({}, one) else None)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "Quality vs random": st.column_config.NumberColumn(format="%.1f%%"),
            "Objective": st.column_config.NumberColumn(format="%.1f"),
            "Worst team": st.column_config.NumberColumn(format="%.2f"),
        })

    default_pid = str(selected.project_id) if selected is not None else pids[0]
    pid = st.selectbox("Scored project", pids, index=pids.index(default_pid) if default_pid in pids else 0,
                       format_func=lambda p: names[p], key="compliance_project")
    st.markdown(f"#### {names[pid]}")
    render_project_compliance(pid, reports[pid], by_id.get(pid), view)
    render_composition_views(pid, reports[pid], by_id.get(pid), view)


st.title("📈 Analytics")
st.markdown("Detailed analysis of team formation results, diversity metrics, and fairness.")

with st.expander("ℹ️ Understanding Analytics Metrics (Click to learn)", expanded=False):
    st.markdown(extensions.text("analytics.help", """
    ### What the analytics show

    - **Rule Compliance**: the rule-based scores (quality vs random, hard violations,
      repeat pairs) and how every rule and team fares.
    - **Diversity Deep Dive**: how each attribute's values spread over the teams.
    - **Network View**: who worked with whom across the projects.
    - **Algorithm Performance**: the solver's statistics and the settings used.
    """))

require_results()

alloc = active_allocation()
result = alloc.formation_result
projects = alloc.projects
result_config = result.config

loader = get_data_loader()
students = loader.get_students()
metrics = DiversityMetrics(result_config, cohort_students=students)

roster_ids = {s.student_id for s in students}
departed_warning(projects, [s.student_id for s in students],
                 dataset_note="Cohort comparisons below use the current dataset.")

ALL_PROJECTS_LABEL = "📊 All projects (aggregates)"
project_labels = [
    f"Project {p.project_id} (imported)" if p.imported else f"Project {p.project_id}"
    for p in projects
]
imported_projects = [p for p in projects if p.imported]
selected_project_label = st.selectbox(
    "Project",
    options=[ALL_PROJECTS_LABEL] + project_labels,
    key="analytics_project",
)
if selected_project_label == ALL_PROJECTS_LABEL:
    project = None
else:
    project = projects[project_labels.index(selected_project_label)]
scoped_projects = projects if project is None else [project]
st.caption("Team sizes and the Diversity Deep Dive follow this selection; Fairness and Network are cross-project by nature.")

tab_rules, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✅ Rule Compliance",
    "📊 Overview",
    "🎯 Diversity Deep Dive",
    "⚖️ Fairness Analysis",
    "🔗 Network View",
    "📉 Algorithm Performance"
])

with tab_rules:
    st.subheader("Rule Compliance")
    status = st.container()
    reports = active_reports()
    label = "🧮 Score with the current rules" if not reports else "🔄 Re-score with the current rules"
    if st.button(label, key="analytics_evaluate",
                 help="Evaluates the active teams with the policy on the Allocation Rules page "
                      "(no teams are changed). The result is saved as a new run, so the run the "
                      "teams came from keeps the rules it was formed with."):
        with st.spinner("Scoring teams…"):
            try:
                new_reports = evaluate_projects(projects, st.session_state.config, students,
                                                raw_df=loader.get_dataframe())
            except Exception as exc:
                status.error(f"Could not score the allocation: {exc}")
                new_reports = None
        if new_reports:
            parent = run_history.active_run()
            stored = run_activation.record_rescored_allocation(
                new_reports, students, raw_df=loader.get_dataframe())
            reports = new_reports
            status.success(
                f"Scored {len(new_reports)} project(s) with the current rules and saved them as "
                f"run **{stored.run_id}** (now active)"
                + (f"; {parent.run_id} keeps the rules it was formed with." if parent else "."))
        elif new_reports is not None:
            status.info("No generated project to score (imported projects are history only).")
    generated = [p for p in projects if not p.imported]
    stale = [p.name for p in generated
             if str(p.project_id) in reports
             and not report_matches_teams(p, reports[str(p.project_id)], roster_ids)]
    unscored = [p.name for p in generated if str(p.project_id) not in reports]
    if not reports:
        status.info("This allocation has no rule report yet (it was produced by a legacy "
                    "algorithm, imported or loaded from a session). Score it with the allocation "
                    "rules of the current configuration to see rule compliance, quality versus "
                    "random and the worst teams.")
    elif stale:
        status.warning(f"The teams of {', '.join(stale)} changed after they were scored — the "
                       "numbers below describe the earlier teams. Re-score to update them.")
    elif unscored:
        status.info(f"{', '.join(unscored)} {'has' if len(unscored) == 1 else 'have'} not "
                    "been scored.")
    if reports:
        render_rule_compliance(reports, projects, students, loader.get_dataframe(),
                               selected=project)

with tab1:
    st.subheader("Results Overview")
    if imported_projects:
        st.caption(
            f"Totals include {len(imported_projects)} imported (locked) project(s) — "
            f"generated and imported teams are aggregated together."
        )
    n_teams_total = sum(len(p.teams) for p in projects)
    if reports:
        overview = summarize_reports(reports)
        render_headline_row(overview, reports, trailing=(
            ("Projects", len(projects), None),
            ("Execution Time", f"{result.execution_time:.2f}s",
             ("Duration of the formation run; imported (locked) projects require no "
              "runtime" if imported_projects else None)),
        ))
        render_scored_rules_caption()
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Projects", len(projects))
        col2.metric("Execution Time", f"{result.execution_time:.2f}s",
                    help=("Duration of the formation run; imported (locked) projects require no "
                          "runtime" if imported_projects else None))
        col3.metric("Teams", n_teams_total)
        st.info("No rule-based scores yet: score the allocation on the ✅ Rule Compliance tab to "
                "see quality vs random, hard violations and repeat pairs.")

    st.markdown("---")
    st.subheader("Classic Diversity Metrics")
    st.caption("The legacy display index of the classic algorithms (entropy-based, 0-1 per team; "
               "attributes missing from the data still count). The rules engine does not "
               "optimise it and it can rank runs differently from quality vs random — compare "
               "runs by the rule-based numbers.")
    ccol1, ccol2 = st.columns(2)
    ccol1.metric("Classic diversity score (sum)", f"{result.total_score:.2f}",
                 help="Sum of weighted classic diversity scores across all teams")
    ccol2.metric("Classic score per team", f"{result.total_score / max(1, n_teams_total):.3f}",
                 help="Average classic diversity score per team")
    shown_scores = extensions.without_inactive_metrics(result.diversity_scores)
    if shown_scores:
        categories = list(shown_scores.keys())
        values = list(shown_scores.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Diversity Score',
            line_color='#171C54',
            fillcolor='rgba(23, 28, 84, 0.3)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Classic Diversity Profile"
        )
        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team Size Distribution")
        all_sizes = [team.size for proj in scoped_projects for team in proj.teams]
        if all_sizes:
            fig = px.histogram(
                x=all_sizes,
                nbins=max(all_sizes) - min(all_sizes) + 1,
                title="Distribution of Team Sizes",
                labels={"x": "Team Size", "y": "Count"},
                color_discrete_sequence=["#2871AD"]
            )
            targets: Dict[int, List[str]] = {}
            for proj in scoped_projects:
                if proj.imported:
                    continue
                try:
                    idx = int(proj.project_id) - 1
                except (TypeError, ValueError):
                    idx = 0
                target = result_config.get_team_sizes_for_project(idx)[2]
                targets.setdefault(int(target), []).append(proj.project_id)
            for target, pids in sorted(targets.items()):
                label = "Target" if len(targets) == 1 else f"Target P{', P'.join(pids)}"
                fig.add_vline(x=target, line_dash="dash", annotation_text=label, line_color="#5D8972")
            st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Classic Score Distribution")
        team_scores = [
            classic_team_score(metrics, team)
            for proj in scoped_projects
            for team in proj.teams
        ]
        fig = px.histogram(
            x=team_scores,
            nbins=20,
            title="Distribution of Classic Team Diversity Scores",
            labels={"x": "Diversity Score", "y": "Count"},
            color_discrete_sequence=["#5D8972"]
        )
        st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("Diversity Analysis by Attribute")

    team_entries = deep_dive_team_entries(projects, project)
    team_order = list(dict.fromkeys(team_label for team_label, _, _ in team_entries))
    has_imported_teams = any(imported for _, _, imported in team_entries)
    members = [m for _, team, _ in team_entries for m in team.members]

    attribute_options = available_attributes(members)
    if not attribute_options:
        st.info("The teams' students have no categorical attributes to analyse.")
    else:
        selected_attr = st.selectbox(
            "Select Attribute",
            options=list(attribute_options.keys()),
            format_func=lambda x: attribute_options[x][0],
            key="analytics_attribute",
        )
        attr_label, score_key = attribute_options[selected_attr]
        pattern_kwargs = (
            {"pattern_shape": "Imported", "pattern_shape_map": IMPORT_PATTERNS}
            if has_imported_teams else {}
        )

        all_values = [attribute_value(m, selected_attr) or MISSING_LABEL
                      for proj in projects for team in proj.teams for m in team.members]
        kept_values = capped_values(all_values)
        value_colors = category_color_map(bucket_value(v, kept_values) for v in all_values)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {attr_label} Distribution per Team")
            dist_df = distribution_frame(team_entries, selected_attr, keep=kept_values)
            if kept_values is not None:
                grouped = len({v for v in all_values if v not in kept_values})
                st.caption(f"The {MAX_DEEP_DIVE_VALUES} most common values have their own colour; "
                           f"{grouped} less common value(s) are grouped as Other.")
            if not dist_df.empty:
                fig = px.bar(
                    dist_df, x="Team", y="Count", color="Value",
                    title=f"{attr_label} by Team", barmode="stack",
                    color_discrete_map=value_colors,
                    category_orders={"Team": team_order, "Value": list(value_colors)},
                    **pattern_kwargs,
                )
                team_axis(fig, team_order)
                st.plotly_chart(fig, width="stretch", key="deep_dive_distribution")

        with col2:
            st.markdown(f"### {attr_label} Diversity Scores (classic index)")
            scores_df = diversity_score_frame(team_entries, metrics, selected_attr, score_key)
            fig = classic_score_bar(scores_df, x="Team", y="Diversity Score",
                                    title=f"{attr_label} Diversity by Team",
                                    category_orders={"Team": team_order},
                                    **pattern_kwargs)
            team_axis(fig, team_order)
            st.plotly_chart(fig, width="stretch", key="deep_dive_scores")

        if has_imported_teams:
            st.caption("Hatched (/) bars belong to imported projects — imported = locked history "
                       "(metrics computed from recorded teams).")


with tab3:
    st.subheader("Fairness Analysis")
    st.markdown("Ensuring no student is consistently placed in low or high diversity teams.")
    st.caption("Based on the classic diversity index (legacy display score), not on the "
               "rule-based penalties.")
    fair = fairness_view(metrics, projects)
    if not fair.student_avgs:
        st.info("No per-student data to analyse.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fairness Score", f"{fair.fairness['fairness_score']:.3f}",
                      help="1 − variance of per-student average scores (1.0 = identical experience)")
            st.metric("Score Variance", f"{fair.fairness['overall_variance']:.4f}",
                      help="Lower is better - indicates consistent experience")
        with col2:
            st.metric("Min Average Score", f"{fair.fairness['min_average']:.3f}",
                      help="Lowest average diversity score experienced by any student")
            st.metric("Max Average Score", f"{fair.fairness['max_average']:.3f}",
                      help="Highest average diversity score experienced by any student")

        st.markdown("---")
        st.subheader("Student Experience Distribution")
        fig = px.histogram(
            x=fair.student_avgs,
            nbins=20,
            title="Distribution of Average Diversity Scores per Student",
            labels={"x": "Average Diversity Score", "y": "Number of Students"},
            color_discrete_sequence=["#2871AD"]
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Outlier Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Students with Lower Diversity Experience")
            if fair.outliers_low:
                for sid, stats in fair.outliers_low[:10]:
                    st.write(f"• **{sid}**: avg={stats['average']:.3f}")
            else:
                st.success("No significant outliers on the low end")
        with col2:
            st.markdown("### Students with Higher Diversity Experience")
            if fair.outliers_high:
                for sid, stats in fair.outliers_high[:10]:
                    st.write(f"• **{sid}**: avg={stats['average']:.3f}")
            else:
                st.success("No significant outliers on the high end")

with tab4:
    st.subheader("Collaboration Network")
    st.markdown("Visualize who works with whom across projects.")
    if len(projects) > 1 and result.overlap_matrix:
        net = network_view(students, result.overlap_matrix)

        edge_traces = []
        by_weight: Dict[int, Dict[str, List[Optional[float]]]] = {}
        for a, b, weight in net.edges:
            seg = by_weight.setdefault(weight, {"x": [], "y": []})
            seg["x"] += [net.positions[a][0], net.positions[b][0], None]
            seg["y"] += [net.positions[a][1], net.positions[b][1], None]
        for weight, seg in sorted(by_weight.items()):
            edge_traces.append(go.Scatter(
                x=seg["x"], y=seg["y"], mode="lines",
                line=dict(width=0.5 + 0.75 * (weight - 1), color="#C7C8CA" if weight == 1 else "#BD0226"),
                hoverinfo="none", name=f"together {weight}×",
            ))
        nodes = list(net.positions)
        node_trace = go.Scatter(
            x=[net.positions[n][0] for n in nodes],
            y=[net.positions[n][1] for n in nodes],
            mode="markers",
            hoverinfo="text",
            text=[f"{n}: {net.degrees[n]} collaborators" for n in nodes],
            marker=dict(
                size=10,
                color=[net.degrees[n] for n in nodes],
                colorscale=[[0, "#D9E7F2"], [1, "#171C54"]],
                colorbar=dict(title="Collaborators"),
                line=dict(width=1, color="white"),
            ),
            showlegend=False,
        )
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Student Collaboration Network",
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        )
        st.plotly_chart(fig, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes (Students)", net.n_nodes)
        col2.metric("Edges (Collaborations)", net.n_edges)
        col3.metric("Network Density", f"{net.density:.3f}")

        st.subheader("Most Connected Students")
        st.dataframe(pd.DataFrame(net.top_connected, columns=["Student", "Unique Collaborators"]),
                     width="stretch", hide_index=True)
    else:
        st.info("Network analysis requires multiple projects to show meaningful collaboration patterns.")

with tab5:
    st.subheader("Algorithm Performance Analysis")
    stats_all: Dict[str, Any] = result.algorithm_stats or {}
    per_project: Dict[str, Dict[str, Any]] = stats_all.get("per_project") or {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Execution Statistics")
        st.write(f"- **Algorithm:** {result_config.algorithm}")
        st.write(f"- **Execution Time:** {result.execution_time:.3f}s")
        final_score_note = (
            f" *(aggregated over all projects incl. {len(imported_projects)} imported (locked))*"
            if imported_projects else ""
        )
        if reports:
            st.write(f"- **Objective (rule-based):** "
                     f"{summarize_reports(reports)['objective']:.2f} (lower is better)")
        st.write(f"- **Classic diversity score:** {result.total_score:.2f}{final_score_note}")
        scalar_stats = {
            k: v for k, v in stats_all.items()
            if k != "per_project" and not isinstance(v, (list, dict))
        }
        if scalar_stats and not per_project:
            st.markdown("### Algorithm-Specific Stats")
            for key, value in scalar_stats.items():
                st.write(f"- **{key.replace('_', ' ').title()}:** {value}")

    with col2:
        st.markdown("### Configuration Used")
        st.caption("Settings frozen when the teams were formed.")
        st.write(f"- **Team Size:** {result_config.team_size_min}-{result_config.team_size_max} "
                 f"(target: {result_config.team_size_target})")
        if int(result_config.num_teams or 0) > 0:
            st.write(f"- **Teams per Project:** {result_config.num_teams}")
        st.write(f"- **Projects:** {result_config.num_projects}")
        st.write(f"- **Students:** {len(students)}")
        st.write(f"- **Repeat cap (rules used for these scores):** "
                 f"{cap_label(rules_of_active_reports())}")
        st.caption(f"The cap of {rules_source_text()}.")
        echo = parameter_echo(result_config)
        if echo:
            st.write(echo)

    if per_project:
        st.markdown("---")
        st.subheader("Per-Project Runs")
        st.dataframe(pd.DataFrame([
            {
                "Project": f"Project {pid}",
                "Optimizer Score": stats.get("score"),
                "Iterations": stats.get("iterations"),
                "Time (s)": stats.get("execution_time"),
                "Converged": "yes" if stats.get("converged") else "no",
            }
            for pid, stats in per_project.items()
        ]), width="stretch", hide_index=True)
        history_rows = [
            {"Project": f"Project {pid}", "Step": step, "Score": value}
            for pid, stats in per_project.items()
            for step, value in enumerate(stats.get("history") or [])
            if value is not None and np.isfinite(value)
        ]
        if history_rows:
            st.subheader("Score Convergence")
            fig = px.line(pd.DataFrame(history_rows), x="Step", y="Score", color="Project",
                          markers=True, title="Optimizer Score per Recorded Step")
            st.plotly_chart(fig, width="stretch")
            st.caption("Optimizer scores of the classic algorithms (higher is better) — a "
                       "different scale from the classic diversity score and from the "
                       "rule-based penalties. Greedy records a single final value.")
    elif stats_all.get("history"):
        st.markdown("---")
        st.subheader("Score Convergence")
        history = [h for h in stats_all.get("history") or [] if h is not None and np.isfinite(h)]
        if history:
            fig = px.line(x=list(range(len(history))), y=history, title="Score Over Iterations",
                          labels={"x": "Iteration", "y": "Score"})
            fig.update_traces(line_color="#171C54")
            st.plotly_chart(fig, width="stretch")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Team Formation", width="stretch"):
        switch_to("formation")
with col2:
    if st.button("🔄 Cross-Project →", width="stretch"):
        switch_to("cross_project")
with col3:
    if st.button("📤 Export →", width="stretch"):
        switch_to("export")
