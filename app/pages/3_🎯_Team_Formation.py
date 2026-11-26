import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import extensions  # noqa: E402
from app.demo import classic_unavailable_where  # noqa: E402
from app.core import classic  # noqa: E402
from app.core import editor_ops as eo  # noqa: E402
from app.core import engine_session as es  # noqa: E402
from app.core import job_runner  # noqa: E402
from app.core import run_activation as ra  # noqa: E402
from app.core import run_history as rh  # noqa: E402
from app.core.assignment_import import validate_against_roster  # noqa: E402
from app.core.config_session import offered_algorithms  # noqa: E402
from app.ui_kit import refuse_large_table  # noqa: E402
from app.core.data_session import (  # noqa: E402
    PARTICIPANTS_SLOT,
    PRIOR_ASSIGNMENTS_SLOT,
    active_rule_count,
    chip,
    clear_import_source,
    departed_warning,
    engine_setup,
    ensure_session_state,
    feasibility_report,
    flash,
    get_data_loader,
    get_import_source,
    is_new_upload,
    mark_upload_processed,
    page_link,
    refresh_imports,
    render_flashes,
    sample_download_button,
    set_import_source,
    switch_to,
)
from app.core.diversity_metrics import DiversityMetrics  # noqa: E402
from app.core.job_runner import CancelFn, ProgressFn  # noqa: E402
from app.core.models import ALGORITHMS, Configuration, FormationResult, Project  # noqa: E402
from app.engine import solver as engine_solver  # noqa: E402
from app.engine.adapters import teams_from_legacy  # noqa: E402
from app.engine.policy import Policy  # noqa: E402
from app.engine.report_views import (  # noqa: E402
    history_section,
    overall,
    rule_entries,
    stats_view,
    teams,
)
from app.engine.runner import JOINT_ROUNDS, ProjectSpec  # noqa: E402
from app.pages._views import (  # noqa: E402
    classic_score_bar,
    classic_team_score,
    render_headline_row,
    render_roster_team_view,
    render_scored_rules_caption,
)
from app.utils.fmt import fmt_pct  # noqa: E402

st.set_page_config(
    page_title="Team Formation | Group Diversifier",
    page_icon="🎯",
    layout="wide",
)

config = ensure_session_state()

ALGORITHMS = list(ALGORITHMS)
ALGORITHM_LABELS = {
    "engine": "🧭 Rules engine (recommended)",
    "greedy": "🚀 Greedy",
    "simulated_annealing": "🔥 Simulated Annealing",
    "genetic": "🧬 Genetic Algorithm",
}
SOURCE_TEXT = {"edited": "edited", "evaluated": "re-scored or evaluated",
               "imported": "imported"}
_SUMMARY_KEY = "_formation_summary"
_SUMMARY_FRESH_KEY = "_formation_summary_fresh"
_ALTERNATIVES_KEY = "_formation_alternatives"
_BACKGROUND_KEY = "_formation_background"
_STOP_REQUEST_KEY = "_formation_stop_requested"
_STOP_BUTTON_KEY = "formation_stop"


def algorithm_label(name: str) -> str:
    return ALGORITHM_LABELS.get(name, name.replace("_", " ").title())


def plain_label(name: str) -> str:
    return algorithm_label(name).split(" ", 1)[-1]


def run_wide(reports: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    summary = es.summarize_reports(reports) if reports else {}
    return int(summary.get("hard_units") or 0), int(summary.get("repeated_pairs") or 0)


def change_text(after: int, before: int) -> str:
    return f"{after} ({after - before:+d})" if after != before else f"{after} (±0)"


def size_issues(result_config: Configuration, projects: List[Project]) -> List[str]:
    issues = []
    for project in projects:
        if project.imported:
            continue
        try:
            p_idx = int(project.project_id) - 1
        except (ValueError, TypeError):
            p_idx = 0
        min_size, max_size, _target = result_config.get_team_sizes_for_project(p_idx)
        for team in project.teams:
            if team.size < min_size:
                issues.append(f"{team.team_id} in {project.name}: {team.size} < {min_size}")
            if team.size > max_size:
                issues.append(f"{team.team_id} in {project.name}: {team.size} > {max_size}")
    return issues


def coverage_issues(projects: List[Project], roster_ids: Set[str], participants) -> List[str]:
    issues = []
    for project in projects:
        if project.imported:
            continue
        pool_ids = participants.pool_for(project.project_id) if participants else None
        pool = set(pool_ids) if pool_ids is not None else set(roster_ids)
        missing = pool - project.all_student_ids
        if missing:
            issues.append(f"{project.name}: not assigned: {sorted(missing)}")
    return issues


def build_recommendations(
    result: FormationResult,
    team_scores: List[float],
    sizes: List[str],
    coverage: List[str],
    summary: Optional[Dict[str, Any]],
    engine_summary: Dict[str, Any],
    policy: Policy,
) -> List[Tuple[str, str]]:
    advice: List[Tuple[str, str]] = []
    n_violations = len(result.constraint_violations)
    over_cap = int(engine_summary.get("over_cap_pairs") or 0)
    if n_violations:
        advice.append(("error",
            f"**{n_violations} team(s) break a classic hard constraint** (see the ⚠️ Violations "
            "tab). Relax the offending constraint on ⚙️ Configuration (e.g. widen team sizes) "
            "or keep local search on and try another algorithm/seed."))
    if over_cap:
        advice.append(("warning",
            f"**{over_cap} pair(s) share more teams than the repeat cap of the allocation rules "
            f"({es.cap_label(policy)}) allows.** More teams per project (smaller teams) leave more "
            "fresh partners; the rules engine enforces the cap directly."))
    if sizes:
        advice.append(("warning", f"**{len(sizes)} team(s) are outside the size limits.**"))
    if coverage:
        advice.append(("error", "**Some participants were not assigned** — see the Validation Summary."))
    if team_scores:
        mean = float(np.mean(team_scores))
        worst = float(min(team_scores))
        spread = float(np.std(team_scores))
        if spread > 0 and worst < mean - 2 * spread:
            advice.append(("info",
                f"The weakest team has a classic diversity score of {worst:.3f} against an "
                f"average of {mean:.3f}. Inspect it in the Team View; increasing the weight of "
                "the attribute it lacks usually helps."))
    if summary and summary.get("mp_failed"):
        advice.append(("warning",
            "Multiprocessing failed and the run fell back to one core. Set *CPU Cores* to 1 "
            "in Configuration to skip the process start-up cost."))
    if not advice:
        advice.append(("success",
            "No hard-constraint violations, no pair over the repeat cap and all teams within "
            "their size limits. Review team composition in the Team View and Analytics."))
    return advice


def engine_recommendations(summary: Dict[str, Any], meta: Dict[str, Any],
                           coverage: List[str]) -> List[Tuple[str, str]]:
    advice: List[Tuple[str, str]] = []
    hard = int(summary.get("hard_units") or 0)
    if hard:
        advice.append(("error",
            f"**{hard} unit(s) of hard-rule violation remain.** The 📋 Rule compliance tab shows "
            "which rules; the feasibility check on 🧭 Allocation Rules tells you whether they "
            "are avoidable. Loosen the rule, make it soft, or give the solver more time."))
    if int(summary.get("over_cap_pairs") or 0):
        advice.append(("warning",
            f"**{summary['over_cap_pairs']} pair(s) share more teams than the repeat cap allows.** "
            "More, smaller teams leave more fresh partners."))
    if any("time limit" in str(w) for w in meta.get("warnings", [])):
        advice.append(("info",
            "The search stopped at its time limit: a longer limit or more restarts "
            "(Allocation Rules → Solver) may improve the result."))
    quality = summary.get("quality_index_soft")
    if quality is not None and float(quality) < 50:
        advice.append(("info",
            f"Quality vs random is {float(quality):.0f}%: many competing rules pull in different "
            "directions. Lower the weights of the least important ones."))
    if coverage:
        advice.append(("error", "**Some participants were not assigned** — see the Validation Summary."))
    if not advice:
        advice.append(("success",
            "Every hard rule holds and no pair exceeds the repeat cap. Review the teams below "
            "or fine-tune them in the Team Editor."))
    return advice


def _quick_sync(widget_key: str, attr: str) -> None:
    cfg = st.session_state.config
    value = st.session_state[widget_key]
    setattr(cfg, attr, value)
    if attr == "num_projects" and len(cfg.per_project_team_sizes) > int(value):
        cfg.per_project_team_sizes = cfg.per_project_team_sizes[: int(value)]


def project_time_cap(settings: Any, n_students: int) -> float:
    return float(engine_solver.effective_time_limit(settings, n_students))


def engine_estimate(policy: Policy, specs: List[Any]) -> str:
    s = es.capped_policy(policy).solver
    total = sum(project_time_cap(s, len(spec.student_ids)) for spec in specs) or float(
        s.time_limit_s)
    total *= (1 + JOINT_ROUNDS) if s.joint_projects else 1
    if s.iterations is not None and s.method == "anneal":
        return f"≤ {total:.0f}s, usually much less"
    return f"≤ {total:.0f}s"


def run_classic(config: Configuration, students: List[Any], *,
                imported_projects: Any = None, participants: Any = None) -> "classic.ClassicRun":
    with st.spinner("Forming teams..."):
        progress_container = st.container()
        with progress_container:
            pcol1, pcol2, pcol3 = st.columns([3, 1, 1])
            with pcol1:
                overall_progress_bar = st.progress(0.0)
            with pcol2:
                overall_status = st.empty()
            with pcol3:
                elapsed_timer = st.empty()
            project_status = st.empty()
            mp_indicator = st.empty()
            algorithm_progress = st.progress(0.0)
            iteration_status = st.empty()
            time_estimate = st.empty()
            completed_projects_log = st.empty()

        completed_log_lines: List[str] = []

        def render(event: "classic.ClassicProgress") -> None:
            if event.stage == "start":
                if event.num_cores > 1:
                    mp_indicator.info(f"**{event.algo_label}** | "
                                      f"up to {event.num_cores} worker processes")
                else:
                    mp_indicator.info(f"💻 **{event.algo_label}** | Single-core mode")
                time_estimate.info(f"⏱️ **Estimated total time:** {event.estimate} "
                                   f"({event.n_students} students × {event.n_generated} projects)")
            elif event.stage == "project_start":
                overall_progress_bar.progress(event.project_idx / event.num_projects)
                overall_status.markdown(f"**{event.project_idx}/{event.num_projects}**")
                project_status.markdown(
                    f"### 🔄 **Project {event.project_idx + 1} of {event.num_projects}**")
                if event.remaining is not None:
                    time_estimate.info(f"⏱️ **Estimated time remaining:** "
                                       f"~{event.remaining:.0f} seconds")
                else:
                    time_estimate.info(
                        "⏱️ **First project running...** (time estimate after completion)")
            elif event.stage == "tick":
                elapsed_timer.markdown(f"**⏱️ {event.elapsed:.1f}s**")
                if event.updated and event.total > 0:
                    it, tot, sc = event.iteration, event.total, event.score
                    algorithm_progress.progress(min(it / tot, 0.99))
                    iteration_status.markdown(
                        f"⚙️ **{it:,}/{tot:,}** ({100 * it / tot:.0f}%) | "
                        f"Score: **{sc:.2f}** | ⏱️ {event.elapsed_project:.1f}s"
                    )
                else:
                    iteration_status.markdown(f"⏳ Working… ⏱️ {event.elapsed_project:.1f}s")
                    if event.last_project_time:
                        algorithm_progress.progress(
                            min(event.elapsed_project / event.last_project_time, 0.99))
            elif event.stage == "project_done":
                algorithm_progress.progress(1.0)
                iteration_status.markdown(
                    f"✅ **Project {event.project_idx + 1} complete** | "
                    f"Score: **{event.score:.2f}** | Iterations: {event.iterations:,} | "
                    f"⏱️ {event.elapsed_project:.1f}s"
                )
                completed_log_lines.append(
                    f"✅ Project {event.project_idx + 1}: Score **{event.score:.2f}** | "
                    f"{event.iterations:,} iterations | {event.elapsed_project:.1f}s"
                )
                completed_projects_log.markdown("\n\n".join(completed_log_lines))
            elif event.stage == "done":
                overall_progress_bar.progress(1.0)
                overall_status.markdown(f"**{event.num_projects}/{event.num_projects}** ✅")
            elif event.stage == "waiting":
                iteration_status.markdown(f"⏳ {event.message}")

        try:
            return classic.run(config, students, imported_projects=imported_projects,
                               participants=participants, progress=render)
        except classic.ClassicRunError as failure:
            if failure.project_idx < 0:
                st.error(f"❌ {failure.cause}")
            else:
                st.error(f"❌ Algorithm failed on project {failure.project_idx + 1}: "
                         f"{failure.cause}")
            st.stop()



def finish_engine_run(engine_result: Any, inputs: Dict[str, Any], *,
                      stopped: bool = False) -> None:
    record, summary = ra.finish_engine_run(engine_result, **inputs, stopped=stopped)
    st.session_state.pop(_ALTERNATIVES_KEY, None)
    st.session_state[_SUMMARY_KEY] = summary
    st.session_state[_SUMMARY_FRESH_KEY] = True
    st.rerun()


def _stop_wait_seconds(inputs: Dict[str, Any]) -> float:
    return 2 * float(inputs["policy"].solver.time_limit_s) + 30


def collect_stopped_run() -> None:
    job_runner.collect_stopped_run(_finish_stopped_run, job_key=_BACKGROUND_KEY,
                                   stop_key=_STOP_REQUEST_KEY,
                                   wait_seconds=_stop_wait_seconds)


def _finish_stopped_run(result: Any, inputs: Dict[str, Any]) -> None:
    finish_engine_run(result, inputs, stopped=True)


def engine_reports_for(projects: List[Project]) -> Dict[str, Dict[str, Any]]:
    reports = es.active_reports()
    generated = {str(p.project_id) for p in projects if not p.imported}
    if reports and generated and generated <= set(reports):
        return reports
    return {}


def active_meta(record: Optional[rh.RunRecord]) -> Dict[str, Any]:
    if record is None:
        return {}
    stored = st.session_state.get(es.META_KEY) or {}
    if stored.get("run_id") == record.run_id:
        return stored
    summary = record.summary or {}
    meta = {"source": record.source, "method": summary.get("method") or record.source,
            "policy_name": str((record.policy or {}).get("name") or ""), "warnings": [],
            "run_id": record.run_id}
    for key in ("seed", "elapsed_s"):
        if key in summary:
            meta[key] = summary[key]
    return meta


def classic_result(result: FormationResult) -> bool:
    return "engine_summary" not in (result.algorithm_stats or {})


st.title("🎯 Team Formation")
st.markdown("Form the teams with the rules engine (or a classic algorithm) and review the result.")
render_flashes()

with st.expander("⚡ Algorithm quick reference", expanded=False):
    st.markdown(f"""
    **Current algorithm:** {algorithm_label(config.algorithm)}

    | Algorithm | How it works |
    |-----------|--------------|
    | 🧭 **Rules engine** | Optimises the rules on the 🧭 Allocation Rules page (attribute goals, head-count quotas, no-lone rules, repeat-pairing caps, preferences) with annealing and/or an exact MILP model, and reports compliance per rule. |
    | 🚀 **Greedy** | Classic: builds teams one at a time from the best candidates, then improves them with pairwise swaps. Fastest. |
    | 🔥 **Simulated Annealing** | Classic: random start, then swap/move proposals that occasionally accept worse states to escape local optima. |
    | 🧬 **Genetic** | Classic: evolves a population of assignments with crossover and mutation, repairing team sizes. Slowest. |

    Classic algorithms use the weights and constraints on the ⚙️ Configuration page; their
    runs are still scored with the allocation rules so you can compare every run on the
    same yardstick.
    """)

with st.expander("ℹ️ How are results scored? (Click to learn)", expanded=False):
    st.markdown("""
    **Rule-based scores (all runs).** Every team gets penalty points from the allocation
    rules (imbalance, broken quotas, isolated students, repeat pairings, …); hard rules
    cost 1000 points per unit. *Quality vs random* compares the soft penalty with random
    teams of the same sizes: 100% = no penalty at all, 0% = no better than random.

    **Repeat pairs** are teammates who already shared a *team* before; groups larger
    than the section limit of the allocation rules ({section_example}) do not count
    toward the repeat cap. **Section-mates reunited** counts teammates whose only earlier
    contact was such a section: a weak soft contact, not a repeat.

    **Classic diversity score (legacy).** A display index kept from the classic
    algorithms: for each team the weighted average of entropy-based per-attribute scores
    (0-1). The rules engine does not optimise it, so it is only shown in the clearly
    labelled classic sections, never to rank runs.
    """.replace("{section_example}", extensions.text("team_formation.section_example", "e.g. large shared groups")))

loader = get_data_loader()
students = loader.get_students()

if not students:
    st.warning("No students loaded. Please load data first.")
    if st.button("Go to Data Overview"):
        switch_to("data")
    st.stop()

roster_ids = {s.student_id for s in students}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Students", len(students))
if int(config.num_teams or 0) > 0:
    col2.metric("Teams per Project", int(config.num_teams))
else:
    col2.metric("Target Team Size", config.team_size_target)
col3.metric("Projects", config.num_projects)
col4.metric("Algorithm", plain_label(config.algorithm))

with st.expander("📥 Import prior projects (locked history)", expanded=False):
    st.markdown(
        "Upload prior team assignments to continue from them: the imported projects "
        "become locked Projects 1..K and formation only generates the remaining ones, "
        "avoiding repeat pairings from the imported history. Imports are kept while "
        "you visit other pages, until you clear them."
    )
    sample_download_button("earlier_groups", label="⬇️ Download sample earlier groups")
    up_file = st.file_uploader(
        "Assignments CSV — columns: team_id, project_id, student_id",
        type="csv",
        key="prior_assignments",
        help="One row per student and project, all earlier projects in one file, numbered "
             "1..K. Also accepted: group_id or group for team_id, identifier or student "
             "for student_id, project for project_id.",
    )
    if (up_file is not None and not refuse_large_table(up_file)
            and is_new_upload(PRIOR_ASSIGNMENTS_SLOT, up_file)):
        set_import_source(
            PRIOR_ASSIGNMENTS_SLOT, up_file.name,
            up_file.getvalue().decode("utf-8-sig", errors="replace"),
        )
        mark_upload_processed(PRIOR_ASSIGNMENTS_SLOT, up_file)

    st.markdown("---")
    st.markdown(
        "**Optional — participant lists per generated project.** Upload a CSV with "
        "columns `project_id,student_id` to restrict who takes part in each new "
        "project. A new project missing from the file uses the full roster. Student "
        "ids are matched leading-zero-insensitively."
    )
    sample_download_button("participants", label="⬇️ Download sample participant lists")
    up_parts = st.file_uploader(
        "Participants CSV — columns: project_id, student_id",
        type="csv",
        key="participants_file",
        help="One row per participant. Also accepted: identifier or student for "
             "student_id, project for project_id.",
    )
    if (up_parts is not None and not refuse_large_table(up_parts)
            and is_new_upload(PARTICIPANTS_SLOT, up_parts)):
        set_import_source(
            PARTICIPANTS_SLOT, up_parts.name,
            up_parts.getvalue().decode("utf-8-sig", errors="replace"),
        )
        mark_upload_processed(PARTICIPANTS_SLOT, up_parts)

    import_errors = refresh_imports(roster_ids, int(config.num_projects))
    st.markdown("---")

    for import_error in import_errors.get(PRIOR_ASSIGNMENTS_SLOT, []):
        st.error(import_error)
    imported_assignments = st.session_state.get("imported_assignments")
    if imported_assignments:
        source = get_import_source(PRIOR_ASSIGNMENTS_SLOT)
        k_imported = imported_assignments.project_count
        n_assignments = sum(
            len(member_ids)
            for teams in imported_assignments.teams_by_project.values()
            for member_ids in teams.values()
        )
        st.success(
            f"🔒 Projects 1–{k_imported} imported and locked ({n_assignments} assignments"
            + (f" from {source[0]}" if source else "")
            + f"). Formation will generate projects {k_imported + 1} and up."
        )
        preview_rows = [
            {
                "Project": pid,
                "Teams": len(teams),
                "Students": sum(len(member_ids) for member_ids in teams.values()),
            }
            for pid, teams in imported_assignments.teams_by_project.items()
        ]
        st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
        for import_warning in imported_assignments.warnings:
            st.warning(import_warning)
    if imported_assignments or get_import_source(PRIOR_ASSIGNMENTS_SLOT):
        if st.button("🗑️ Clear imported history", key="clear_imported_history"):
            clear_import_source(PRIOR_ASSIGNMENTS_SLOT)
            st.session_state.pop("prior_assignments", None)
            st.rerun()

    for import_error in import_errors.get(PARTICIPANTS_SLOT, []):
        st.error(import_error)
    participants_spec = st.session_state.get("project_participants")
    if participants_spec and participants_spec.participants:
        participant_rows = [
            {"Project": pid, "Participants": len(members)}
            for pid, members in sorted(
                participants_spec.participants.items(), key=lambda kv: int(kv[0])
            )
        ]
        st.dataframe(pd.DataFrame(participant_rows), width="stretch", hide_index=True)
        restricted = ", ".join(sorted(participants_spec.participants, key=int))
        st.caption(f"Restricted pools: projects {restricted}. Projects not listed use the full roster.")
    if participants_spec:
        for participant_warning in participants_spec.warnings:
            st.warning(participant_warning)
    if participants_spec or get_import_source(PARTICIPANTS_SLOT):
        if st.button("🗑️ Clear participant lists", key="clear_participants"):
            clear_import_source(PARTICIPANTS_SLOT)
            st.session_state.pop("participants_file", None)
            st.rerun()

imported_assignments = st.session_state.get("imported_assignments")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Run Team Formation")
    with st.expander("⚡ Quick Settings", expanded=False):
        st.caption("Changes here update the shared configuration (same as the ⚙️ Configuration page).")
        quick_col1, quick_col2 = st.columns(2)
        with quick_col1:
            size_min, size_max = int(config.team_size_min), int(config.team_size_max)
            if int(config.num_teams or 0) > 0:
                st.caption(f"Team sizes follow the **{int(config.num_teams)} teams** per project "
                           "set in Configuration.")
            elif config.per_project_team_sizes:
                st.caption("Team sizes are set per project in Configuration.")
            elif size_min < size_max:
                st.session_state["quick_team_size"] = min(max(int(config.team_size_target), size_min), size_max)
                st.slider(
                    "Team Size",
                    min_value=size_min,
                    max_value=size_max,
                    key="quick_team_size",
                    on_change=_quick_sync,
                    args=("quick_team_size", "team_size_target"),
                    help="Target team size within the configured min/max",
                )
            else:
                st.caption(f"Team size is fixed at **{size_min}** (min = max).")
        with quick_col2:
            st.session_state["quick_projects"] = min(max(int(config.num_projects), 1), 10)
            st.slider(
                "Projects",
                min_value=1,
                max_value=10,
                key="quick_projects",
                on_change=_quick_sync,
                args=("quick_projects", "num_projects"),
            )
        offered = offered_algorithms()
        if len(offered) > 1:
            st.session_state["quick_algorithm"] = (
                config.algorithm if config.algorithm in offered else offered[0]
            )
            st.selectbox(
                "Algorithm",
                options=offered,
                format_func=algorithm_label,
                key="quick_algorithm",
                on_change=_quick_sync,
                args=("quick_algorithm", "algorithm"),
            )
        else:
            st.caption(f"Algorithm: **{algorithm_label(offered[0])}** (the classic "
                       f"algorithms are not available on {classic_unavailable_where()}).")

imported_count_now = imported_assignments.project_count if imported_assignments else 0
n_generated_now = max(0, int(config.num_projects) - imported_count_now)
use_engine = config.algorithm == "engine"

setup = engine_setup(config, refresh=False)
rules_problems: List[str] = []
feasibility_unconfirmed = False
if setup is not None:
    rules_problems = setup.policy.validate(es.attribute_kinds(setup.frame))
with col1:
    if setup is not None:
        policy_now = setup.policy
        if use_engine:
            seed_text = "random seed" if policy_now.solver.seed is None else f"seed {policy_now.solver.seed}"
            st.markdown(f"**Rules:** {policy_now.name} · {active_rule_count(policy_now)} active "
                        f"rules · {policy_now.solver.method}, {seed_text}, "
                        f"{policy_now.solver.time_limit_s:g} s per project")
        else:
            st.caption(f"Classic run: the result is also scored with the rules "
                       f"**{policy_now.name}** so it compares with engine runs.")
        page_link("rules", "Edit the allocation rules")
        if rules_problems:
            st.error("**The rules need fixing before the engine can run:**\n"
                     + "\n".join(f"- {p}" for p in rules_problems))
        elif use_engine and setup.specs:
            feas = feasibility_report(setup)
            if feas.errors:
                st.error(f"**Feasibility check: {feas.summary()}** — not every rule can hold. "
                         "The run is blocked until you confirm: the engine would only minimise "
                         "the violations.\n"
                         + "\n".join(f"- {(f.rule_label + ': ') if f.rule_label else ''}{f.message}"
                                     for f in feas.errors[:5]))
                feasibility_unconfirmed = not st.checkbox(
                    "Run anyway, despite the feasibility errors", key="run_anyway_feasibility",
                    help="Same as `--force` on the command line.")
            elif feas.warnings:
                with st.expander(f"⚠️ Feasibility check: {feas.summary()}", expanded=False):
                    for finding in feas.warnings:
                        st.warning((f"**{finding.rule_label}** — " if finding.rule_label else "")
                                   + finding.message)
            else:
                st.success(f"✅ Feasibility check: {feas.summary().lower()}.")

with col2:
    st.subheader("Actions")
    if use_engine and setup is not None:
        est = engine_estimate(setup.policy, setup.specs)
        st.caption(f"⏱️ {est} for {n_generated_now} generated project(s).")
    else:
        est = classic.estimate(config, len(students), max(1, n_generated_now))
        st.caption(f"⏱️ Rough estimate: {est} for {n_generated_now} generated project(s) "
                   f"on {int(config.num_cores)} core(s).")
    run_button = st.button(
        f"🚀 Run Team Formation\n({est})",
        type="primary",
        width="stretch",
        help=f"Estimated time: {est}",
        key="run_team_formation",
    )
    run_blocked_reason = None
    if imported_assignments and config.num_projects <= imported_assignments.project_count:
        run_blocked_reason = (
            f"Projects is set to {config.num_projects} but "
            f"{imported_assignments.project_count} projects are imported and locked — "
            f"set Projects to at least {imported_assignments.project_count + 1} "
            f"(Quick Settings above or Configuration)."
        )
    elif use_engine and rules_problems:
        run_blocked_reason = "Fix the allocation rules first (see the messages on the left)."
    elif use_engine and feasibility_unconfirmed:
        run_blocked_reason = ("The feasibility check found errors (see the left). Fix the rules "
                              "or sizes, or tick “Run anyway” to run regardless.")
    if run_button and run_blocked_reason:
        st.error(run_blocked_reason)

    if es.has_active_allocation():
        if st.button("🗑️ Clear Results", width="stretch"):
            es.clear_active_allocation()
            st.session_state.pop(_SUMMARY_KEY, None)
            st.session_state.pop(_ALTERNATIVES_KEY, None)
            st.rerun()

collect_stopped_run()

if run_button and not run_blocked_reason:
    run_imp = st.session_state.get("imported_assignments")
    if run_imp is not None:
        run_imp = copy.deepcopy(run_imp)
        validate_against_roster(run_imp, roster_ids)
    imported_count = run_imp.project_count if run_imp else 0
    n_generated_projects = config.num_projects - imported_count
    run_config = copy.deepcopy(config)
    run_participants = st.session_state.get("project_participants")
    raw_df = loader.get_dataframe()
    frame = es.roster_frame(students, raw_df)
    run_policy = es.current_policy(run_config, frame)

    if run_config.algorithm == "engine":
        engine_inputs = {"config": run_config, "policy": run_policy, "imported": run_imp,
                         "students": list(students), "n_generated": n_generated_projects,
                         "raw_df": raw_df}

        def engine_work(progress: ProgressFn, cancel: CancelFn):
            return es.run_engine(run_config, students, raw_df=raw_df,
                                 imported_assignments=run_imp, participants=run_participants,
                                 policy=run_policy, progress=progress, cancel=cancel)

        st.info(f"🧭 **Rules engine** · {run_policy.name} · {run_policy.solver.method} · "
                f"{n_generated_projects} project(s)")
        try:
            engine_result, _prior = job_runner.run_in_background(
                engine_work, title="Forming teams", stop_job=engine_inputs,
                job_key=_BACKGROUND_KEY, stop_key=_STOP_REQUEST_KEY,
                stop_button_key=_STOP_BUTTON_KEY)
        except Exception as exc:
            st.error(f"❌ The rules engine could not form teams: {exc}")
            page_link("rules", "Check the allocation rules")
            st.stop()
        finish_engine_run(engine_result, engine_inputs)

    else:
        algo_name = plain_label(run_config.algorithm)
        classic_run = run_classic(run_config, students, imported_projects=run_imp,
                                  participants=run_participants)
        classic.activate(run_config, students, classic_run)
        projects = classic_run.projects
        total_time = float(classic_run.result.execution_time)
        per_project_stats = classic_run.stats["per_project"]

        evaluation_warnings: List[str] = []
        try:
            reports = es.evaluate_projects(projects, run_config, students, policy=run_policy,
                                           raw_df=raw_df)
        except Exception as exc:
            reports = {}
            evaluation_warnings.append(f"Could not score the run with the allocation rules: {exc}")
        es.set_active_reports(reports, run_policy if reports else None)
        record = ra.record_run(
            label=f"{algo_name} (classic)", source="legacy", run_config=run_config,
            policy=run_policy, projects=projects, reports=reports,
            extra_summary={"method": run_config.algorithm, "elapsed_s": round(total_time, 3),
                           "seed": run_config.random_seed, "algorithm": run_config.algorithm},
            roster=ra.roster_identity(students, raw_df),
        )
        es.set_result_meta({
            "method": run_config.algorithm, "elapsed_s": total_time,
            "warnings": evaluation_warnings, "seed": run_config.random_seed,
            "policy_name": run_policy.name, "source": "legacy", "run_id": record.run_id,
        })
        st.session_state.pop(_ALTERNATIVES_KEY, None)

        mp_requested = run_config.num_cores > 1
        mp_failed = mp_requested and any(
            not s.get("multiprocessing_successful") and int(s.get("requested_cores") or 1) > 1
            for s in per_project_stats.values()
        )
        st.session_state[_SUMMARY_KEY] = {
            "total_time": total_time,
            "teams_formed": sum(len(p.teams) for p in projects if not p.imported),
            "n_projects": len(projects),
            "n_generated": n_generated_projects,
            "algorithm": run_config.algorithm,
            "num_cores": run_config.num_cores,
            "mp_failed": mp_failed,
            "per_project": per_project_stats,
            "run_label": rh.run_title(record),
            "run_id": record.run_id,
        }
        st.session_state[_SUMMARY_FRESH_KEY] = True
        st.rerun()


def render_engine_headline(reports: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    summary = es.summarize_reports(reports)
    render_headline_row(summary, reports, preferences=True,
                        preferences_help=f"Avoid requests violated: "
                                         f"{int(summary.get('avoid_violated') or 0)}")
    if int(summary.get("repeated_pairs") or 0) > 0:
        page_link("cross_project", "See why on Cross-Project › Repeat Pairs", icon="🔁",
                  help="A reason and a suggested fix for every repeat pair: requested together, "
                       "placed by hand, the search ended early, or unavoidable here.")
    seed = meta.get("seed")
    seed_text = "random seed" if seed is None else f"seed {seed}"
    source = meta.get("source")
    if source == "legacy":
        st.caption(f"Scored with the allocation rules **{meta.get('policy_name', '')}** "
                   f"(classic {plain_label(str(meta.get('method', '')))} run, {seed_text}).")
    elif source == "engine":
        parts = [f"Rules **{meta.get('policy_name', '')}**", f"method **{meta.get('method')}**"]
        if "seed" in meta:
            parts.append(seed_text)
        if meta.get("elapsed_s") is not None:
            parts.append(f"{float(meta['elapsed_s']):.1f} s")
        st.caption(" · ".join(parts))
    elif meta:
        st.caption(f"Scored with the allocation rules **{meta.get('policy_name', '')}** "
                   f"({SOURCE_TEXT.get(str(source), str(source))} allocation).")
    return summary


def render_rule_compliance(reports: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for pid, report in reports.items():
        for rule in rule_entries(report):
            if rule.kind == "attribute":
                rows.append({
                    "Project": pid, "Rule": rule.label,
                    "Type": f"attribute · {rule.goal}", "Hard": "soft",
                    "Compliance %": float(rule.quality_pct or 0.0),
                    "Teams OK": "", "Points": float(rule.points or 0.0),
                })
            else:
                rows.append({
                    "Project": pid, "Rule": rule.label, "Type": rule.kind,
                    "Hard": "hard" if rule.hard else "soft",
                    "Compliance %": float(rule.compliance_pct or 0.0),
                    "Teams OK": f"{rule.teams_compliant}/{rule.teams_total}",
                    "Points": float(rule.points or 0.0),
                })
        history = history_section(report)
        if history:
            reunited = es.section_reunions({}, {pid: report})
            rows.append({
                "Project": pid, "Rule": "Repeat pairings", "Type": "history", "Hard": "",
                "Compliance %": None, "Teams OK": "",
                "Points": None,
                "Detail": f"{history.repeated_pairs or 0} repeated, "
                          f"{history.over_cap_pairs or 0} over the cap"
                          + (f", {reunited} section-mates reunited" if reunited else ""),
            })
    if not rows:
        st.info("No rule results.")
        return
    st.dataframe(
        pd.DataFrame(rows), width="stretch", hide_index=True,
        column_config={
            "Compliance %": st.column_config.ProgressColumn(
                "Compliance", min_value=0, max_value=100, format="%.0f%%",
                help="Count/isolation/pair rules: share of teams that comply. Attribute goals: "
                     "quality vs random (100% = perfect)."),
            "Points": st.column_config.NumberColumn("Penalty points", format="%.2f"),
        },
    )


def render_team_cards(projects: List[Project], reports: Dict[str, Dict[str, Any]],
                      frame: pd.DataFrame, policy: Policy) -> None:
    generated = [p for p in projects if str(p.project_id) in reports]
    if not generated:
        st.info("No generated projects to show.")
        return
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    index = c1.selectbox("Project", range(len(generated)), key="engine_cards_project",
                         format_func=lambda i: generated[i].name)
    worst_first = c2.toggle("Worst teams first", value=True, key="engine_cards_worst_first")
    only_issues = c3.toggle("Only teams with violations", value=False, key="engine_cards_issues")
    project = generated[index]
    report = reports[str(project.project_id)]
    attr_options = [c for c in frame.columns if c not in ("student_id", "prefer_teammates",
                                                          "avoid_teammates")]
    preferred = [a for a in policy.referenced_attributes() if a in attr_options]
    shown = st.multiselect("Attributes on the cards", attr_options,
                           default=(preferred or attr_options)[:4], key="engine_cards_attrs")
    by_id = frame.set_index("student_id")
    team_views = list(teams(report))
    if worst_first:
        team_views.sort(key=lambda t: (-int(t.hard_units or 0), -float(t.penalty or 0)))
    if only_issues:
        team_views = [t for t in team_views if t.issues]
    if not team_views:
        st.success("No team has a rule violation.")
        return
    team_ids = [t.team_id for t in project.teams]
    columns = st.columns(3)
    for n, team in enumerate(team_views):
        k = int(team.index or 0)
        name = team_ids[k] if k < len(team_ids) else (
            team.name if team.name is not None else f"Team {k + 1}")
        with columns[n % 3]:
            with st.container(border=True):
                st.markdown(f"**{name}** · {team.size} members · penalty "
                            f"**{float(team.penalty or 0):.2f}**")
                issues = team.issues or []
                if issues:
                    st.markdown(" ".join(
                        chip(i.get("text", ""), "red" if bool(i.get("hard")) else "orange")
                        for i in issues))
                else:
                    st.markdown(":green-badge[all rules met]")
                rows = []
                for sid in team.members or []:
                    row = {"ID": sid}
                    for attr in shown:
                        value = by_id.at[sid, attr] if sid in by_id.index else None
                        row[attr] = "—" if value is None else str(value)
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_alternatives(projects: List[Project], reports: Dict[str, Dict[str, Any]]) -> None:
    from app.engine.alternatives import generate_alternatives

    st.caption("Several good allocations of one project that differ noticeably from each other "
               "— handy when you want to choose by hand. Each alternative is saved as its own run.")
    generated = [p for p in projects if not p.imported and str(p.project_id) in reports]
    if not generated:
        st.caption("Run the rules engine first.")
        return
    st.caption("Alternatives respect the history of **every other project** of the run, later "
               "ones included, so choosing one cannot quietly break the repeat cap in a later "
               "project. The list shows the run-wide totals each alternative leads to.")
    c1, c2, c3 = st.columns([1.2, 1, 2])
    pid = c1.selectbox("Project", [str(p.project_id) for p in generated], key="alt_project",
                       format_func=lambda p: f"Project {p}")
    k_alt = c2.number_input("How many", min_value=2, max_value=6, value=3, step=1, key="alt_k")
    min_diff = c3.slider("Minimum difference", 0.1, 0.9, 0.4, 0.05, key="alt_min_diff",
                         help="Share of teammate pairs that must differ from every earlier "
                              "alternative")
    if st.button("🔀 Generate alternatives", key="alt_run"):
        idx = next(i for i, p in enumerate(projects) if str(p.project_id) == pid)
        project = projects[idx]
        raw_df = loader.get_dataframe()
        frame = es.roster_frame(students, raw_df)
        policy = es.current_policy(config, frame)
        teams = teams_from_legacy(project)
        eff = es.project_config(config, idx)
        spec = ProjectSpec(project_id=pid, student_ids=[sid for t in teams for sid in t],
                           team_sizes=[len(t) for t in teams], size_min=int(eff.team_size_min),
                           size_max=int(eff.team_size_max), size_target=int(eff.team_size_target))
        prior = es.other_projects_history(projects, idx)

        def work(progress: ProgressFn, cancel: CancelFn):
            return generate_alternatives(frame, es.capped_policy(policy), spec,
                                         prior_projects=prior, k=int(k_alt),
                                         min_difference=float(min_diff), progress=progress,
                                         cancel=cancel)

        try:
            alternatives = job_runner.run_in_background(work, title="Generating alternatives")
        except NotImplementedError:
            st.info("Alternative generation is not available in this build. Rerun with a "
                    "different seed (Allocation Rules → Solver) to get a different allocation.")
            return
        except Exception as exc:
            st.error(f"Could not generate alternatives: {exc}")
            return
        alloc = es.active_allocation()
        parent = alloc.run_id if alloc is not None else None
        active = rh.active_run()
        same_rules = (active is not None and bool(active.policy)
                      and eo.scoring_signature(active.policy) == eo.scoring_signature(policy)
                      and eo.fingerprint(active.projects)
                      == eo.fingerprint(es.projects_to_ids(projects)))
        if same_rules:
            base_reports = dict(reports)
        else:
            try:
                base_reports = es.evaluate_projects(projects, config, students, policy=policy,
                                                    raw_df=raw_df)
            except ValueError as exc:
                st.error(f"Could not score the run with the current rules: {exc}")
                return
        rescored = [pid] + [str(p.project_id) for p in projects[idx + 1:] if not p.imported]
        hard_now, repeats_now = run_wide(base_reports)
        saved = []
        for n, alt in enumerate(alternatives or [], start=1):
            plain = es.projects_to_ids(projects)
            for entry in plain:
                if entry["project_id"] == pid:
                    entry["teams"] = [list(t) for t in alt.teams]
                    entry["team_names"] = [f"team_{t + 1}" for t in range(len(alt.teams))]
            alt_projects = es.projects_from_ids(plain, students)
            alt_reports = dict(base_reports)
            alt_reports.update(es.evaluate_projects(alt_projects, config, students,
                                                    policy=policy, raw_df=raw_df,
                                                    project_ids=rescored))
            alt_stats = stats_view(alt.stats)
            record = ra.record_run(
                label=f"alternative {n} for project {pid}", source="engine",
                run_config=config, policy=policy, projects=alt_projects, reports=alt_reports,
                extra_summary={"method": "alternatives",
                               "objective_gap": alt_stats.objective_gap,
                               "pair_overlap": alt_stats.pair_overlap},
                roster=ra.roster_identity(students, raw_df),
                parent=parent, activate=False,
            )
            hard_alt, repeats_alt = run_wide(alt_reports)
            overlap = alt_stats.pair_overlap or {}
            overlaps = [float(v) for v in (overlap.values() if isinstance(overlap, dict)
                                           else overlap)]
            saved.append({"run_id": record.run_id, "label": rh.run_title(record),
                          "objective": float(alt.objective),
                          "gap": float(alt_stats.objective_gap or 0.0),
                          "max_overlap": max(overlaps) if overlaps else None,
                          "quality": overall(alt_reports[pid]).quality_index_soft,
                          "hard": hard_alt, "repeats": repeats_alt,
                          "hard_before": hard_now, "repeats_before": repeats_now,
                          "rescored_all": not same_rules})
        st.session_state[_ALTERNATIVES_KEY] = saved
        flash(f"Saved {len(saved)} alternative(s) as runs "
              f"{', '.join(s['run_id'] for s in saved)}.")
        st.rerun()

    saved = st.session_state.get(_ALTERNATIVES_KEY) or []
    if saved:
        st.markdown("**Latest alternatives** (run-wide totals: change against the run they "
                    "were generated from)")
        if any(entry.get("rescored_all") for entry in saved):
            st.caption("The allocation rules changed after the run was scored, so the run and "
                       "every alternative were scored with the current rules (the run's own "
                       "headline still uses the earlier ones).")
        for entry in saved:
            a1, a2 = st.columns([4, 1], vertical_alignment="center")
            overlap = entry.get("max_overlap")
            hard, hard_before = entry.get("hard"), entry.get("hard_before")
            repeats, repeats_before = entry.get("repeats"), entry.get("repeats_before")
            run_wide_text = ""
            if hard is not None and hard_before is not None:
                run_wide_text = (f" · run-wide hard violations {change_text(hard, hard_before)}"
                                 f" · repeat pairs {change_text(repeats, repeats_before)}")
            a1.markdown(
                f"**{entry['label']}** · objective {entry['objective']:.2f} "
                f"(gap {entry['gap']:+.2f}) · quality {fmt_pct(entry.get('quality'), digits=0, missing='—')}"
                + (f" · max pair overlap {float(overlap):.0%}" if overlap is not None else "")
                + run_wide_text)
            if hard is not None and hard_before is not None and hard > hard_before:
                a1.warning(f"Using {entry['run_id']} adds {hard - hard_before} unit(s) of hard-rule "
                           "violation across the run (e.g. repeat pairs in a later project).")
            if a2.button("Use", key=f"alt_use_{entry['run_id']}", width="stretch"):
                record = rh.get_run(entry["run_id"])
                if record is None:
                    st.warning("That run is no longer in the history.")
                else:
                    try:
                        warnings = eo.activate_run(record, config, students,
                                                   raw_df=loader.get_dataframe())
                    except (eo.EditError, ValueError, KeyError) as exc:
                        st.error(f"Could not activate {rh.run_title(record)}: {exc}")
                    else:
                        flash(f"{rh.run_title(record)} is now the active allocation.")
                        for warning in warnings:
                            flash(warning, "warning")
                        if hard is not None and hard_before is not None and hard > hard_before:
                            flash(f"Run-wide hard violations went from {hard_before} to {hard}.",
                                  "warning")
                        st.rerun()


if es.has_active_allocation():
    alloc = es.active_allocation()
    result = alloc.formation_result
    projects = alloc.projects
    result_config = result.config
    metrics = DiversityMetrics(result_config, cohort_students=students)
    reports = engine_reports_for(projects)
    active_record = rh.active_run()
    meta = active_meta(active_record)
    summary = st.session_state.get(_SUMMARY_KEY)
    if summary and (active_record is None or summary.get("run_id") != active_record.run_id):
        summary = None
    is_engine_run = meta.get("source") == "engine"
    from_classic = classic_result(result)
    unsaved_edits = not from_classic and active_record is not None and alloc.unsaved_edits

    st.markdown("---")
    st.subheader("📊 Results Summary")

    if summary:
        fresh = st.session_state.pop(_SUMMARY_FRESH_KEY, False)
        if fresh:
            verb = "Stopped early and kept" if summary.get("stopped_early") else "Formed"
            st.success(
                f"✅ **{verb} {summary['teams_formed']} teams across {summary['n_generated']} "
                f"generated project(s) in {summary['total_time']:.1f} seconds.**"
                + (f" Saved as **{summary['run_label']}**." if summary.get("run_label") else "")
            )
        with st.expander("📊 Execution Summary", expanded=fresh):
            ecol1, ecol2 = st.columns(2)
            ecol1.metric("Total Time", f"{summary['total_time']:.1f}s")
            ecol1.metric(
                "Avg Time per Generated Project",
                f"{summary['total_time'] / max(1, summary['n_generated']):.2f}s",
            )
            ecol2.metric("Teams Formed", summary["teams_formed"],
                         help="Teams of the generated projects (imported projects excluded).")
            if summary.get("algorithm") == "engine":
                ecol2.metric("Worker processes", summary["num_cores"])
                seeds = meta.get("seeds") or []
                st.caption(f"Method {meta.get('method', '—')} · seed "
                           f"{'random' if meta.get('seed') is None else meta.get('seed')}"
                           + (f" (restart seeds {', '.join(str(s) for s in seeds)})"
                              if seeds and seeds[0] is not None else ""))
            else:
                ecol2.metric("CPU Cores Requested", summary["num_cores"])
            if summary.get("mp_failed"):
                st.warning(
                    f"⚠️ **Multiprocessing failed:** requested {summary['num_cores']} cores but "
                    "the run fell back to single-core execution. Results are valid; set CPU "
                    "Cores to 1 to avoid the start-up overhead."
                )
            rows = []
            for pid, stats in summary.get("per_project", {}).items():
                row = {
                    "Project": f"Project {pid}",
                    "Optimizer Score": stats.get("score"),
                    "Iterations": stats.get("iterations"),
                    "Time (s)": stats.get("execution_time"),
                    "Converged": "yes" if stats.get("converged") else "no",
                    "Cores Used": stats.get("actual_cores_used"),
                }
                if stats.get("restarts_planned"):
                    row["Restarts"] = (f"{int(stats.get('restarts_completed') or 0)}/"
                                       f"{int(stats['restarts_planned'])}")
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                if summary.get("algorithm") == "engine":
                    st.caption("Engine scores are penalty points: lower is better. 'Converged: "
                               "no' means the search hit its time limit (or was stopped) before "
                               "its move budget was used up.")

    engine_summary: Dict[str, Any] = {}
    if reports:
        engine_summary = render_engine_headline(reports, meta)
    elif not from_classic:
        st.info("This allocation has no rule-based report (it was restored or edited "
                "elsewhere).")
    if unsaved_edits:
        st.caption(f"**+ unsaved Team Editor edits** of {rh.run_title(active_record)}: the numbers "
                   "describe the edited allocation.")
    departed_warning(projects, [s.student_id for s in students])
    for warning in meta.get("warnings", []):
        st.warning(f"⚠️ {warning}")

    team_scores = [
        classic_team_score(metrics, team)
        for project in projects if not project.imported
        for team in project.teams
    ]
    generated_teams = sum(len(p.teams) for p in projects if not p.imported)
    participants_spec = st.session_state.get("project_participants")
    sizes_found = size_issues(result_config, projects)
    coverage_found = coverage_issues(projects, roster_ids, participants_spec)
    policy_view = es.rules_of_active_reports()

    mcols = st.columns(3 if reports else 4)
    mcols[0].metric("Execution Time", f"{result.execution_time:.2f}s")
    mcols[1].metric("Teams Formed", generated_teams,
                    help="Teams of the generated projects (imported projects excluded).")
    mcols[2].metric(
        "Violations", len(result.constraint_violations),
        help=("Team-level rule issues (hard and soft) from the rule report; details in the "
              "⚠️ Violations tab." if not from_classic else
              "Classic constraint violations (⚙️ Configuration › Constraints); the rule-based "
              "numbers are above."))
    if not reports:
        mcols[3].metric("Classic diversity score", f"{result.total_score:.2f}",
                        help="Legacy display index (sum of per-team entropy-based diversity, "
                             "0-1 each). This run has no rule-based report to show instead.")

    with st.expander("📈 Assessment & Recommendations", expanded=False):
        if reports and not from_classic:
            advice = engine_recommendations(engine_summary, meta, coverage_found)
        else:
            advice = build_recommendations(result, team_scores, sizes_found, coverage_found,
                                           summary, engine_summary, policy_view)
        for level, text in advice:
            getattr(st, level)(text)
        if team_scores:
            st.markdown("**Classic diversity metrics** (legacy display index, 0-1 per team; the "
                        "rules engine does not optimise it — compare runs by quality vs random)")
            acol1, acol2, acol3 = st.columns(3)
            acol1.metric("Weakest team (classic)", f"{min(team_scores):.3f}")
            acol2.metric("Average team (classic)", f"{np.mean(team_scores):.3f}")
            acol3.metric("Strongest team (classic)", f"{max(team_scores):.3f}")
            if reports:
                st.caption(f"Sum over all teams: {result.total_score:.2f}.")

    link_cols = st.columns(3)
    with link_cols[0]:
        page_link("editor", "Fine-tune in the Team Editor")
    with link_cols[1]:
        page_link("compare", "Compare runs")
    with link_cols[2]:
        page_link("analytics", "Analytics")

    if reports and is_engine_run:
        with st.expander("🔀 Generate alternatives", expanded=False):
            render_alternatives(projects, reports)

    tab_names = []
    if reports:
        tab_names += ["🧭 Teams & rules", "📋 Rule compliance"]
    tab_names += ["👥 Team View", "📈 Classic diversity scores", "🔄 Overlap Analysis",
                  "⚠️ Violations"]
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    if reports:
        frame_now = es.roster_frame(students, loader.get_dataframe())
        with tabs["🧭 Teams & rules"]:
            render_team_cards(projects, reports, frame_now, policy_view)
        with tabs["📋 Rule compliance"]:
            render_rule_compliance(reports)

    with tabs["👥 Team View"]:
        extension_team_view = (extensions.hook("ui.team_formation.team_view")
                            if extensions.active() else None)
        if extension_team_view is not None:
            extension_team_view(projects, students, metrics)
        else:
            render_roster_team_view(projects, es.roster_frame(students, loader.get_dataframe()))

    with tabs["📈 Classic diversity scores"]:
        st.subheader("Classic Diversity Score Breakdown")
        st.caption("Legacy entropy-based display index (0-1) of the classic algorithms. The "
                   "rules engine does not optimise it; judge allocations by the rule-based "
                   "numbers (quality vs random, hard violations, repeat pairs) instead.")
        shown_scores = extensions.without_inactive_metrics(result.diversity_scores)
        if shown_scores:
            scores_df = pd.DataFrame([
                {"Attribute": k.replace("_", " ").title(), "Score": v}
                for k, v in shown_scores.items()
            ])
            st.plotly_chart(classic_score_bar(scores_df, x="Attribute", y="Score",
                                              title="Average Diversity Scores by Attribute"),
                            width="stretch")

        if len(projects) > 1:
            st.subheader("Diversity Across Projects")
            project_scores = []
            for project in projects:
                project_metrics = metrics.calculate_project_diversity(project)
                project_scores.append({
                    "Project": project.name + (" (imported)" if project.imported else ""),
                    "Average Score": project_metrics.get("average_weighted", 0),
                    "Min Score": project_metrics.get("min_score", 0),
                    "Max Score": project_metrics.get("max_score", 0),
                })
            fig = px.bar(
                pd.DataFrame(project_scores),
                x="Project",
                y=["Average Score", "Min Score", "Max Score"],
                barmode="group",
                title="Diversity Scores by Project"
            )
            st.plotly_chart(fig, width="stretch")

    with tabs["🔄 Overlap Analysis"]:
        st.subheader("Cross-Project Overlap Analysis")
        if len(projects) > 1:
            counts = es.allocation_pair_history(projects, policy_view,
                                                es.roster_frame(students, loader.get_dataframe()),
                                                even_when_off=True)
            render_scored_rules_caption("Counted with",
                                        ", the rules the numbers on this page were scored with")
            pair_count_dist: Dict[int, int] = {}
            for count in counts.cap_counts.values():
                pair_count_dist[count] = pair_count_dist.get(count, 0) + 1
            if reports:
                repeat_total = int(engine_summary.get("repeated_pairs") or 0)
                over_total = int(engine_summary.get("over_cap_pairs") or 0)
            else:
                cap = es.repeat_cap(policy_view)
                counted = policy_view.history.enabled
                repeat_total = sum(1 for c in counts.cap_counts.values() if counted and c > 1)
                over_total = sum(1 for c in counts.cap_counts.values()
                                 if cap is not None and c > cap)
            ocol1, ocol2 = st.columns(2)
            with ocol1:
                st.markdown("### Pairing Distribution")
                dist_df = pd.DataFrame([
                    {"Teams shared": k, "Number of Pairs": v}
                    for k, v in sorted(pair_count_dist.items())
                ])
                if not dist_df.empty:
                    fig = px.bar(dist_df, x="Teams shared", y="Number of Pairs",
                                 title="How Many Teams Pairs Have Shared")
                    st.plotly_chart(fig, width="stretch")
            with ocol2:
                st.markdown("### Overlap Statistics")
                st.metric("Pairs who shared a team", len(counts.cap_counts))
                st.metric("Repeat pairs (generated projects)", repeat_total,
                          help="Teammates in a generated project who already shared a team "
                               "before — the same number as the headline above.")
                st.metric("Over the cap (generated projects)", over_total)
                st.metric("Section-mates reunited", es.section_reunions(engine_summary, reports),
                          help="Teammates whose only earlier contact was a section-sized group.")
                st.metric("Repeat cap (rules used for these scores)", es.cap_label(policy_view),
                          help=f"The cap of {es.rules_source_text()}.")
            repeated = [
                {"Project": pid, "Student 1": pair.a, "Student 2": pair.b,
                 "Teams shared before": pair.times_before,
                 "Met in": ", ".join(str(m) for m in pair.met_in or []),
                 "Over the cap": "yes" if pair.over_cap else ""}
                for pid, report in reports.items()
                for pair in history_section(report).pairs
            ]
            if repeated:
                st.markdown("### Repeat Pairs")
                repeated.sort(key=lambda r: (-int(r["Teams shared before"] or 0), r["Project"]))
                st.dataframe(pd.DataFrame(repeated[:50]), width="stretch", hide_index=True)
        else:
            st.info("Overlap analysis is available when multiple projects are generated.")

    with tabs["⚠️ Violations"]:
        st.subheader("Constraint Violations")
        tagged = ([(True, text) for text in result.constraint_violations] if from_classic
                  else es.tagged_violations(projects, es.active_reports()))
        n_hard = sum(1 for hard, _ in tagged if hard)
        n_soft = len(tagged) - n_hard
        if tagged:
            if n_hard:
                st.error(f"Found {n_hard} hard-rule violation(s)"
                         + (f" and {n_soft} soft-rule issue(s)" if n_soft else ""))
            else:
                st.warning(f"Every hard rule holds; {n_soft} soft-rule issue(s) remain (they "
                           "cost penalty points but are allowed)")
            with st.expander("ℹ️ What do these violations mean?"):
                if not from_classic:
                    st.markdown(
                        """
                        Each line is a team that breaks an allocation rule, checked against
                        what happened **before** that project (imported history and earlier
                        generated projects). **[hard]** lines break a hard rule (1000 points
                        per unit) and are listed first; **[soft]** lines break a soft rule,
                        which only costs its weight. The 📋 Rule compliance tab summarises
                        them per rule; the feasibility check on 🧭 Allocation Rules shows
                        whether a rule can be met at all.
                        """
                    )
                else:
                    st.markdown(
                        """
                        Each line is a generated team that breaks a **hard** rule, checked against
                        what happened **before** that project (imported history and earlier
                        generated projects):

                        - **max_repeat_pairing** — a pair shares a team more often than the
                          classic *Max Repeat Pairings* setting allows (classic algorithms only;
                          it counts every imported group, sections included).
                        - **team_size** — a team is outside the min/max size.
                        - **avoid_constraint** — two students who asked to avoid each other share a team.

                        **What you can do:** raise *Max Repeat Pairings* or widen team sizes
                        (⚙️ Configuration), keep *Local search repair* on, or try another algorithm.
                        The rule-based numbers at the top (repeat pairs, hard violations) use the
                        repeat cap of the allocation rules instead.
                        """
                    )
            for hard, violation in tagged:
                if hard:
                    st.error(f"• {violation} [hard]")
                else:
                    st.warning(f"• {violation} [soft]")
        else:
            st.success("✅ No constraint violations!")

        st.markdown("### Validation Summary")
        if coverage_found:
            st.error("Participants not assigned:")
            for coverage_issue in coverage_found:
                st.write(f"• {coverage_issue}")
        else:
            st.success("✅ All participants of generated projects assigned to teams")
        if sizes_found:
            st.error("Team size issues:")
            for issue in sizes_found:
                st.write(f"• {issue}")
        else:
            st.success("✅ All teams within size limits")

st.markdown("---")
ncol1, ncol2, ncol3, ncol4 = st.columns(4)
with ncol1:
    if st.button("← Configuration", width="stretch"):
        switch_to("configuration")
with ncol2:
    if st.button("🧭 Allocation Rules", width="stretch"):
        switch_to("rules")
with ncol3:
    if st.button("📈 Analytics →", width="stretch"):
        switch_to("analytics")
with ncol4:
    if st.button("📤 Export →", width="stretch"):
        switch_to("export")
