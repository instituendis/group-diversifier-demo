import math
import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import extensions  # noqa: E402
from app.demo import classic_unavailable_where  # noqa: E402
from app.core.config_session import (  # noqa: E402
    MANUAL_SIZES_KEY,
    apply_configuration,
    apply_pending_preset,
    derived_team_sizes,
    offered_algorithms,
    queue_preset,
)
from app.core.data_session import (  # noqa: E402
    ensure_session_state,
    get_data_loader,
    is_new_upload,
    mark_upload_processed,
    page_link,
    render_flashes,
    switch_to,
)
from app.core.models import ALGORITHMS, Configuration  # noqa: E402
from app.engine.policy import Policy  # noqa: E402
from app.engine.sizing import balanced_sizes, describe_sizes, team_count  # noqa: E402
from app.utils.exporters import import_configuration_from_json  # noqa: E402

st.set_page_config(
    page_title="Configuration | Group Diversifier",
    page_icon="⚙️",
    layout="wide",
)

ensure_session_state()

ALGORITHMS = list(ALGORITHMS)
ALGORITHM_LABELS = {
    "engine": "🧭 Rules engine (recommended)",
    "greedy": "🚀 Greedy (classic, fast)",
    "simulated_annealing": "🔥 Simulated Annealing (classic)",
    "genetic": "🧬 Genetic Algorithm (classic, thorough)",
}
CLASSIC_NOTE = ("These settings drive the **classic algorithms** (greedy, simulated annealing, "
                "genetic). The rules engine uses the rules on the 🧭 Allocation Rules page instead.")

config = st.session_state.config


def _seed_widget(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


apply_pending_preset()

st.title("⚙️ Configuration")
st.markdown("Configure team formation parameters, diversity weights, and constraints.")
render_flashes()

with st.expander("ℹ️ What is this page?", expanded=False):
    st.markdown("""
    This page sets **how many teams of what size** are formed and **which algorithm**
    forms them.

    - **Team Settings**: minimum, maximum and target team sizes (or a number of teams),
      the number of projects and per-project sizes
    - **Algorithm**: the rules engine (recommended; its rules live on the 🧭 Allocation
      Rules page) or a classic algorithm (greedy, simulated annealing, genetic)
    - **Diversity Weights / Constraints (classic)**: settings used by the classic
      algorithms only, on the profile dimensions
    - **Summary**: review all settings before running
    """)


def _extension_ui(name: str):
    return extensions.hook(f"ui.configuration.{name}") if extensions.active() else None


def _preset_description(path):
    import json
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    text = payload.get("description") if isinstance(payload, dict) else None
    return text.strip() if isinstance(text, str) else ""


with st.expander("📁 Import / Presets", expanded=False):
    st.markdown(
        "Load a **full configuration JSON** (downloaded from the Export page): it replaces "
        "weights, dimensions, proportionality, constraints, algorithm settings and team sizes "
        "including per-project overrides (a file without allocation rules keeps the current "
        "rules). A **bundled preset** only sets the classic algorithms' weights and "
        "constraints; team sizes, projects, the algorithm and the allocation rules stay."
    )

    def _apply_imported_configuration(payload, source_label):
        apply_configuration(payload["configuration"], source_label=source_label)
        st.rerun()

    def _apply_classic_preset(payload, source_label):
        apply_configuration(payload["configuration"], mode="preset",
                            source_label=source_label)
        st.rerun()

    _import_col, _preset_col = st.columns(2)

    with _import_col:
        st.markdown("#### 📥 Import from file")
        uploaded_file = st.file_uploader(
            "Configuration JSON file",
            type=["json"],
            help="A configuration export downloaded from the Export page",
        )
        if uploaded_file is not None and is_new_upload("config_json", uploaded_file):
            try:
                imported = import_configuration_from_json(
                    uploaded_file.getvalue().decode("utf-8")
                )
            except (ValueError, UnicodeDecodeError) as e:
                st.error(f"Could not import configuration: {e}")
            else:
                mark_upload_processed("config_json", uploaded_file)
                _apply_imported_configuration(imported, uploaded_file.name)

    with _preset_col:
        st.markdown("#### 🧩 Bundled presets (classic weights)")
        presets_dir = extensions.preset_dirs(ROOT_DIR / "config")[0]
        preset_files = sorted(presets_dir.glob("*.json")) if presets_dir.is_dir() else []
        if not preset_files:
            st.info("No presets found.")
        else:
            preset_names = [p.stem for p in preset_files]
            chosen_preset = st.selectbox(
                "Available presets", preset_names, key="config_preset_choice"
            )
            _preset_text = _preset_description(presets_dir / f"{chosen_preset}.json")
            if _preset_text:
                st.caption(_preset_text)
            if st.button("Load Preset", type="primary"):
                preset_path = presets_dir / f"{chosen_preset}.json"
                try:
                    imported = import_configuration_from_json(
                        preset_path.read_text(encoding="utf-8")
                    )
                except ValueError as e:
                    st.error(f"Could not load preset '{chosen_preset}': {e}")
                else:
                    _apply_classic_preset(imported, chosen_preset)

loader = get_data_loader()
students = loader.get_students()
n_students = len(students)

if n_students == 0:
    st.warning("No students loaded. Please load data first.")
    if st.button("Go to Data Overview"):
        switch_to("data")
    st.stop()

st.info(f"📊 **{n_students} students** available for team formation")


def imported_project_row(imp, p_idx: int) -> str:
    teams = imp.teams_by_project.get(imp.imported_project_ids[p_idx], {}) if imp else {}
    sizes = [len(m) for m in teams.values() if m]
    detail = f"{describe_sizes(sizes)} · {sum(sizes)} students" if sizes else "no teams"
    return (
        '<div style="background: rgba(128,128,128,0.14); color: rgba(128,128,128,0.95); '
        'border: 1px dashed rgba(128,128,128,0.45); border-radius: 6px; '
        'padding: 0.55rem 0.8rem; margin: 0.25rem 0 0.75rem;">'
        f"📥 <b>Project {p_idx + 1}</b> · uploaded (locked) · {detail}"
        '<br><span style="font-size: 0.85em;">Its teams are already set by the upload, so '
        "sizes here do not apply.</span></div>"
    )


def render_tuning(config: Configuration) -> None:
    if config.algorithm == "greedy":
        st.caption("Greedy has no tuning parameters. With more than one core it runs "
                   "independent restarts and keeps the best result.")
    elif config.algorithm == "simulated_annealing":
        st.markdown("#### Parameter Tuning")
        auto_temperature = st.toggle(
            "Calibrate temperature automatically",
            value=bool(getattr(config, "sa_auto_temperature", True)),
            key="sa_auto_temperature",
            help="Derive the start temperature from sampled score changes and the cooling "
                 "from Max Iterations (Initial Temperature and Cooling Rate are then ignored)",
        )
        col1, col2 = st.columns(2)
        with col1:
            sa_temp = st.number_input(
                "Initial Temperature",
                min_value=100.0,
                max_value=10000.0,
                value=float(min(max(config.sa_initial_temp, 100.0), 10000.0)),
                step=100.0,
                disabled=auto_temperature,
                help="Controls initial randomness: higher values accept more worse moves early on."
            )
            sa_cooling = st.slider(
                "Cooling Rate",
                min_value=0.9,
                max_value=0.999,
                value=float(min(max(config.sa_cooling_rate, 0.9), 0.999)),
                step=0.001,
                format="%.3f",
                disabled=auto_temperature,
                help="Temperature multiplier per iteration (slower cooling = longer search)."
            )
        with col2:
            sa_iterations = st.number_input(
                "Max Iterations",
                min_value=1000,
                max_value=100000,
                value=int(min(max(config.sa_iterations, 1000), 100000)),
                step=1000,
                help="Upper bound on proposed moves per chain."
            )
        if not auto_temperature:
            effective = int(math.log(0.1 / sa_temp) / math.log(sa_cooling)) + 1
            if effective < sa_iterations:
                st.info(f"With this temperature and cooling rate annealing ends after about "
                        f"{effective:,} iterations; raise the cooling rate to use more of the "
                        f"{int(sa_iterations):,} allowed.")
        config.sa_auto_temperature = bool(auto_temperature)
        config.sa_initial_temp = sa_temp
        config.sa_cooling_rate = sa_cooling
        config.sa_iterations = int(sa_iterations)
    elif config.algorithm == "genetic":
        st.markdown("#### Parameter Tuning")
        col1, col2 = st.columns(2)
        with col1:
            ga_pop = st.number_input(
                "Population Size",
                min_value=20,
                max_value=500,
                value=int(min(max(config.ga_population_size, 20), 500)),
                step=10,
                help="Solutions evolved in parallel. Larger populations explore more but run slower."
            )
            ga_gen = st.number_input(
                "Generations",
                min_value=50,
                max_value=2000,
                value=int(min(max(config.ga_generations, 50), 2000)),
                step=50,
                help="Evolution cycles; the run stops early when it stagnates."
            )
        with col2:
            ga_mut = st.slider(
                "Mutation Rate",
                min_value=0.01,
                max_value=0.5,
                value=float(min(max(config.ga_mutation_rate, 0.01), 0.5)),
                step=0.01,
                help="Probability of random changes: low = stable, high = more exploration."
            )
        config.ga_population_size = int(ga_pop)
        config.ga_generations = int(ga_gen)
        config.ga_mutation_rate = ga_mut


tab1, tab4, tab2, tab3, tab5 = st.tabs([
    "👥 Team Settings",
    "🔧 Algorithm",
    "⚖️ Diversity Weights (classic)",
    "🚫 Constraints (classic)",
    "📋 Summary"
])

with tab1:
    st.subheader("Team Size Configuration")
    st.caption("Define how many students should be in each team. The algorithm will try to create teams close to the target size.")
    _cap = max(2, n_students)
    size_mode = st.radio(
        "Set team sizes by",
        ["Team size", "Number of teams"],
        horizontal=True,
        index=1 if int(config.num_teams or 0) > 0 else 0,
        help="Pick a target number of teams and the min/target/max sizes are derived "
             f"from the {n_students} students (e.g. 20 teams → 12×5 + 8×6)."
    )
    if size_mode == "Number of teams":
        if int(config.num_teams or 0) == 0:
            st.session_state[MANUAL_SIZES_KEY] = (int(config.team_size_min),
                                                  int(config.team_size_max),
                                                  int(config.team_size_target))
        max_teams = max(1, n_students // 2)
        _seed_widget(
            "n_teams_input",
            min(int(config.num_teams) if int(config.num_teams or 0) > 0
                else max(1, round(n_students / max(1, config.team_size_target))), max_teams),
        )
        if st.session_state["n_teams_input"] > max_teams:
            st.session_state["n_teams_input"] = max_teams
        n_teams = st.number_input(
            "Number of Teams",
            min_value=1,
            max_value=max_teams,
            key="n_teams_input",
            help="All students are split into this many teams per project"
        )
        team_size_min, team_size_target, team_size_max, dist = derived_team_sizes(
            n_students, int(n_teams))
        st.success(
            f"→ **{n_teams} teams**: " + dist
            + f"  (min {team_size_min} / target {team_size_target} / max {team_size_max})"
        )
    else:
        base_min, base_max, base_target = (int(config.team_size_min), int(config.team_size_max),
                                           int(config.team_size_target))
        manual = st.session_state.get(MANUAL_SIZES_KEY)
        if int(config.num_teams or 0) > 0 and manual:
            base_min, base_max, base_target = manual
        col1, col2, col3 = st.columns(3)
        with col1:
            team_size_min = st.number_input(
                "Minimum Team Size",
                min_value=2,
                max_value=_cap,
                value=min(max(base_min, 2), _cap),
                help="Smallest allowed team size"
            )
        with col2:
            team_size_max = st.number_input(
                "Maximum Team Size",
                min_value=team_size_min,
                max_value=_cap,
                value=min(max(base_max, team_size_min), _cap),
                help="Largest allowed team size"
            )
        with col3:
            team_size_target = st.number_input(
                "Target Team Size",
                min_value=team_size_min,
                max_value=team_size_max,
                value=min(max(base_target, team_size_min), team_size_max),
                help="Preferred team size (algorithm will try to achieve this)"
            )
        st.session_state[MANUAL_SIZES_KEY] = (int(team_size_min), int(team_size_max),
                                              int(team_size_target))

    st.markdown("---")
    st.subheader("Team Distribution Preview")
    n_teams_min = math.ceil(n_students / team_size_max)
    n_teams_max = n_students // team_size_min
    n_teams_target = team_count(n_students, int(team_size_min), int(team_size_max),
                                int(team_size_target),
                                int(n_teams) if size_mode == "Number of teams" else 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Min Teams Possible", n_teams_min)
    col2.metric("Target Teams", n_teams_target)
    col3.metric("Max Teams Possible", n_teams_max)
    if n_teams_min > n_teams_max:
        st.error(f"No split of {n_students} students fits sizes {team_size_min}-{team_size_max}; "
                 "widen the size range.")
    elif n_teams_target:
        st.info(f"Target split: {describe_sizes(balanced_sizes(n_students, n_teams_target))}")

    st.markdown("---")
    st.subheader("Project Configuration")
    num_projects = st.slider(
        "Number of Projects",
        min_value=1,
        max_value=10,
        value=min(max(int(config.num_projects), 1), 10),
        help="Number of different project rotations to generate"
    )

    if num_projects > 1:
        st.info(f"🔄 Teams will be rotated across {num_projects} projects to maximize diversity of collaborations")
        st.markdown("#### Per-Project Team Sizes")
        customize_sizes = st.toggle(
            "Customize team sizes per project",
            value=bool(config.per_project_team_sizes),
            help="When enabled, each project can have its own min/target/max team size; "
                 "otherwise the global settings above apply to every project"
        )

        imported_cfg = st.session_state.get("imported_assignments")
        imported_slot_count = imported_cfg.project_count if imported_cfg else 0
        if customize_sizes:
            per_project_sizes = []
            for p_idx in range(num_projects):
                override = (
                    config.per_project_team_sizes[p_idx]
                    if p_idx < len(config.per_project_team_sizes) else None
                ) or {}
                slot_imported = p_idx < imported_slot_count
                if slot_imported:
                    st.markdown(imported_project_row(imported_cfg, p_idx), unsafe_allow_html=True)
                    per_project_sizes.append(dict(override) if override else {
                        "min": int(team_size_min), "target": int(team_size_target),
                        "max": int(team_size_max),
                        "num_teams": int(n_teams) if size_mode == "Number of teams" else 0,
                    })
                    continue
                c0, c1, c2, c3 = st.columns([2, 2, 2, 2])
                default_teams = (int(override.get("num_teams", 0) or 0) if override
                                 else int(n_teams) if size_mode == "Number of teams" else 0)
                with c0:
                    p_teams = st.number_input(
                        f"Project {p_idx + 1} — # teams",
                        min_value=0,
                        max_value=max(1, n_students // 2),
                        value=min(default_teams, max(1, n_students // 2)),
                        key=f"pp_teams_{p_idx}",
                        help=f"0 = set sizes manually. Otherwise sizes are derived from the "
                             f"{n_students} students (e.g. 20 teams → 12×5 + 8×6).",
                    )
                if p_teams > 0:
                    p_min, p_target, p_max, dist = derived_team_sizes(n_students, int(p_teams))
                    c1.caption(f"min **{p_min}**")
                    c2.caption(f"target **{p_target}**")
                    c3.caption(f"max **{p_max}** ({dist})")
                else:
                    with c1:
                        p_min = st.number_input(
                            f"Project {p_idx + 1} — min", min_value=2, max_value=_cap,
                            value=int(min(max(override.get("min", team_size_min), 2), _cap)),
                            key=f"pp_min_{p_idx}",
                            )
                    with c2:
                        p_target = st.number_input(
                            f"Project {p_idx + 1} — target", min_value=2, max_value=_cap,
                            value=int(min(max(override.get("target", team_size_target), 2), _cap)),
                            key=f"pp_target_{p_idx}",
                            )
                    with c3:
                        p_max = st.number_input(
                            f"Project {p_idx + 1} — max", min_value=2, max_value=_cap,
                            value=int(min(max(override.get("max", team_size_max), 2), _cap)),
                            key=f"pp_max_{p_idx}",
                            )
                    if p_min > p_target or p_target > p_max:
                        st.warning(f"Project {p_idx + 1}: target must be between min and max — values adjusted")
                        p_min, p_target, p_max = sorted([p_min, p_target, p_max])
                per_project_sizes.append({
                    "min": int(p_min), "target": int(p_target), "max": int(p_max),
                    "num_teams": int(p_teams) if p_teams > 0 else 0,
                })
            config.per_project_team_sizes = per_project_sizes
        else:
            config.per_project_team_sizes = []
    else:
        config.per_project_team_sizes = []

    config.team_size_min = int(team_size_min)
    config.team_size_max = int(team_size_max)
    config.team_size_target = int(team_size_target)
    config.num_teams = int(n_teams) if size_mode == "Number of teams" else 0
    config.num_projects = int(num_projects)

with tab2:
    st.subheader("Diversity Weight Configuration")
    st.info(CLASSIC_NOTE)
    page_link("rules", "Open Allocation Rules")
    st.caption("🎯 These weights drive the classic team-formation optimizers.")

    st.subheader("Configuration Presets")

    quick_presets = _extension_ui("quick_presets")
    if quick_presets is not None:
        quick_presets()
    else:
        _all_dims = list(config.dimensions_config or {})
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("🎯 Balanced Preset", width="stretch", disabled=not _all_dims,
                         help="Every profile dimension at weight 50"):
                queue_preset({}, dim_weights={name: 50.0 for name in _all_dims})
                st.rerun()
        with preset_cols[1]:
            if st.button("⬜ All Off", width="stretch", disabled=not _all_dims,
                         help="Every profile dimension at weight 0 (switched off)"):
                queue_preset({}, dim_weights={name: 0.0 for name in _all_dims})
                st.rerun()
    st.markdown("---")

    with st.expander("ℹ️ How do weights affect the optimizer? (Click to learn)", expanded=False):
        st.markdown(extensions.text("config.weights_help", """
        ### Weight System Explained

        Each team's optimizer score is a **sum of bonus and penalty terms**. Every
        diversity weight (0-100) scales its attribute's diversity bonus; the
        preference bonus, avoid penalty and overlap penalty on the Constraints tab
        are **absolute** amounts added to the same sum.

        - **0** switches an attribute off completely.
        - Raising one weight increases that attribute's share of the score.
        - Because the preference and penalty terms are absolute, scaling *all*
          weights up or down changes how diversity trades off against them —
          weights are **not** just relative ratios.
        """))

    weight_sliders = _extension_ui("weight_sliders")
    weights = (weight_sliders(config, students, _seed_widget)
               if weight_sliders is not None else {})

    hidden_dim_names = (extensions.hook("ui.configuration.hidden_dimension_names", frozenset())
                        if extensions.active() else frozenset())
    profile_dims = {
        name: dim for name, dim in (config.dimensions_config or {}).items()
        if name not in hidden_dim_names
    }
    if profile_dims:
        st.markdown("---")
        st.subheader("Profile Dimensions")
        st.caption(extensions.text(
            "config.dimensions_caption",
            "The attributes of the active profile and your data. "
            "Weight 0 disables a dimension in optimization."))
        dim_cols = st.columns(2)
        for i, (dim_name, dim_cfg) in enumerate(sorted(profile_dims.items())):
            with dim_cols[i % 2]:
                key = f"dim_weight_{dim_name}"
                _seed_widget(key, int(min(max(float(dim_cfg.get("weight", 50)), 0), 100)))
                new_weight = st.slider(
                    dim_name.replace("_", " ").title(),
                    min_value=0,
                    max_value=100,
                    key=key,
                    help=f"Diversity weight for '{dim_name}'"
                )
                config.dimensions_config[dim_name]["weight"] = float(new_weight)
                config.dimensions_config[dim_name]["enabled"] = new_weight > 0
    elif weight_sliders is None:
        st.info("The active profile has no dimensions to weight. Add some on the "
                "Profile Manager page.")

    st.markdown("---")
    st.subheader("Weight Distribution")
    weights.update({
        name.replace("_", " ").title(): float(cfg.get("weight", 0))
        for name, cfg in sorted(profile_dims.items())
    })
    fig = go.Figure(data=[go.Bar(x=list(weights.keys()), y=list(weights.values()),
                                 marker_color="#2871AD")])
    fig.update_layout(yaxis_title="Weight", yaxis_range=[0, 100], height=300)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("⚖️ Proportionality (quota-style balance)")
    st.caption(extensions.text(
        "config.proportionality_caption",
        "Each team should mirror the roster's category shares — e.g. with 40% of the "
        "students in one city, a 5-person team is happiest at 2 of them."))
    if not config.proportionality_config:
        config.proportionality_config = {}
    prop_candidates = sorted(
        name for name, dim in profile_dims.items() if dim.get("type") == "categorical"
    )
    prop_cols = st.columns(2)
    for i, attr in enumerate(prop_candidates):
        with prop_cols[i % 2]:
            prop_cfg = config.proportionality_config.get(attr, {})
            _seed_widget(f"prop_enabled_{attr}", bool(prop_cfg.get("enabled", False)))
            _seed_widget(f"prop_weight_{attr}", int(min(max(float(prop_cfg.get("weight", 50)), 0), 100)))
            prop_enabled = st.toggle(
                attr.replace("_", " ").title(),
                key=f"prop_enabled_{attr}",
                help=f"Balance {attr} shares across teams to match the cohort",
            )
            prop_weight = st.slider(
                "Weight",
                min_value=0,
                max_value=100,
                key=f"prop_weight_{attr}",
                disabled=not prop_enabled,
                help="Pull strength toward the cohort's shares",
            )
            config.proportionality_config[attr] = {
                "enabled": bool(prop_enabled),
                "weight": float(prop_weight),
            }

with tab3:
    st.subheader("Constraint Configuration")
    st.info(CLASSIC_NOTE)

    with st.expander("ℹ️ How do constraints work? (Click to learn)", expanded=False):
        st.markdown(extensions.text("config.constraints_help", """
        ### Constraint System Explained

        | Type | Behavior | Example |
        |------|----------|---------|
        | **Hard** | Must not be violated; the search repairs violations first | "Alice and Bob cannot be together" |
        | **Soft** | Traded off against diversity via a penalty/bonus | "Prefer Bob as a teammate" |

        | Constraint | Hard/Soft | What it does |
        |------------|-----------|-------------|
        | Team size | Hard | Team sizes within min/max |
        | Avoid Teammates | Configurable | Students who marked "avoid" are not placed together |
        | Prefer Teammates | Soft (bonus) | Bonus when a preferred teammate is in the same team |
        | Max Repeat Pairings (classic only) | Hard | The same pair may share a team at most this many times across projects (every imported group included). The rules engine uses the repeat cap on 🧭 Allocation Rules › 🔁 History & preferences instead |
        | Overlap Penalty | Soft | Penalty for pairs above the repeat limit |
        """))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Preference Constraints")
        st.caption("Control how student preferences are handled")
        avoid_is_hard = st.radio(
            "Avoid Teammate Constraint",
            options=[True, False],
            format_func=lambda x: "Hard (Never together)" if x else "Soft (Penalized)",
            index=0 if config.avoid_is_hard_constraint else 1,
            help="Hard = students who want to avoid each other will never be in the same team"
        )
        if not avoid_is_hard:
            avoid_penalty = st.slider(
                "Avoid Penalty",
                min_value=10,
                max_value=500,
                value=int(min(max(config.avoid_soft_penalty, 10), 500)),
                help="Penalty applied when avoided students are in same team"
            )
        else:
            avoid_penalty = config.avoid_soft_penalty
        prefer_bonus = st.slider(
            "Prefer Teammate Bonus",
            min_value=0,
            max_value=100,
            value=int(min(max(config.prefer_bonus, 0), 100)),
            help="Bonus when preferred teammates are in same team"
        )

        st.markdown("### Cross-Project Constraints")
        st.caption("Control repeat pairings across multiple projects — for the **classic "
                   "algorithms only**. The rules engine and every repeat number in the app "
                   "use the repeat cap on 🧭 Allocation Rules › 🔁 History & preferences.")
        if num_projects > 1:
            max_repeat = st.slider(
                "Max Repeat Pairings (classic algorithms only)",
                min_value=1,
                max_value=num_projects,
                value=min(max(int(config.max_repeat_pairings), 1), num_projects),
                help="Classic algorithms: maximum times the same pair can share a team across "
                     "projects (1 = never repeat a pairing). Not used by the rules engine."
            )
            max_repeat_is_hard = st.radio(
                "Repeat limit is",
                options=[True, False],
                format_func=lambda x: "Hard (never exceeded)" if x else "Soft (penalised)",
                index=0 if config.max_repeat_is_hard else 1,
                horizontal=True,
                key="max_repeat_is_hard",
                help="Soft turns the limit into the Overlap Penalty per pair above it",
            )
            overlap_penalty = st.slider(
                "Overlap Penalty",
                min_value=10,
                max_value=200,
                value=int(min(max(config.overlap_penalty, 10), 200)),
                help="Penalty for each repeat pairing above the limit"
            )
        else:
            max_repeat = max(1, int(config.max_repeat_pairings))
            max_repeat_is_hard = bool(config.max_repeat_is_hard)
            overlap_penalty = config.overlap_penalty

    optional_constraints = _extension_ui("optional_constraints")
    if optional_constraints is not None:
        with col2:
            optional_constraints(config, students)

    config.avoid_is_hard_constraint = avoid_is_hard
    config.avoid_soft_penalty = float(avoid_penalty)
    config.prefer_bonus = float(prefer_bonus)
    config.max_repeat_pairings = int(max_repeat)
    config.max_repeat_is_hard = bool(max_repeat_is_hard)
    config.overlap_penalty = float(overlap_penalty)

with tab4:
    st.subheader("Algorithm Configuration")

    with st.expander("📊 Algorithm overview", expanded=False):
        st.markdown("""
        | Algorithm | How it works | Relative runtime |
        |-----------|--------------|------------------|
        | 🧭 **Rules engine** | Optimises the allocation rules (attribute goals, head-count quotas, no-lone rules, repeat caps, preferences, fairness) with simulated annealing and/or an exact MILP model; reports compliance per rule and quality vs random teams. | Bounded by the time limit per project (🧭 Allocation Rules › Solver) |
        | 🚀 **Greedy** | Classic: builds teams one at a time, adding the candidate that improves the team score most, then repairs/improves the result with pairwise swaps (local search). | Fastest classic — a few seconds per project for ≈100 people |
        | 🔥 **Simulated Annealing** | Classic: starts from a random assignment and proposes swaps/moves, sometimes accepting worse ones early on to escape local optima. | Similar to greedy at default settings; grows with the iterations actually run |
        | 🧬 **Genetic Algorithm** | Classic: evolves a population of assignments by crossover and mutation, repairing team sizes, then applies local search. | Slowest; grows with population × generations |

        Classic algorithms use the weights and constraints of this page; every run (classic
        or engine) is also scored with the allocation rules, so runs can be compared on
        Compare Runs.
        """)

    offered = offered_algorithms()
    if len(offered) > 1:
        algorithm = st.selectbox(
            "Select Algorithm",
            options=offered,
            format_func=lambda x: ALGORITHM_LABELS[x],
            index=offered.index(config.algorithm) if config.algorithm in offered else 0,
        )
    else:
        algorithm = offered[0]
        st.info(f"**{ALGORITHM_LABELS[algorithm]}** · The classic algorithms (greedy, "
                f"simulated annealing, genetic) are not available on {classic_unavailable_where()}; "
                "they remain available when you run Group Diversifier on your own computer.")
    config.algorithm = algorithm

    if algorithm == "engine":
        current = Policy.from_dict(config.policy) if config.policy else None
        st.success("The rules engine optimises the rules on the **🧭 Allocation Rules** page: "
                   "attribute goals, head-count quotas, no-lone rules, repeat-pairing caps, "
                   "student preferences, fairness and the solver settings (method, time limit, "
                   "restarts, seed, workers).")
        if current is not None:
            seed_text = "random" if current.solver.seed is None else current.solver.seed
            st.caption(f"Current rules: **{current.name}** · solver {current.solver.method}, "
                       f"{current.solver.time_limit_s:g} s per project, "
                       f"{current.solver.restarts} restart(s), seed {seed_text}")
        else:
            st.caption("No rules yet: they are suggested from the data when you open "
                       "Allocation Rules or run the engine.")
        page_link("rules", "Open Allocation Rules")
        num_cores = int(config.num_cores)
        local_search = bool(getattr(config, "local_search", True))
    else:
        st.caption("Classic algorithm settings. Results are still scored with the allocation "
                   "rules so they compare with engine runs.")
        max_cores = os.cpu_count() or 1
        num_cores = st.slider(
            "Number of CPU Cores",
            min_value=1,
            max_value=max(2, max_cores),
            value=min(max(int(config.num_cores), 1), max(2, max_cores)),
            help=f"Worker processes for parallel restarts/chains. This machine has {max_cores} cores. "
                 "Process start-up costs a few seconds, so 1 core is often fastest for small cohorts."
        )
        local_search = st.toggle(
            "Local search repair",
            value=bool(getattr(config, "local_search", True)),
            key="local_search_toggle",
            help="Finish every algorithm with a pairwise-swap improvement pass",
        )
        seed_col1, seed_col2 = st.columns(2)
        random_each_run = seed_col1.checkbox(
            "New random seed every run",
            value=config.random_seed is None,
            key="classic_seed_random",
            help="Off = the same settings always produce the same teams",
        )
        seed_value = seed_col2.number_input(
            "Random seed",
            min_value=0,
            value=int(config.random_seed if config.random_seed is not None else 42),
            step=1,
            key="classic_seed",
            disabled=random_each_run,
        )
        config.random_seed = None if random_each_run else int(seed_value)

    config.num_cores = int(num_cores)
    config.local_search = bool(local_search)

    if algorithm != "engine":
        render_tuning(config)

with tab5:
    st.subheader("Configuration Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Team Settings")
        st.write(f"- **Team Size:** {config.team_size_min} - {config.team_size_max} (target: {config.team_size_target})")
        st.write(f"- **Projects:** {config.num_projects}")
        st.write(f"- **Students:** {n_students}")
        st.write(f"- **Expected Teams:** ~{n_teams_target} per project")
        if config.per_project_team_sizes:
            for idx, override in enumerate(config.per_project_team_sizes):
                detail = (f"{override['num_teams']} teams" if override.get("num_teams")
                          else f"{override['min']}-{override['max']} (target {override['target']})")
                st.write(f"  - Project {idx + 1}: {detail}")
        st.markdown("### Algorithm")
        st.write(f"- **Selected:** {ALGORITHM_LABELS.get(config.algorithm, config.algorithm)}")
        if config.algorithm == "engine":
            if config.policy:
                summary_policy = Policy.from_dict(config.policy)
                st.write(f"- **Rules:** {summary_policy.name}")
                st.write(f"- **Solver:** {summary_policy.solver.method}, "
                         f"{summary_policy.solver.time_limit_s:g} s per project")
        else:
            st.write(f"- **CPU Cores:** {config.num_cores}")
            st.write(f"- **Local Search Repair:** {'On' if config.local_search else 'Off'}")
            st.write(f"- **Seed:** {'random' if config.random_seed is None else config.random_seed}")
        if config.algorithm == "simulated_annealing":
            if config.sa_auto_temperature:
                st.write("- **Temperature:** calibrated automatically")
            else:
                st.write(f"- **Temperature:** {config.sa_initial_temp}")
                st.write(f"- **Cooling Rate:** {config.sa_cooling_rate}")
            st.write(f"- **Iterations:** {config.sa_iterations}")
        elif config.algorithm == "genetic":
            st.write(f"- **Population:** {config.ga_population_size}")
            st.write(f"- **Generations:** {config.ga_generations}")
            st.write(f"- **Mutation Rate:** {config.ga_mutation_rate}")
    summary_weights = _extension_ui("summary_weight_lines")
    summary_constraints = _extension_ui("summary_constraint_lines")
    with col2:
        st.markdown("### Diversity Weights (classic)")
        for line in (summary_weights(config, students) if summary_weights is not None else []):
            st.write(line)
        for name, dim_cfg in sorted(profile_dims.items()):
            if dim_cfg.get("enabled", True):
                st.write(f"- {name.replace('_', ' ').title()}: {float(dim_cfg.get('weight', 0)):g}")
        st.markdown("### Constraints (classic)")
        st.write(f"- Avoid: {'Hard' if config.avoid_is_hard_constraint else 'Soft'}")
        st.write(f"- Prefer Bonus: {config.prefer_bonus}")
        if config.num_projects > 1:
            st.write(f"- Max Repeat Pairings (classic only): {config.max_repeat_pairings} "
                     f"({'hard' if config.max_repeat_is_hard else 'soft'})")
        for line in (summary_constraints(config) if summary_constraints is not None else []):
            st.write(line)

st.session_state.config = config

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Back to Data Overview", width="stretch"):
        switch_to("data")
with col2:
    if st.button("🧭 Allocation Rules", width="stretch"):
        switch_to("rules")
with col3:
    if st.button("Continue to Team Formation →", type="primary", width="stretch"):
        switch_to("formation")
