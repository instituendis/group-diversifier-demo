from __future__ import annotations

import copy
import os
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

from app import demo, extensions, hosting
from app.core.models import ALGORITHMS, DEFAULT_ALGORITHM, Configuration
from app.core.session_state import resolve_state
from app.ui_kit import queue_flash

__all__ = [
    "MANUAL_SIZES_KEY",
    "PENDING_PRESET_KEY",
    "preset_key_map",
    "PRESET_KEPT_FIELDS",
    "apply_configuration",
    "apply_pending_preset",
    "clamp_num_cores",
    "coerce_hosted_algorithm",
    "derived_team_sizes",
    "new_configuration",
    "enforce_hosted_configuration",
    "offered_algorithms",
    "purge_widget_state",
    "queue_preset",
    "sanitize_config",
]

MANUAL_SIZES_KEY = "_config_manual_team_sizes"

PRESET_KEPT_FIELDS = frozenset({
    "team_size_min", "team_size_max", "team_size_target", "num_teams", "num_projects",
    "per_project_team_sizes", "profile_name", "profile_version", "custom_profiles", "policy",
    "algorithm", "num_cores",
})

PENDING_PRESET_KEY = "_pending_preset"

def preset_key_map() -> Dict[str, str]:
    return dict(extensions.hook("preset_key_map", {}))


_WIDGET_PREFIXES = ("dim_weight_", "prop_enabled_", "prop_weight_", "pp_")
_WIDGET_KEYS = frozenset({"n_teams_input", "local_search_toggle", "max_repeat_is_hard",
                          "classic_seed_random", "classic_seed", "sa_auto_temperature"})


def _session() -> Any:
    return resolve_state()


def clamp_num_cores(value: Any, *, max_cores: Optional[int] = None) -> int:
    if max_cores is None:
        max_cores = os.cpu_count() or 1
    try:
        cores = int(value)
    except (TypeError, ValueError):
        cores = 1
    return hosting.clamp_workers(max(1, min(cores, max_cores)))


def new_configuration() -> Configuration:
    cfg = Configuration()
    if demo.demo_mode():
        cfg.team_size_min, cfg.team_size_max, cfg.team_size_target = demo.DEMO_TEAM_SIZES
        cfg.num_projects = demo.DEMO_NUM_PROJECTS
        cfg.profile_name = demo.profile_name()
        try:
            import json
            policy = json.loads(demo.DEMO_POLICY.read_text(encoding="utf-8"))
            cfg.policy = policy.get("policy", policy)
        except (OSError, ValueError):
            pass
    return cfg


def offered_algorithms() -> List[str]:
    return [DEFAULT_ALGORITHM] if demo.classic_hidden() else list(ALGORITHMS)


def coerce_hosted_algorithm(cfg: Configuration) -> Optional[str]:
    if not demo.classic_hidden() or cfg.algorithm == DEFAULT_ALGORITHM:
        return None
    previous = cfg.algorithm
    cfg.algorithm = DEFAULT_ALGORITHM
    return f"algorithm '{previous}' replaced by the rules engine: {hosting.CLASSIC_UNAVAILABLE}"


def enforce_hosted_configuration(state: Any = None) -> Optional[str]:
    state = _session() if state is None else state
    config = state.get("config")
    if config is None:
        return None
    note = coerce_hosted_algorithm(config)
    if note:
        queue_flash(state, f"ℹ️ The configuration's {note}.", "info")
    return note


def sanitize_config(cfg: Configuration) -> List[str]:
    notes = []
    if cfg.algorithm not in ALGORITHMS:
        notes.append(f"unknown algorithm '{cfg.algorithm}' replaced by the rules engine")
        cfg.algorithm = DEFAULT_ALGORITHM
    hosted_note = coerce_hosted_algorithm(cfg)
    if hosted_note:
        notes.append(hosted_note)
    clamped = clamp_num_cores(cfg.num_cores)
    if clamped != cfg.num_cores:
        notes.append(f"CPU cores clamped to {clamped}")
        cfg.num_cores = clamped
    sizes = sorted([max(2, int(cfg.team_size_min)), max(2, int(cfg.team_size_target)),
                    max(2, int(cfg.team_size_max))])
    if sizes != [cfg.team_size_min, cfg.team_size_target, cfg.team_size_max]:
        notes.append(f"team sizes reordered to min {sizes[0]} / target {sizes[1]} / max {sizes[2]}")
        cfg.team_size_min, cfg.team_size_target, cfg.team_size_max = sizes
    if not 1 <= int(cfg.num_projects) <= 10:
        cfg.num_projects = max(1, min(int(cfg.num_projects), 10))
        notes.append(f"projects clamped to {cfg.num_projects}")
    if int(cfg.max_repeat_pairings) < 1:
        cfg.max_repeat_pairings = 1
        notes.append("max repeat pairings raised to 1 (0 forbids every pairing)")
    if len(cfg.per_project_team_sizes) > cfg.num_projects:
        cfg.per_project_team_sizes = cfg.per_project_team_sizes[: cfg.num_projects]
        notes.append("per-project sizes beyond the project count dropped")
    return notes


def derived_team_sizes(n_students: int, teams: int) -> Tuple[int, int, int, str]:
    base, rem = divmod(int(n_students), int(teams))
    distribution = (f"{teams - rem}×{base} + {rem}×{base + 1}" if rem
                    else f"{teams}×{base}")
    return base, base, base + (1 if rem else 0), distribution


def purge_widget_state(state: Any = None) -> List[str]:
    state = _session() if state is None else state
    prefixes = _WIDGET_PREFIXES + tuple(extensions.hook("config_widget_prefixes", ()))
    stale = [
        k for k in list(state)
        if isinstance(k, str) and (k.startswith(prefixes) or k in _WIDGET_KEYS)
    ]
    for k in stale:
        del state[k]
    return stale


def apply_configuration(new: Configuration, mode: str = "import", *,
                        source_label: str = "", state: Any = None) -> None:
    state = _session() if state is None else state
    current = state["config"]
    purge_widget_state(state)
    if mode == "preset":
        notes = _apply_classic_fields(new, current)
        queue_flash(state, f"✅ Classic preset applied: {source_label}. Only the classic "
                           "algorithms' weights and constraints changed; team sizes, projects, "
                           "the algorithm and the allocation rules are as before.")
        if notes:
            queue_flash(state, "Adjusted while applying: " + "; ".join(notes), "warning")
        return
    if mode != "import":
        raise ValueError(f"unknown mode: {mode!r}")
    notes = sanitize_config(new)
    if not new.policy and current.policy:
        new.policy = copy.deepcopy(current.policy)
        notes.append("the file has no allocation rules, so the current rules were kept")
    changes = []
    if new.algorithm != current.algorithm:
        changes.append(f"algorithm {current.algorithm} → {new.algorithm}")
    if int(new.num_projects) != int(current.num_projects):
        changes.append(f"projects {current.num_projects} → {new.num_projects}")
    old_sizes = (current.team_size_min, current.team_size_target, current.team_size_max,
                 current.num_teams, current.per_project_team_sizes)
    new_sizes = (new.team_size_min, new.team_size_target, new.team_size_max,
                 new.num_teams, new.per_project_team_sizes)
    if old_sizes != new_sizes:
        changes.append("team sizes")
    state["config"] = new
    state.pop(MANUAL_SIZES_KEY, None)
    queue_flash(state, f"✅ Configuration applied: {source_label}")
    if changes:
        queue_flash(state, "The imported configuration changed: " + "; ".join(changes) + ".",
                    "warning")
    if notes:
        queue_flash(state, "Adjusted while importing: " + "; ".join(notes), "warning")


def _apply_classic_fields(preset: Configuration, current: Configuration) -> List[str]:
    for field in fields(Configuration):
        if field.name in PRESET_KEPT_FIELDS:
            continue
        if field.name == "dimensions_config":
            for name, dim in (current.dimensions_config or {}).items():
                preset_dim = (preset.dimensions_config or {}).get(name)
                if preset_dim:
                    dim["weight"] = float(preset_dim.get("weight", dim.get("weight", 0)))
                    dim["enabled"] = bool(preset_dim.get("enabled", dim.get("enabled", True)))
            continue
        setattr(current, field.name, copy.deepcopy(getattr(preset, field.name)))
    return sanitize_config(current)


def queue_preset(weights: Dict[str, float], dim_weights: Optional[Dict[str, float]] = None,
                 proportionality: Optional[Dict[str, Dict[str, Any]]] = None,
                 state: Any = None) -> None:
    state = _session() if state is None else state
    state[PENDING_PRESET_KEY] = {
        "weights": weights, "dim_weights": dim_weights or {},
        "proportionality": proportionality or {},
    }


def apply_pending_preset(state: Any = None) -> None:
    state = _session() if state is None else state
    pending = state.pop(PENDING_PRESET_KEY, None)
    if not pending:
        return
    config = state["config"]
    key_map = preset_key_map()
    for attr, value in pending["weights"].items():
        setattr(config, attr, value)
        if attr in key_map:
            state[key_map[attr]] = int(min(max(value, 0), 100))
    for dim_name, value in pending.get("dim_weights", {}).items():
        if dim_name in (config.dimensions_config or {}):
            config.dimensions_config[dim_name]["weight"] = float(value)
            config.dimensions_config[dim_name]["enabled"] = value > 0
            state[f"dim_weight_{dim_name}"] = int(value)
    for attr, p_cfg in pending.get("proportionality", {}).items():
        p_enabled = bool(p_cfg.get("enabled", False))
        p_weight = int(p_cfg.get("weight", 50))
        if not config.proportionality_config:
            config.proportionality_config = {}
        config.proportionality_config[attr] = {"enabled": p_enabled, "weight": float(p_weight)}
        state[f"prop_enabled_{attr}"] = p_enabled
        state[f"prop_weight_{attr}"] = p_weight
