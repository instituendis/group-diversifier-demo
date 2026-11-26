from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .student_fields import (parse_categorical_field, parse_int, parse_list_field,
                             parse_multi_select_field, parse_number,
                             parse_ordinal_weight, text_value)

ALGORITHMS: Tuple[str, ...] = ("engine", "greedy", "simulated_annealing", "genetic")
CLASSIC_ALGORITHMS: Tuple[str, ...] = ("greedy", "simulated_annealing", "genetic")
DEFAULT_ALGORITHM = "engine"


@dataclass
class BaseStudent:
    student_id: str
    prefer_teammates: List[str] = field(default_factory=list)
    avoid_teammates: List[str] = field(default_factory=list)
    group_number: Optional[int] = None
    previous_teams: Dict[str, str] = field(default_factory=dict)

    dynamic_categorical: Dict[str, str] = field(default_factory=dict)

    dynamic_ordinal: Dict[str, int] = field(default_factory=dict)

    dynamic_numeric: Dict[str, float] = field(default_factory=dict)

    dynamic_multi_select: Dict[str, List[str]] = field(default_factory=dict)

    def typed_attributes(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def _parse_typed(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def _typed_head_dict(self) -> Dict[str, Any]:
        return {}

    def _typed_tail_dict(self) -> Dict[str, Any]:
        return {}

    def _parse_derived(self, data: Dict[str, Any]) -> None: ...

    def _typed_attribute(self, attribute: str) -> Any:
        return getattr(self, attribute, None)

    def get_attribute_value(self, attribute: str) -> Any:
        if attribute in self.dynamic_categorical:
            return self.dynamic_categorical[attribute]
        elif attribute in self.dynamic_ordinal:
            return self.dynamic_ordinal[attribute]
        elif attribute in self.dynamic_numeric:
            return self.dynamic_numeric[attribute]
        elif attribute in self.dynamic_multi_select:
            return self.dynamic_multi_select[attribute]
        return self._typed_attribute(attribute)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "student_id": self.student_id,
            "prefer_teammates": "|".join(self.prefer_teammates),
            "avoid_teammates": "|".join(self.avoid_teammates),
            "group_number": self.group_number,
        }
        result.update(self._typed_head_dict())
        result.update({f"team_project_{k}": v for k, v in self.previous_teams.items()})
        result.update(self._typed_tail_dict())

        for mapping in (self.dynamic_categorical, self.dynamic_ordinal, self.dynamic_numeric):
            for key, value in mapping.items():
                result.setdefault(key, value)
        for key, values in self.dynamic_multi_select.items():
            result.setdefault(key, "|".join(values))

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  profile_config: Optional[Dict[str, Any]] = None):
        def text(key: str) -> Optional[str]:
            return text_value(data.get(key))

        previous_teams = {}
        for key, value in data.items():
            if str(key).startswith("team_project_") and text(key):
                previous_teams[str(key)[len("team_project_"):]] = text(key)

        student = cls(
            student_id=text("student_id") or "",
            prefer_teammates=parse_list_field(data.get("prefer_teammates")),
            avoid_teammates=parse_list_field(data.get("avoid_teammates")),
            group_number=parse_int(data.get("group_number")),
            previous_teams=previous_teams,
            **cls._parse_typed(data),
        )
        student._parse_derived(data)

        for dim_name, dim_config in (profile_config or {}).items():
            if dim_name not in data:
                continue
            value = data[dim_name]
            dim_type = dim_config.get("type")
            if dim_type == "categorical":
                parsed = parse_categorical_field(value, dim_config.get("categories") or [])
                if parsed is not None:
                    student.dynamic_categorical[dim_name] = parsed
            elif dim_type == "ordinal":
                weight = parse_ordinal_weight(value, dim_config.get("levels") or [])
                if weight is not None:
                    student.dynamic_ordinal[dim_name] = weight
            elif dim_type == "numeric":
                numeric_val = parse_number(value, decimal_comma=False)
                if numeric_val is not None:
                    student.dynamic_numeric[dim_name] = numeric_val
            elif dim_type == "multi_select":
                parsed_list = parse_multi_select_field(value, dim_config.get("max_selections"))
                if parsed_list:
                    student.dynamic_multi_select[dim_name] = parsed_list

        return student


_BASE_DICT_KEYS: Tuple[str, ...] = (
    "team_size_min", "team_size_max", "team_size_target", "num_teams", "num_projects",
    "per_project_team_sizes",
    "avoid_is_hard_constraint", "avoid_soft_penalty", "prefer_bonus",
    "max_repeat_pairings", "overlap_penalty", "max_repeat_is_hard",
    "algorithm", "num_cores", "random_seed",
    "sa_initial_temp", "sa_cooling_rate", "sa_iterations", "sa_auto_temperature",
    "ga_population_size", "ga_generations", "ga_mutation_rate",
    "local_search",
    "profile_name", "profile_version", "dimensions_config",
    "proportionality_config", "policy", "custom_profiles",
)


@dataclass
class BaseConfiguration:
    team_size_min: int = 3
    team_size_max: int = 8
    team_size_target: int = 5
    num_teams: int = 0
    num_projects: int = 1

    per_project_team_sizes: List[Dict[str, int]] = field(default_factory=list)

    def get_team_sizes_for_project(self, project_idx: int) -> Tuple[int, int, int]:
        override = None
        if 0 <= project_idx < len(self.per_project_team_sizes):
            override = self.per_project_team_sizes[project_idx]
        if not override:
            return self.team_size_min, self.team_size_max, self.team_size_target
        return (
            int(override.get("min", self.team_size_min)),
            int(override.get("max", self.team_size_max)),
            int(override.get("target", self.team_size_target)),
        )

    def effective_config_for_project(self, project_idx: int) -> "BaseConfiguration":
        min_size, max_size, target = self.get_team_sizes_for_project(project_idx)
        override = (
            self.per_project_team_sizes[project_idx]
            if 0 <= project_idx < len(self.per_project_team_sizes) else {}
        ) or {}
        if override:
            num_teams = int(override.get("num_teams", 0) or 0)
        else:
            num_teams = int(self.num_teams or 0)
        return replace(
            self,
            team_size_min=min_size,
            team_size_max=max_size,
            team_size_target=target,
            num_teams=num_teams,
        )

    avoid_is_hard_constraint: bool = True
    avoid_soft_penalty: float = 100.0
    prefer_bonus: float = 10.0
    max_repeat_pairings: int = 1
    overlap_penalty: float = 50.0

    max_repeat_is_hard: bool = True

    algorithm: str = DEFAULT_ALGORITHM
    num_cores: int = 1
    random_seed: Optional[int] = 42
    sa_initial_temp: float = 1000.0
    sa_cooling_rate: float = 0.99
    sa_iterations: int = 3000
    sa_auto_temperature: bool = True
    ga_population_size: int = 50
    ga_generations: int = 150
    ga_mutation_rate: float = 0.15

    local_search: bool = True

    profile_name: str = "default"
    profile_version: str = "1.0"
    dimensions_config: Dict[str, Any] = field(default_factory=dict)
    custom_profiles: List[str] = field(default_factory=list)
    proportionality_config: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)

    _DICT_KEYS = _BASE_DICT_KEYS

    def _dict_value(self, key: str) -> Any:
        if key == "per_project_team_sizes":
            return [dict(s) for s in self.per_project_team_sizes]
        if key == "proportionality_config":
            return {k: dict(v) for k, v in self.proportionality_config.items()}
        if key == "policy":
            return json.loads(json.dumps(self.policy)) if self.policy else {}
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {key: self._dict_value(key) for key in self._DICT_KEYS}

    def is_dimension_enabled(self, dimension_name: str) -> bool:
        if dimension_name in self.dimensions_config:
            return self.dimensions_config[dimension_name].get("enabled", True)
        return True

    def get_dimension_weight(self, dimension_name: str) -> float:
        if dimension_name in self.dimensions_config:
            return self.dimensions_config[dimension_name].get("weight", 50.0)
        return 50.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        valid_keys = cls.__dataclass_fields__.keys()
        payload = {k: v for k, v in data.items() if k in valid_keys}

        sizes = payload.get("per_project_team_sizes")
        if sizes is not None:
            payload["per_project_team_sizes"] = [
                dict(entry) for entry in sizes if entry
            ]

        policy = payload.get("policy")
        if policy is not None:
            payload["policy"] = dict(policy) if isinstance(policy, dict) else {}

        for nested in ("dimensions_config", "proportionality_config"):
            value = payload.get(nested)
            if value is not None:
                payload[nested] = {
                    key: (dict(v) if isinstance(v, dict) else v)
                    for key, v in dict(value).items()
                }
        dims = payload.get("dimensions_config")
        if dims:
            payload["dimensions_config"] = {
                key: v for key, v in dims.items()
                if isinstance(v, dict) and v.get("type")
            }

        if "random_seed" in payload and payload["random_seed"] is not None:
            try:
                payload["random_seed"] = int(payload["random_seed"])
            except (TypeError, ValueError):
                payload["random_seed"] = 42

        return cls(**payload)
