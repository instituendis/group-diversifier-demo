from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

HARD_PENALTY = 1000.0

POLICY_VERSION = 1

CATEGORICAL_GOALS = ("balance", "diversify", "cluster", "ignore")
NUMERIC_GOALS = ("balance", "spread", "cluster", "ignore")
ATTRIBUTE_KINDS = ("categorical", "numeric", "multi")
COUNT_MODES = ("any_of", "each")
PAIR_KINDS = ("together", "apart")
CONTACT_MODELS = ("uniform", "team_size")
SOLVER_METHODS = ("anneal", "milp", "milp+anneal")


@dataclass
class AttributeRule:
    attribute: str
    goal: str = "balance"
    weight: float = 1.0
    enabled: bool = True
    kind: Optional[str] = None
    order: Optional[List[str]] = None
    missing_as_category: bool = False


@dataclass
class CountRule:
    attribute: str
    values: List[str] = field(default_factory=list)
    mode: str = "any_of"
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    numeric_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    hard: bool = False
    weight: float = 5.0
    enabled: bool = True
    label: str = ""


@dataclass
class IsolationRule:
    attribute: str
    values: List[str] = field(default_factory=list)
    min_together: int = 2
    hard: bool = False
    weight: float = 5.0
    enabled: bool = True
    label: str = ""


@dataclass
class NumericBoundsRule:
    attribute: str
    mean_min: Optional[float] = None
    mean_max: Optional[float] = None
    max_range: Optional[float] = None
    hard: bool = False
    weight: float = 3.0
    enabled: bool = True
    label: str = ""


@dataclass
class SharedValueRule:
    attribute: str
    min_shared: int = 1
    hard: bool = False
    weight: float = 5.0
    enabled: bool = True
    label: str = ""


@dataclass
class PairRule:
    kind: str
    students: List[str] = field(default_factory=list)
    hard: bool = True
    weight: float = 10.0
    enabled: bool = True
    label: str = ""
    projects: Optional[List[str]] = None


@dataclass
class HistoryPolicy:
    enabled: bool = True
    repeat_weight: float = 3.0
    max_times_together: Optional[int] = 1
    hard_cap: bool = True
    cap_weight: float = 20.0
    decay: float = 1.0
    contact_model: str = "team_size"
    reference_team_size: int = 5
    ignore_groups_larger_than: Optional[int] = 12


@dataclass
class PreferencePolicy:
    use_student_preferences: bool = True
    prefer_weight: float = 1.0
    avoid_hard: bool = True
    avoid_weight: float = 10.0


@dataclass
class FairnessPolicy:
    worst_team_weight: float = 0.25


@dataclass
class SolverSettings:
    method: str = "anneal"
    time_limit_s: float = 10.0
    restarts: int = 4
    seed: Optional[int] = 42
    iterations: Optional[int] = None
    flexible_sizes: bool = False
    polish: bool = True
    joint_projects: bool = False
    mip_gap: float = 0.01
    workers: int = 1
    scale_time_limit: bool = True


@dataclass
class Policy:
    name: str = "Custom"
    description: str = ""
    version: int = POLICY_VERSION
    attributes: List[AttributeRule] = field(default_factory=list)
    counts: List[CountRule] = field(default_factory=list)
    isolation: List[IsolationRule] = field(default_factory=list)
    numeric_bounds: List[NumericBoundsRule] = field(default_factory=list)
    shared: List[SharedValueRule] = field(default_factory=list)
    pairs: List[PairRule] = field(default_factory=list)
    history: HistoryPolicy = field(default_factory=HistoryPolicy)
    preferences: PreferencePolicy = field(default_factory=PreferencePolicy)
    fairness: FairnessPolicy = field(default_factory=FairnessPolicy)
    solver: SolverSettings = field(default_factory=SolverSettings)

    def attribute_rule(self, attribute: str) -> Optional[AttributeRule]:
        for rule in self.attributes:
            if rule.attribute == attribute:
                return rule
        return None

    def active_attributes(self) -> List[AttributeRule]:
        return [
            r for r in self.attributes
            if r.enabled and r.goal != "ignore" and r.weight > 0
        ]

    def referenced_attributes(self) -> List[str]:
        names: List[str] = []
        for rule in [*self.attributes, *self.counts, *self.isolation,
                     *self.numeric_bounds, *self.shared]:
            if rule.enabled and rule.attribute not in names:
                names.append(rule.attribute)
        return names

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for rule in data["counts"]:
            if rule["numeric_range"] is not None:
                rule["numeric_range"] = list(rule["numeric_range"])
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Policy":
        if not data:
            return cls()
        if isinstance(data, Policy):
            return data
        payload = _known(cls, data)
        payload["attributes"] = [
            AttributeRule(**_known(AttributeRule, r)) for r in data.get("attributes", []) or []
        ]
        counts = []
        for r in data.get("counts", []) or []:
            rule = _known(CountRule, r)
            rng = rule.get("numeric_range")
            if rng is not None:
                lo, hi = (list(rng) + [None, None])[:2]
                rule["numeric_range"] = (lo, hi)
            counts.append(CountRule(**rule))
        payload["counts"] = counts
        payload["isolation"] = [
            IsolationRule(**_known(IsolationRule, r)) for r in data.get("isolation", []) or []
        ]
        payload["numeric_bounds"] = [
            NumericBoundsRule(**_known(NumericBoundsRule, r))
            for r in data.get("numeric_bounds", []) or []
        ]
        payload["shared"] = [
            SharedValueRule(**_known(SharedValueRule, r)) for r in data.get("shared", []) or []
        ]
        payload["pairs"] = [
            PairRule(**_id_texts(_known(PairRule, r), ("students", "projects")))
            for r in data.get("pairs", []) or []
        ]
        payload["history"] = HistoryPolicy(**_known(HistoryPolicy, data.get("history") or {}))
        payload["preferences"] = PreferencePolicy(
            **_known(PreferencePolicy, data.get("preferences") or {})
        )
        payload["fairness"] = FairnessPolicy(**_known(FairnessPolicy, data.get("fairness") or {}))
        payload["solver"] = SolverSettings(**_known(SolverSettings, data.get("solver") or {}))
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> "Policy":
        return cls.from_dict(json.loads(text))

    def validate(self, attribute_kinds: Optional[Dict[str, str]] = None) -> List[str]:
        problems: List[str] = []
        kinds = attribute_kinds or {}

        def known(attr: str, where: str) -> Optional[str]:
            if kinds and attr not in kinds:
                problems.append(f"{where}: attribute '{attr}' is not in the roster")
                return None
            return kinds.get(attr)

        seen = set()
        for rule in self.attributes:
            where = f"Attribute rule '{rule.attribute}'"
            if rule.attribute in seen:
                problems.append(f"{where}: duplicate rule")
            seen.add(rule.attribute)
            if rule.weight < 0:
                problems.append(f"{where}: weight must be >= 0")
            if rule.kind is not None and rule.kind not in ATTRIBUTE_KINDS:
                problems.append(f"{where}: unknown kind '{rule.kind}'")
            kind = rule.kind
            if rule.enabled:
                kind = rule.kind or known(rule.attribute, where)
            goals = NUMERIC_GOALS if kind == "numeric" else CATEGORICAL_GOALS
            if rule.goal not in goals:
                problems.append(
                    f"{where}: goal '{rule.goal}' is not valid for a "
                    f"{kind or 'categorical'} attribute (use one of {', '.join(goals)})"
                )

        for rule in self.counts:
            where = f"Count rule '{rule.label or rule.attribute}'"
            if not rule.enabled:
                continue
            kind = known(rule.attribute, where)
            if rule.mode not in COUNT_MODES:
                problems.append(f"{where}: unknown mode '{rule.mode}'")
            if rule.min_count is None and rule.max_count is None:
                problems.append(f"{where}: set a minimum and/or a maximum")
            if rule.min_count is not None and rule.min_count < 0:
                problems.append(f"{where}: minimum must be >= 0")
            if rule.max_count is not None and rule.max_count < 0:
                problems.append(f"{where}: maximum must be >= 0")
            if (rule.min_count is not None and rule.max_count is not None
                    and rule.min_count > rule.max_count):
                problems.append(f"{where}: minimum exceeds maximum")
            if kind == "numeric" and rule.numeric_range is None:
                problems.append(f"{where}: numeric attributes need a numeric_range")
            if kind in ("categorical", "multi") and rule.numeric_range is not None:
                problems.append(f"{where}: numeric_range only applies to numeric attributes")
            if (rule.numeric_range is None and rule.mode == "any_of" and not rule.values):
                problems.append(f"{where}: list at least one value (or use mode 'each')")
            if rule.weight < 0:
                problems.append(f"{where}: weight must be >= 0")

        for rule in self.isolation:
            where = f"Isolation rule '{rule.label or rule.attribute}'"
            if not rule.enabled:
                continue
            kind = known(rule.attribute, where)
            if kind == "numeric":
                problems.append(f"{where}: isolation rules need a categorical attribute")
            if rule.min_together < 2:
                problems.append(f"{where}: min_together must be >= 2")
            if rule.weight < 0:
                problems.append(f"{where}: weight must be >= 0")

        for rule in self.numeric_bounds:
            where = f"Numeric bounds rule '{rule.label or rule.attribute}'"
            if not rule.enabled:
                continue
            kind = known(rule.attribute, where)
            if kind and kind != "numeric":
                problems.append(f"{where}: needs a numeric attribute")
            if rule.mean_min is None and rule.mean_max is None and rule.max_range is None:
                problems.append(f"{where}: set a mean bound and/or a maximum range")
            if (rule.mean_min is not None and rule.mean_max is not None
                    and rule.mean_min > rule.mean_max):
                problems.append(f"{where}: minimum mean exceeds maximum mean")
            if rule.max_range is not None and rule.max_range < 0:
                problems.append(f"{where}: maximum range must be >= 0")
            if rule.weight < 0:
                problems.append(f"{where}: weight must be >= 0")

        for rule in self.shared:
            where = f"Shared-value rule '{rule.label or rule.attribute}'"
            if not rule.enabled:
                continue
            kind = known(rule.attribute, where)
            if kind == "numeric":
                problems.append(f"{where}: needs a categorical or multi-value attribute")
            if rule.min_shared < 1:
                problems.append(f"{where}: min_shared must be >= 1")
            if rule.weight < 0:
                problems.append(f"{where}: weight must be >= 0")

        for rule in self.pairs:
            where = f"Pair rule '{rule.label or ', '.join(rule.students)}'"
            if not rule.enabled:
                continue
            if rule.kind not in PAIR_KINDS:
                problems.append(f"{where}: kind must be 'together' or 'apart'")
            if len(set(rule.students)) < 2:
                problems.append(f"{where}: needs at least two distinct students")

        h = self.history
        if h.contact_model not in CONTACT_MODELS:
            problems.append(f"History: unknown contact model '{h.contact_model}'")
        if h.max_times_together is not None and h.max_times_together < 1:
            problems.append("History: max_times_together must be >= 1 (or None for no cap)")
        if not 0.0 <= h.decay <= 1.0:
            problems.append("History: decay must be between 0 and 1")
        if h.reference_team_size < 2:
            problems.append("History: reference_team_size must be >= 2")

        if not 0.0 <= self.fairness.worst_team_weight <= 1.0:
            problems.append("Fairness: worst_team_weight must be between 0 and 1")

        s = self.solver
        if s.method not in SOLVER_METHODS:
            problems.append(f"Solver: unknown method '{s.method}'")
        if s.time_limit_s <= 0:
            problems.append("Solver: time limit must be positive")
        if s.restarts < 1:
            problems.append("Solver: restarts must be >= 1")
        if s.workers < 1:
            problems.append("Solver: workers must be >= 1")
        return problems


def _known(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in dict(data).items() if k in names}


def _id_text(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return str(int(value)) if float(value).is_integer() else str(value)


def _id_texts(rule: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    for key in keys:
        if isinstance(rule.get(key), (list, tuple)):
            rule[key] = [_id_text(v) for v in rule[key]]
    return rule
