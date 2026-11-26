from __future__ import annotations

import json
import typing
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from app.engine.policy import (
    AttributeRule,
    CountRule,
    FairnessPolicy,
    HistoryPolicy,
    IsolationRule,
    NumericBoundsRule,
    PairRule,
    Policy,
    PreferencePolicy,
    SharedValueRule,
    SolverSettings,
)

_POLICY_CONTENT_KEYS: Tuple[str, ...] = tuple(f.name for f in fields(Policy)
                                              if f.name not in ("name", "description", "version"))

_RUNS_FORMAT = "group-diversifier-runs"

_POLICY_RULE_LISTS = {"attributes": AttributeRule, "counts": CountRule,
                      "isolation": IsolationRule, "numeric_bounds": NumericBoundsRule,
                      "shared": SharedValueRule, "pairs": PairRule}
_POLICY_SECTIONS = {"history": HistoryPolicy, "preferences": PreferencePolicy,
                    "fairness": FairnessPolicy, "solver": SolverSettings}

_CONFIG_EMPTY_TEXT = ("this configuration export has no allocation policy yet (set one up "
                      "on the Allocation Rules page, or use --preset)")


class PolicyFileError(ValueError):
    def __init__(self, kind: str, detail: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class PolicyFile:
    policy: Dict[str, Any]
    origin: str
    problems: List[str]


def _json_kind(value: Any) -> str:
    return {dict: "an object", list: "a list", str: "text", bool: "true/false",
            type(None): "null"}.get(type(value), "a number")


def _no_fields(policy: Dict[str, Any]) -> PolicyFileError:
    found = ", ".join(sorted(map(str, policy))[:6]) or "an empty object"
    return PolicyFileError(
        "fields", found,
        f"no policy fields found (expected some of {', '.join(_POLICY_CONTENT_KEYS)}; "
        f"found {found}). Pass a policy JSON, a CLI --report file or an app configuration "
        "export.")


def policy_envelope(data: Any) -> PolicyFile:
    if not isinstance(data, dict):
        raise PolicyFileError("object", _json_kind(data),
                              f"expected a JSON object with policy fields, got "
                              f"{_json_kind(data)}")
    if any(key in data for key in _POLICY_CONTENT_KEYS):
        return _checked(data, "")
    if "policy" in data:
        if not isinstance(data["policy"], dict):
            raise PolicyFileError("policy-object", _json_kind(data["policy"]),
                                  f"'policy' is {_json_kind(data['policy'])}, expected an "
                                  "object with policy fields")
        return _checked(data["policy"], "the rules stored in the file")
    if isinstance(data.get("configuration"), dict):
        nested = data["configuration"]
        if "policy" not in nested:
            raise PolicyFileError("config-missing", _json_kind(nested.get("policy")),
                                  _CONFIG_EMPTY_TEXT)
        if not nested["policy"]:
            raise PolicyFileError("config-empty", _json_kind(nested["policy"]),
                                  _CONFIG_EMPTY_TEXT)
        if not isinstance(nested["policy"], dict):
            raise PolicyFileError("config-object", _json_kind(nested["policy"]),
                                  f"'configuration.policy' is {_json_kind(nested['policy'])}, "
                                  "expected an object with policy fields")
        return _checked(nested["policy"], "the rules of the configuration export")
    if data.get("format") == _RUNS_FORMAT:
        raise PolicyFileError("runs", "",
                              "this is a run-history bundle, not a policy; download the "
                              "rules of one run on the Allocation Rules page, or pass a "
                              "CLI --report")
    raise _no_fields(data)


def _checked(policy: Dict[str, Any], origin: str) -> PolicyFile:
    if not any(key in policy for key in _POLICY_CONTENT_KEYS):
        raise _no_fields(policy)
    return PolicyFile(policy=policy, origin=origin, problems=policy_type_problems(policy))


def read_policy_text(text: str) -> PolicyFile:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyFileError("json", str(exc), f"not valid JSON ({exc})") from None
    return policy_envelope(data)


def read_policy_bytes(raw: bytes) -> PolicyFile:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PolicyFileError("json", str(exc), f"not valid JSON ({exc})") from None
    return read_policy_text(text)


def _type_mismatch(value: Any, hint: Any) -> Optional[str]:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union:
        if value is None and type(None) in args:
            return None
        others = [a for a in args if a is not type(None)]
        return _type_mismatch(value, others[0]) if len(others) == 1 else None
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is bool:
        return None if isinstance(value, bool) else "true or false"
    if hint is int:
        whole = is_number and float(value).is_integer()
        return None if whole else "a whole number"
    if hint is float:
        return None if is_number else "a number"
    if hint is str:
        return None if isinstance(value, str) else "text"
    if origin is list:
        if not isinstance(value, list):
            return "a list"
        item = args[0] if args else Any
        for element in value:
            if item is str and (isinstance(element, str) or
                                (isinstance(element, (int, float))
                                 and not isinstance(element, bool))):
                continue
            if _type_mismatch(element, item) is not None:
                return f"a list of {'text values' if item is str else item.__name__}"
        return None
    if origin is tuple:
        if not isinstance(value, (list, tuple)) or len(value) > len(args):
            return f"a list of at most {len(args)} values"
        for element, item in zip(value, args):
            if _type_mismatch(element, item) is not None:
                return "numbers or null"
        return None
    return None


def _is_text_list(hint: Any) -> bool:
    if typing.get_origin(hint) is typing.Union:
        others = [a for a in typing.get_args(hint) if a is not type(None)]
        return len(others) == 1 and _is_text_list(others[0])
    return typing.get_origin(hint) is list and typing.get_args(hint) == (str,)


def _as_text(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return str(int(value)) if float(value).is_integer() else str(value)


def numbers_as_text(data: Dict[str, Any]) -> Dict[str, Any]:
    def fix(obj: Any, cls: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        hints = typing.get_type_hints(cls)
        return {name: [_as_text(v) for v in value]
                if isinstance(value, list) and _is_text_list(hints.get(name)) else value
                for name, value in obj.items()}

    out = dict(data)
    for key, cls in _POLICY_RULE_LISTS.items():
        if isinstance(out.get(key), list):
            out[key] = [fix(rule, cls) for rule in out[key]]
    for key, cls in _POLICY_SECTIONS.items():
        if isinstance(out.get(key), dict):
            out[key] = fix(out[key], cls)
    return out


def policy_type_problems(data: Dict[str, Any]) -> List[str]:
    problems: List[str] = []

    def check(obj: Any, cls: Any, where: str) -> None:
        if not isinstance(obj, dict):
            problems.append(f"{where}: expected an object with fields, got {json.dumps(obj)}")
            return
        hints = typing.get_type_hints(cls)
        for name, value in obj.items():
            hint = hints.get(name)
            if hint is None:
                continue
            expected = _type_mismatch(value, hint)
            if expected:
                problems.append(f"{where}.{name}: expected {expected}, got {json.dumps(value)}")

    for key, cls in _POLICY_RULE_LISTS.items():
        rules = data.get(key)
        if rules is None:
            continue
        if not isinstance(rules, list):
            problems.append(f"{key}: expected a list of rules, got {json.dumps(rules)}")
            continue
        for k, rule in enumerate(rules):
            check(rule, cls, f"{key}[{k}]")
    for key, cls in _POLICY_SECTIONS.items():
        if data.get(key) is not None:
            check(data[key], cls, key)
    for name in ("name", "description"):
        if name in data and not isinstance(data[name], str):
            problems.append(f"{name}: expected text, got {json.dumps(data[name])}")
    if "version" in data and _type_mismatch(data["version"], int):
        problems.append(f"version: expected a whole number, got {json.dumps(data['version'])}")
    return problems
