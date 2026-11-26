from __future__ import annotations

from pathlib import Path
from typing import Any


def installed() -> bool:
    return False


def active() -> bool:
    return False


def hook(name: str, default: Any = None) -> Any:
    return default


def text(key: str, default: str) -> str:
    return default


def without_inactive_settings(config: Any) -> Any:
    return config


def without_inactive_metrics(scores: Any) -> Any:
    return scores


def preset_dirs(config_dir: Any) -> list:
    return [Path(config_dir) / "presets"]


def example_policy_dirs(config_dir: Any) -> list:
    return [Path(config_dir) / "policies"]


def _reset_for_tests() -> None: ...
