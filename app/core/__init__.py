from importlib import import_module
from typing import Any

_EXPORTS = {
    "Student": ".models",
    "Team": ".models",
    "Project": ".models",
    "Configuration": ".models",
    "DataLoader": ".data_loader",
    "DiversityMetrics": ".diversity_metrics",
}

__all__ = [
    "Student",
    "Team",
    "Project",
    "Configuration",
    "DataLoader",
    "DiversityMetrics",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
