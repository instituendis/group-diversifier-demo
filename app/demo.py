from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from app import hosting

__all__ = [
    "BANNER_TEXT",
    "DEMO_COHORT",
    "DEMO_DIR",
    "DEMO_NUM_PROJECTS",
    "DEMO_POLICY",
    "DEMO_PRIOR_PROJECTS",
    "DEMO_PROFILE",
    "DEMO_SAMPLE_FILES",
    "DEMO_TEAM_SIZES",
    "ENV_DEMO",
    "MARKER_FILE",
    "ROOT_DIR",
    "classic_hidden",
    "classic_unavailable_where",
    "demo_mode",
    "DEMO_SOURCE_URL",
    "source_url",
    "profile_name",
]

ROOT_DIR = Path(__file__).resolve().parent.parent

ENV_DEMO = "GROUP_DIVERSIFIER_DEMO"
MARKER_FILE = "DEMO_MODE"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

DEMO_DIR = ROOT_DIR / "data" / "demo"
DEMO_COHORT = DEMO_DIR / "demo_cohort.csv"
DEMO_POLICY = ROOT_DIR / "config" / "policies" / "demo_fun.json"

DEMO_PRIOR_PROJECTS = DEMO_DIR / "demo_prior_projects.csv"

DEMO_PROFILE = "demo"
DEFAULT_PROFILE = "default"

DEMO_TEAM_SIZES: Tuple[int, int, int] = (3, 5, 4)
DEMO_NUM_PROJECTS = 3

DEMO_SAMPLE_FILES: Dict[str, Tuple[str, str]] = {
    "cohort": ("demo_cohort.csv", "a fictional roster of 100 people with playful columns "
                                  "(colour, ice cream, city, hobbies, pet, coffees)"),
    "earlier_groups": ("demo_prior_projects.csv", "one earlier project: the 100 people in 25 teams of 4"),
    "participants": ("demo_participants.csv",
                     "who takes part in projects 2 and 3: people 97-100 sit out project 2"),
    "preferences": ("demo_preferences.csv",
                    "prefer/avoid requests; several ids are separated by |"),
    "reference_groups": ("demo_reference_groups.csv",
                         "the earlier project, then projects 2 and 3 split by hand"),
}

BANNER_TEXT = "🎈 Demo mode — the bundled sample data is fictional"


def demo_mode() -> bool:
    raw = os.environ.get(ENV_DEMO, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return (ROOT_DIR / MARKER_FILE).is_file()


def classic_hidden() -> bool:
    return hosting.hosted() and not demo_mode()


def classic_unavailable_where() -> str:
    return "this hosted server"


def profile_name() -> str:
    return DEMO_PROFILE if demo_mode() else DEFAULT_PROFILE


DEMO_SOURCE_URL = "https://github.com/instituendis/group-diversifier-demo"


def source_url() -> str:
    if hosting._http_url(hosting._env(hosting.ENV_SOURCE_URL)) is None and demo_mode():
        return DEMO_SOURCE_URL
    return hosting.source_url()
