from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.engine_session.summaries import section_reunions, show_section_reunions
from app import extensions
from .fmt import fmt, fmt_pct

QUALITY_HELP = ("How much lower the soft penalty is than the average random allocation with "
                "the same team sizes (100% = no imbalance at all, 0% = random level). Hard "
                "rules are left out so a single broken cap cannot hide everything else.")
HARD_HELP = "Units of hard-rule violation (each costs 1000 points); 0 = every hard rule holds."

SECTION_MATES_HELP = (
    extensions.text('headline.section_mates_help', "Teammates whose only earlier contact was a group larger than the history's ignore_groups_larger_than (e.g. a large section): not repeats and not capped, only a weak soft contact. Set history.ignore_groups_larger_than to null (app: Allocation Rules › History & preferences › 'Treat large imported groups as teams') to keep them apart too."))


@dataclass(frozen=True)
class HeadlineRow:
    key: str
    label: str
    value: Any
    text: str
    tone: Optional[str] = None
    help: Optional[str] = None


def repeat_cap_text(repeats: int, over_cap: int) -> str:
    repeats, over_cap = int(repeats or 0), int(over_cap or 0)
    return f"{repeats} ({over_cap} over cap)" if over_cap else str(repeats)


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def headline_rows(summary: Optional[Dict[str, Any]],
                  reports: Optional[Dict[str, Dict[str, Any]]] = None) -> List[HeadlineRow]:
    summary = summary or {}
    quality = summary.get("quality_index_soft")
    hard = summary.get("hard_units")
    repeats = summary.get("repeated_pairs")
    over_cap = summary.get("over_cap_pairs")
    worst = summary.get("worst_team_penalty")
    rows = [
        HeadlineRow("quality", "Quality vs random", quality, fmt_pct(quality),
                    "good" if _positive(quality) else None, QUALITY_HELP),
        HeadlineRow("hard", "Hard violations", hard, fmt(hard, digits=0),
                    "bad" if _positive(hard) else None, HARD_HELP),
        HeadlineRow("repeats", "Repeat pairs", repeats, repeat_cap_text(repeats, over_cap)),
        HeadlineRow("over_cap", "Pairs over the repeat cap", over_cap, fmt(over_cap, digits=0)),
        HeadlineRow("worst", "Worst team penalty", worst, fmt(worst)),
    ]
    if show_section_reunions(summary, reports):
        reunions = section_reunions(summary, reports)
        rows.append(HeadlineRow("section_mates", "Section-mates reunited", reunions,
                                fmt(reunions, digits=0), None, SECTION_MATES_HELP))
    return rows
