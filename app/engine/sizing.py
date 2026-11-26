from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..core.models import Configuration


def project_config(config: "Configuration", project_idx: int) -> "Configuration":
    effective = config.effective_config_for_project(project_idx)
    if 0 <= project_idx < len(config.per_project_team_sizes):
        override = config.per_project_team_sizes[project_idx] or {}
        if override and not int(override.get("num_teams", 0) or 0):
            effective.num_teams = 0
    return effective


def team_count(n: int, size_min: int, size_max: int, target: int, num_teams: int = 0) -> int:
    if n <= 0:
        return 0
    if num_teams and num_teams > 0:
        return max(1, min(int(num_teams), n))
    size_min = max(1, int(size_min))
    size_max = max(size_min, int(size_max))
    target = min(max(int(target), size_min), size_max)
    ideal = max(1, round(n / target))
    lo = math.ceil(n / size_max)
    hi = n // size_min
    if lo <= hi:
        return min(max(ideal, lo), hi)
    return ideal


def balanced_sizes(n: int, teams: int) -> List[int]:
    if teams <= 0 or n <= 0:
        return []
    base, extra = divmod(n, teams)
    return [base + 1] * extra + [base] * (teams - extra)


def size_order_key(size: int, target: Optional[int]) -> Tuple[int, int]:
    return (abs(int(size) - int(target)), int(size)) if target is not None else (0, int(size))


def order_sizes_target_first(sizes: Sequence[int], target: Optional[int] = None) -> List[int]:
    sizes = [int(s) for s in sizes]
    if not sizes:
        return []
    if target is None or int(target) <= 0:
        counts = Counter(sizes)
        return sorted(sizes, key=lambda s: (-counts[s], s))
    return sorted(sizes, key=lambda s: size_order_key(s, target))


def plan_team_sizes(n: int, size_min: int, size_max: int, target: int,
                    num_teams: int = 0) -> List[int]:
    sizes = balanced_sizes(n, team_count(n, size_min, size_max, target, num_teams))
    return order_sizes_target_first(sizes, target)


def size_problems(sizes: List[int], size_min: int, size_max: int) -> List[str]:
    problems = []
    small = [s for s in sizes if s < size_min]
    large = [s for s in sizes if s > size_max]
    if small:
        problems.append(
            f"{len(small)} team(s) would have {_size_range(small)} member(s), "
            f"below the minimum of {size_min}"
        )
    if large:
        problems.append(
            f"{len(large)} team(s) would have {_size_range(large)} members, "
            f"above the maximum of {size_max}"
        )
    return problems


def _size_range(sizes: List[int]) -> str:
    lo, hi = min(sizes), max(sizes)
    if lo == hi:
        return str(lo)
    counts = ", ".join(f"{sizes.count(s)} x {s}" for s in sorted(set(sizes)))
    return f"{lo}-{hi} ({counts})"


def all_sizes_listed(sizes: List[int], size_min: int, size_max: int) -> bool:
    return bool(sizes) and all(s < size_min or s > size_max for s in sizes)


def size_bounds_for(sizes: List[int], size_min: int, size_max: int,
                    flexible: bool) -> Tuple[int, int]:
    if not sizes:
        return size_min, size_max
    if flexible:
        return min(size_min, min(sizes)), max(size_max, max(sizes))
    return min(sizes), max(sizes)


def describe_sizes(sizes: List[int]) -> str:
    if not sizes:
        return "no teams"
    counts = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    parts = ", ".join(f"{c} x {s}" for s, c in sorted(counts.items()))
    return f"{len(sizes)} teams: {parts}"


def resolve_sizes(n: int, size_min: int, size_max: int, target: int,
                  num_teams: int = 0, explicit: Optional[List[int]] = None) -> List[int]:
    if explicit and sum(explicit) == n and all(s > 0 for s in explicit):
        return list(explicit)
    return plan_team_sizes(n, size_min, size_max, target, num_teams)


_AUTO_MAX_ITERATIONS = 120_000


def auto_iterations(n: int, n_teams: int) -> int:
    if n < 2 or n_teams < 2:
        return 0
    return int(min(max(600 * n, 20_000), _AUTO_MAX_ITERATIONS))


def polish_evaluations(n: int) -> int:
    return int(max(30_000, 100 * n))
