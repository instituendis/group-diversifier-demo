from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Sequence, Tuple, TYPE_CHECKING
import hashlib
import random

if TYPE_CHECKING:
    from ..core.models import Student, Team, Configuration
    from ..constraints.base import ConstraintEvaluator

LEGACY_HARD_PENALTY = 1000.0

Key = Tuple[float, float]


@dataclass
class AlgorithmResult:
    teams: List["Team"]
    score: float
    execution_time: float
    iterations: int = 0
    converged: bool = True
    stats: Dict[str, Any] = field(default_factory=dict)
    history: List[float] = field(default_factory=list)
    hard_violations: float = 0.0
    soft_score: float = 0.0


@dataclass
class TeamSizePlan:
    sizes: List[int]
    notes: List[str] = field(default_factory=list)


def plan_team_sizes(n_students: int, config: "Configuration") -> TeamSizePlan:
    from ..engine.sizing import plan_team_sizes as engine_plan, size_problems

    notes: List[str] = []
    if n_students <= 0:
        return TeamSizePlan([], notes)
    size_min = max(1, int(config.team_size_min))
    size_max = max(size_min, int(config.team_size_max))
    target = max(1, int(config.team_size_target))
    requested = int(getattr(config, "num_teams", 0) or 0)
    sizes = engine_plan(n_students, size_min, size_max, target, requested)
    problems = size_problems(sizes, size_min, size_max)
    if problems:
        cause = (f"Requested {requested} teams for {n_students} students"
                 if requested > 0 else
                 f"{n_students} students cannot be split into teams of {size_min}-{size_max}")
        notes.append(f"{cause}: using {len(sizes)} teams; " + "; ".join(problems) + ".")
    return TeamSizePlan(sizes, notes)


def input_fingerprint(students: Sequence["Student"],
                      pair_counts: Optional[Dict[tuple, int]] = None,
                      *extra: Any) -> str:
    digest = hashlib.sha256()
    for sid in sorted(str(s.student_id) for s in students):
        digest.update(sid.encode("utf-8"))
        digest.update(b"\x00")
    for pair, count in sorted((pair_counts or {}).items()):
        digest.update(f"{pair[0]}|{pair[1]}|{count};".encode("utf-8"))
    for item in extra:
        digest.update(repr(item).encode("utf-8"))
    return digest.hexdigest()


def derive_seed(base_seed: Optional[int], *salts: Any) -> int:
    if base_seed is None:
        return random.SystemRandom().randrange(2 ** 31)
    text = "|".join([str(int(base_seed))] + [str(s) for s in salts])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def penalised_value(key: Key) -> float:
    return -key[1] - LEGACY_HARD_PENALTY * key[0]


class FormationCancelled(Exception): ...


class ProgressCallback:
    def __init__(self, callback: Optional[Callable[[int, int, float], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None):
        self.callback = callback
        self.cancel = cancel
        self.current_iteration = 0
        self.total_iterations = 0
        self.current_score = 0.0

    @property
    def cancelled(self) -> bool:
        return bool(self.cancel is not None and self.cancel())

    def check(self) -> None:
        if self.cancelled:
            raise FormationCancelled("Team formation was cancelled")

    def update(self, iteration: int, total: int, score: float) -> None:
        self.check()
        self.current_iteration = iteration
        self.total_iterations = total
        self.current_score = score

        if self.callback:
            self.callback(iteration, total, score)

    def get_progress(self) -> float:
        if self.total_iterations == 0:
            return 0.0
        return self.current_iteration / self.total_iterations


class TeamFormationStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.progress_callback: Optional[ProgressCallback] = None
    
    def set_progress_callback(self, callback: ProgressCallback) -> None:
        self.progress_callback = callback
    
    def report_progress(self, iteration: int, total: int, score: float) -> None:
        if self.progress_callback:
            self.progress_callback.update(iteration, total, score)

    def check_cancelled(self) -> None:
        if self.progress_callback is not None:
            self.progress_callback.check()

    @abstractmethod
    def form_teams(
        self,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        existing_pair_counts: Dict[tuple, int] = None,
        **kwargs
    ) -> AlgorithmResult:
        pass
    
    def team_key(
        self,
        team: "Team",
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int] = None,
        **kwargs,
    ) -> Key:
        kwargs["pair_counts"] = pair_counts or {}
        key_fn = getattr(evaluator, "team_key", None)
        if key_fn is not None:
            return key_fn(team, **kwargs)
        return (
            float(evaluator.hard_violation_count(team, **kwargs)),
            -evaluator.calculate_soft_score(team, **kwargs),
        )

    def solution_key(
        self,
        teams: Sequence["Team"],
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int] = None,
    ) -> Key:
        units = 0.0
        neg_soft = 0.0
        for team in teams:
            u, s = self.team_key(team, evaluator, pair_counts)
            units += u
            neg_soft += s
        return units, neg_soft

    def calculate_team_score(
        self,
        team: "Team",
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int] = None
    ) -> float:
        units, neg_soft = self.team_key(team, evaluator, pair_counts)
        return float('-inf') if units > 0 else -neg_soft

    def calculate_solution_score(
        self,
        teams: List["Team"],
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int] = None
    ) -> float:
        units, neg_soft = self.solution_key(teams, evaluator, pair_counts)
        return float('-inf') if units > 0 else -neg_soft

    @staticmethod
    def score_from_key(key: Key) -> float:
        return float('-inf') if key[0] > 0 else -key[1]

    def local_search_swaps(
        self,
        teams: List["Team"],
        evaluator: "ConstraintEvaluator",
        base_pair_counts: Dict[tuple, int] = None,
        max_evaluations: int = 20000,
        max_passes: int = 8,
    ) -> Dict[str, int]:
        if len(teams) < 2:
            return {"swaps": 0, "evaluations": 0, "passes": 0, "repair_swaps": 0}

        pair_counts = base_pair_counts or {}

        def key_of(team: "Team") -> Key:
            return self.team_key(team, evaluator, pair_counts)

        ordered = sorted(teams, key=lambda t: t.team_id)
        keys = {id(team): key_of(team) for team in ordered}

        attempts = [0]

        def try_swap(team_a: "Team", student_a, team_b: "Team", student_b) -> bool:
            attempts[0] += 1
            if attempts[0] % 256 == 0:
                self.check_cancelled()
            team_a.remove_member(student_a.student_id)
            team_b.remove_member(student_b.student_id)
            team_a.add_member(student_b)
            team_b.add_member(student_a)
            key_a = key_of(team_a)
            key_b = key_of(team_b)
            baseline_a = keys[id(team_a)]
            baseline_b = keys[id(team_b)]
            new_units = key_a[0] + key_b[0]
            old_units = baseline_a[0] + baseline_b[0]
            if new_units < old_units or (
                new_units == old_units
                and key_a[1] + key_b[1] < baseline_a[1] + baseline_b[1] - 1e-9
            ):
                keys[id(team_a)] = key_a
                keys[id(team_b)] = key_b
                return True
            team_a.remove_member(student_b.student_id)
            team_b.remove_member(student_a.student_id)
            team_a.add_member(student_a)
            team_b.add_member(student_b)
            return False

        repair_swaps = 0
        repair_evals = 0
        while repair_evals < max_evaluations and any(keys[id(t)][0] > 0 for t in ordered):
            progressed = False
            for team_a in [t for t in ordered if keys[id(t)][0] > 0]:
                for student_a in sorted(team_a.members, key=lambda s: s.student_id):
                    if keys[id(team_a)][0] <= 0 or repair_evals >= max_evaluations:
                        break
                    if not team_a.has_member(student_a.student_id):
                        continue
                    swapped = False
                    for team_b in ordered:
                        if team_b is team_a or swapped:
                            continue
                        for student_b in sorted(team_b.members, key=lambda s: s.student_id):
                            if repair_evals >= max_evaluations:
                                break
                            repair_evals += 1
                            if try_swap(team_a, student_a, team_b, student_b):
                                repair_swaps += 1
                                progressed = swapped = True
                                break
            if not progressed:
                break

        if repair_evals < max_evaluations and any(keys[id(t)][0] > 0 for t in ordered):
            gained, used = self._plateau_repair(ordered, keys, key_of,
                                                max_evaluations - repair_evals,
                                                check=self.check_cancelled)
            repair_swaps += gained
            repair_evals += used

        n_evals = 0
        n_swaps = 0
        passes_done = 0

        for pass_idx in range(max_passes):
            passes_done = pass_idx + 1
            improved = False
            for i in range(len(ordered)):
                if n_evals >= max_evaluations:
                    break
                team_a = ordered[i]
                for j in range(i + 1, len(ordered)):
                    if n_evals >= max_evaluations:
                        break
                    team_b = ordered[j]
                    members_a = sorted(team_a.members, key=lambda s: s.student_id)
                    members_b = sorted(team_b.members, key=lambda s: s.student_id)
                    for student_a in members_a:
                        if n_evals >= max_evaluations:
                            break
                        if not team_a.has_member(student_a.student_id):
                            continue
                        for student_b in members_b:
                            if n_evals >= max_evaluations:
                                break
                            if not team_b.has_member(student_b.student_id):
                                continue
                            n_evals += 1
                            if try_swap(team_a, student_a, team_b, student_b):
                                n_swaps += 1
                                improved = True
                                break
                if not improved and n_evals >= max_evaluations:
                    break
            if not improved:
                break

        return {"swaps": n_swaps + repair_swaps, "evaluations": n_evals + repair_evals,
                "passes": passes_done, "repair_swaps": repair_swaps}

    @staticmethod
    def _plateau_repair(ordered: List["Team"], keys: Dict[int, Key],
                        key_of: Callable[["Team"], Key], budget: int,
                        check: Optional[Callable[[], None]] = None) -> Tuple[int, int]:
        rng = random.Random(len(ordered) * 7919 + sum(t.size for t in ordered))
        total = sum(keys[id(t)][0] for t in ordered)
        best_total = total
        best_members = [list(t.members) for t in ordered]
        best_keys = dict(keys)
        accepted = 0
        used = 0
        while used < budget and best_total > 0:
            if check is not None and used % 256 == 0:
                check()
            violating = [t for t in ordered if keys[id(t)][0] > 0]
            if not violating:
                break
            team_a = rng.choice(violating)
            team_b = rng.choice([t for t in ordered if t is not team_a])
            if not team_a.members or not team_b.members:
                break
            student_a = rng.choice(team_a.members)
            student_b = rng.choice(team_b.members)
            team_a.remove_member(student_a.student_id)
            team_b.remove_member(student_b.student_id)
            team_a.add_member(student_b)
            team_b.add_member(student_a)
            used += 1
            key_a, key_b = key_of(team_a), key_of(team_b)
            delta = key_a[0] + key_b[0] - keys[id(team_a)][0] - keys[id(team_b)][0]
            if delta <= 0:
                keys[id(team_a)], keys[id(team_b)] = key_a, key_b
                total += delta
                accepted += 1
                if total < best_total:
                    best_total = total
                    best_members = [list(t.members) for t in ordered]
                    best_keys = dict(keys)
            else:
                team_a.remove_member(student_b.student_id)
                team_b.remove_member(student_a.student_id)
                team_a.add_member(student_a)
                team_b.add_member(student_b)
        for team, members in zip(ordered, best_members):
            team.members = members
            team._invalidate_cache()
        keys.clear()
        keys.update(best_keys)
        return accepted, used

    def build_result(
        self,
        teams: List["Team"],
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int],
        execution_time: float,
        iterations: int = 0,
        converged: bool = True,
        stats: Optional[Dict[str, Any]] = None,
        history: Optional[List[float]] = None,
    ) -> AlgorithmResult:
        key = self.solution_key(teams, evaluator, pair_counts)
        stats = dict(stats or {})
        stats.setdefault("n_teams", len(teams))
        stats.setdefault("team_sizes", [t.size for t in teams])
        stats["hard_violations"] = key[0]
        stats["soft_score"] = -key[1]
        history = list(history) if history is not None else [penalised_value(key)]
        stats.setdefault("history", history)
        return AlgorithmResult(
            teams=teams,
            score=self.score_from_key(key),
            execution_time=execution_time,
            iterations=iterations,
            converged=converged,
            stats=stats,
            history=history,
            hard_violations=key[0],
            soft_score=-key[1],
        )

    @staticmethod
    def better(result_a: Optional[AlgorithmResult], result_b: AlgorithmResult) -> bool:
        if result_a is None:
            return True
        return (result_b.hard_violations, -result_b.soft_score) < (
            result_a.hard_violations, -result_a.soft_score)

    def validate_solution(
        self,
        teams: List["Team"],
        students: List["Student"],
        evaluator: "ConstraintEvaluator"
    ) -> List[str]:
        issues = []

        assigned = set()
        for team in teams:
            for student in team.members:
                if student.student_id in assigned:
                    issues.append(f"Student {student.student_id} assigned multiple times")
                assigned.add(student.student_id)
        
        expected = {s.student_id for s in students}
        missing = expected - assigned
        extra = assigned - expected
        
        if missing:
            issues.append(f"Students not assigned: {missing}")
        if extra:
            issues.append(f"Unknown students assigned: {extra}")

        for team in teams:
            violations = evaluator.get_violations(team)
            issues.extend(violations)
        
        return issues


class TeamBuilder:
    def __init__(self, team_id: str, project_id: str = None):
        self.team_id = team_id
        self.project_id = project_id
        self.members: List["Student"] = []
    
    def add_member(self, student: "Student") -> "TeamBuilder":
        self.members.append(student)
        return self
    
    def remove_member(self, student_id: str) -> "TeamBuilder":
        self.members = [s for s in self.members if s.student_id != student_id]
        return self
    
    def clear(self) -> "TeamBuilder":
        self.members = []
        return self
    
    def build(self) -> "Team":
        from ..core.models import Team
        return Team(
            team_id=self.team_id,
            members=self.members.copy(),
            project_id=self.project_id
        )
    
    @property
    def size(self) -> int:
        return len(self.members)
    
    @property
    def member_ids(self) -> set:
        return {s.student_id for s in self.members}
