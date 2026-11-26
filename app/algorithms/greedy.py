from __future__ import annotations

import concurrent.futures
import random
import time
from functools import partial
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..constraints.base import ConstructionEvaluator
from .base import (
    FormationCancelled,
    AlgorithmResult,
    TeamBuilder,
    TeamFormationStrategy,
    derive_seed,
    input_fingerprint,
    plan_team_sizes,
)
from .pool_utils import get_process_executor, handle_pool_failure, max_workers

if TYPE_CHECKING:
    from ..core.models import Student, Team, Configuration
    from ..constraints.base import ConstraintEvaluator


GREEDY_SINGLE_CORE_SEED = 42


def _run_greedy_restart(
    restart_seed: int,
    students: List["Student"],
    config: "Configuration",
    evaluator: "ConstraintEvaluator",
    pair_counts: Dict[tuple, int],
    randomize: bool
) -> AlgorithmResult:
    strategy = GreedyStrategy(randomize_start=randomize)
    return strategy._run_single_greedy(
        students, config, evaluator, pair_counts, rng=random.Random(restart_seed)
    )


class GreedyStrategy(TeamFormationStrategy):
    MAX_CANDIDATES_TO_EVALUATE = 25

    def __init__(self, randomize_start: bool = True):
        super().__init__("Greedy")
        self.randomize_start = randomize_start
        self._run_counter = 0

    def _next_seed(self, config: "Configuration", students: List["Student"],
                   pair_counts: Dict[tuple, int]) -> int:
        base = getattr(config, "random_seed", GREEDY_SINGLE_CORE_SEED)
        seed = derive_seed(base, "greedy", self._run_counter,
                           input_fingerprint(students, pair_counts))
        self._run_counter += 1
        return seed

    def form_teams(
        self,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        existing_pair_counts: Dict[tuple, int] = None,
        **kwargs
    ) -> AlgorithmResult:
        start_time = time.time()
        pair_counts = dict(existing_pair_counts or {})
        num_cores = int(kwargs.get("num_cores", config.num_cores) or 1)
        seed = self._next_seed(config, students, pair_counts)

        if num_cores > 1 and self.randomize_start and len(students) > 1:
            return self._parallel_restarts(
                students, config, evaluator, pair_counts, num_cores, start_time, seed
            )

        result = self._run_single_greedy(
            students, config, evaluator, pair_counts, rng=random.Random(seed)
        )
        result.execution_time = time.time() - start_time
        result.stats["seed"] = seed
        return result

    def _parallel_restarts(
        self,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int],
        num_cores: int,
        start_time: float,
        seed: int,
    ) -> AlgorithmResult:
        callback = self.progress_callback
        n_restarts = num_cores
        workers = max_workers(num_cores)
        seeds = [seed] + [derive_seed(seed, "restart", i) for i in range(1, n_restarts)]
        error: Optional[str] = None
        best_result: Optional[AlgorithmResult] = None
        scores: List[float] = []
        futures: Dict[Any, int] = {}

        try:
            executor = get_process_executor(workers)
            restart_func = partial(
                _run_greedy_restart,
                students=students,
                config=config,
                evaluator=evaluator,
                pair_counts=dict(pair_counts),
                randomize=self.randomize_start,
            )
            futures = {executor.submit(restart_func, s): i for i, s in enumerate(seeds)}
            results: List[Optional[AlgorithmResult]] = [None] * n_restarts
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                results[futures[future]] = future.result()
                current = results[futures[future]]
                if self.better(best_result, current):
                    best_result = current
                if callback:
                    callback.update(done, n_restarts, best_result.soft_score)
            best_result = None
            for result in results:
                if self.better(best_result, result):
                    best_result = result
            scores = [r.score for r in results]
        except FormationCancelled:
            for future in futures:
                future.cancel()
            raise
        except Exception as exc:
            handle_pool_failure(workers, exc)
            error = f"{type(exc).__name__}: {exc}"
            best_result = self._run_single_greedy(
                students, config, evaluator, pair_counts, rng=random.Random(seed)
            )

        success = error is None
        best_result.execution_time = time.time() - start_time
        best_result.stats.update({
            "seed": seed,
            "multiprocessing_successful": success,
            "actual_cores_used": workers if success else 1,
            "requested_cores": num_cores,
            "parallel_restarts": n_restarts if success else 1,
        })
        if success:
            best_result.stats["restart_scores"] = scores
        else:
            best_result.stats["multiprocessing_error"] = error
        return best_result

    def _run_single_greedy(
        self,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int],
        rng: Optional[random.Random] = None,
    ) -> AlgorithmResult:
        start_time = time.time()
        if rng is None:
            rng = random.Random(derive_seed(
                getattr(config, "random_seed", GREEDY_SINGLE_CORE_SEED), "greedy-direct"))
        base_pair_counts = dict(pair_counts or {})

        plan = plan_team_sizes(len(students), config)
        team_sizes = plan.sizes

        construction_evaluator = ConstructionEvaluator(evaluator)
        fallback_fills = 0
        unassigned = list(students)
        if self.randomize_start:
            rng.shuffle(unassigned)

        teams: List["Team"] = []
        total_iterations = sum(team_sizes)
        current_iteration = 0

        for team_idx, team_size in enumerate(team_sizes):
            builder = TeamBuilder(f"team_{team_idx + 1}")

            if unassigned:
                builder.add_member(unassigned.pop(0))
                current_iteration += 1
                self.report_progress(current_iteration, total_iterations, 0)

            while builder.size < team_size and unassigned:
                if len(unassigned) <= self.MAX_CANDIDATES_TO_EVALUATE:
                    candidates = list(unassigned)
                else:
                    candidates = rng.sample(unassigned, self.MAX_CANDIDATES_TO_EVALUATE)
                remaining_pool = list(builder.members) + unassigned
                best_student, best_key = self._best_candidate(
                    builder, candidates, construction_evaluator,
                    base_pair_counts, remaining_pool,
                )
                if best_key is not None and best_key[0] > 0 and len(candidates) < len(unassigned):
                    best_student, best_key = self._best_candidate(
                        builder, unassigned, construction_evaluator,
                        base_pair_counts, remaining_pool,
                    )
                    fallback_fills += 1
                builder.add_member(best_student)
                unassigned.remove(best_student)
                current_iteration += 1
                self.report_progress(current_iteration, total_iterations,
                                     -best_key[1] if best_key else 0.0)

            teams.append(builder.build())

        for student in list(unassigned):
            target = min(
                teams,
                key=lambda t: (
                    t.size >= config.team_size_max,
                    evaluator.hard_violation_units(t, pair_counts=base_pair_counts),
                    t.size,
                ),
            )
            target.add_member(student)
            unassigned.remove(student)

        swap_stats = {"swaps": 0, "evaluations": 0, "passes": 0}
        if getattr(config, "local_search", True) and len(teams) > 1:
            swap_stats = self.local_search_swaps(
                teams, evaluator, base_pair_counts,
                max_evaluations=20000, max_passes=8,
            )

        return self.build_result(
            teams, evaluator, base_pair_counts,
            execution_time=time.time() - start_time,
            iterations=current_iteration,
            converged=True,
            stats={
                "arbitrary_fills": fallback_fills,
                "fallback_fills": fallback_fills,
                "local_search": swap_stats,
                "size_notes": plan.notes,
                "multiprocessing_successful": False,
                "actual_cores_used": 1,
                "requested_cores": 1,
            },
        )

    def _best_candidate(self, builder: TeamBuilder, candidates, evaluator,
                        pair_counts, remaining_pool):
        best_student, best_key = None, None
        kwargs = {"pair_counts": pair_counts, "remaining_students": remaining_pool}
        for candidate in candidates:
            builder.add_member(candidate)
            key = evaluator.team_key(builder.build(), **kwargs)
            builder.remove_member(candidate.student_id)
            if best_key is None or key < best_key:
                best_student, best_key = candidate, key
        return best_student, best_key


class RandomizedGreedyStrategy(TeamFormationStrategy):
    def __init__(self, n_restarts: int = 10):
        super().__init__("Randomized Greedy")
        self.n_restarts = n_restarts
        self.greedy = GreedyStrategy(randomize_start=True)

    def form_teams(
        self,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        existing_pair_counts: Dict[tuple, int] = None,
        **kwargs
    ) -> AlgorithmResult:
        start_time = time.time()
        kwargs.setdefault("num_cores", 1)

        best_result: Optional[AlgorithmResult] = None
        all_scores = []
        for i in range(self.n_restarts):
            result = self.greedy.form_teams(
                students, config, evaluator, existing_pair_counts, **kwargs
            )
            all_scores.append(result.score)
            if self.better(best_result, result):
                best_result = result
            self.report_progress(i + 1, self.n_restarts, best_result.soft_score)

        if best_result:
            finite = [s for s in all_scores if s != float('-inf')]
            best_result.execution_time = time.time() - start_time
            best_result.stats["n_restarts"] = self.n_restarts
            best_result.stats["all_scores"] = all_scores
            best_result.stats["score_variance"] = (
                sum((s - best_result.soft_score) ** 2 for s in finite) / len(finite)
                if finite else 0.0
            )

        return best_result
