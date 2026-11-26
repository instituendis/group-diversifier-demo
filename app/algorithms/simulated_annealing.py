from __future__ import annotations

import concurrent.futures
import math
import random
import statistics
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .base import (
    FormationCancelled,
    LEGACY_HARD_PENALTY,
    AlgorithmResult,
    TeamFormationStrategy,
    derive_seed,
    input_fingerprint,
    penalised_value,
    plan_team_sizes,
)
from .pool_utils import get_process_executor, handle_pool_failure, max_workers
from ..core.models import Team

if TYPE_CHECKING:
    from ..core.models import Student, Configuration
    from ..constraints.base import ConstraintEvaluator


SA_DEFAULT_SEED = 42


def _run_sa_chain(
    chain_seed: int,
    students: List["Student"],
    config: "Configuration",
    evaluator: "ConstraintEvaluator",
    pair_counts: Dict[tuple, int],
    initial_temp: float,
    cooling_rate: float,
    max_iterations: int,
    min_temp: float,
    strategy: "SimulatedAnnealingStrategy",
    auto_temperature: bool = True,
) -> AlgorithmResult:
    strategy.min_temp = min_temp
    return strategy._run_single_chain(
        students, config, evaluator, pair_counts,
        initial_temp, cooling_rate, max_iterations,
        rng=random.Random(chain_seed), auto_temperature=auto_temperature,
    )


class SimulatedAnnealingStrategy(TeamFormationStrategy):
    COLD_FRACTION = 0.01

    def __init__(
        self,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.1,
        max_iterations: int = 10000
    ):
        super().__init__("Simulated Annealing")
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations

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

        initial_temp = kwargs.get("initial_temp", config.sa_initial_temp)
        cooling_rate = kwargs.get("cooling_rate", config.sa_cooling_rate)
        max_iterations = int(kwargs.get("max_iterations", config.sa_iterations))
        num_cores = int(kwargs.get("num_cores", config.num_cores) or 1)
        auto_temperature = bool(kwargs.get(
            "auto_temperature", getattr(config, "sa_auto_temperature", True)))
        base = getattr(config, "random_seed", SA_DEFAULT_SEED)
        seed = derive_seed(base, "anneal", input_fingerprint(students, pair_counts))

        if num_cores > 1 and len(students) > 1:
            return self._parallel_chains(
                students, config, evaluator, pair_counts,
                initial_temp, cooling_rate, max_iterations, num_cores, start_time,
                seed, auto_temperature,
            )

        result = self._run_single_chain(
            students, config, evaluator, pair_counts,
            initial_temp, cooling_rate, max_iterations,
            rng=random.Random(seed), auto_temperature=auto_temperature,
        )
        result.execution_time = time.time() - start_time
        result.stats["seed"] = seed
        return result

    def _generate_initial_solution(
        self,
        students: List["Student"],
        config: "Configuration",
        rng: Optional[random.Random] = None,
    ) -> List[Team]:
        rng = rng or random.Random(SA_DEFAULT_SEED)
        shuffled = list(students)
        rng.shuffle(shuffled)
        teams = []
        position = 0
        for index, size in enumerate(plan_team_sizes(len(shuffled), config).sizes):
            teams.append(Team(team_id=f"team_{index + 1}",
                              members=list(shuffled[position:position + size])))
            position += size
        return teams

    def _copy_teams(self, teams: List[Team]) -> List[Team]:
        return [
            Team(team_id=t.team_id, members=list(t.members), project_id=t.project_id)
            for t in teams
        ]

    @staticmethod
    def _propose(teams: List[Team], rng: random.Random, size_min: int,
                 size_max: int) -> Optional[Tuple[str, int, int, int, int]]:
        if len(teams) < 2:
            return None
        a, b = rng.sample(range(len(teams)), 2)
        ta, tb = teams[a], teams[b]
        if ta.size != tb.size and rng.random() < 0.5:
            src, dst = (a, b) if ta.size > tb.size else (b, a)
            if teams[src].size - 1 >= max(1, size_min) and teams[dst].size + 1 <= size_max:
                return ("move", src, dst, rng.randrange(teams[src].size), -1)
        if not ta.members or not tb.members:
            return None
        return ("swap", a, b, rng.randrange(ta.size), rng.randrange(tb.size))

    @staticmethod
    def _apply(teams: List[Team], move) -> Tuple:
        kind, a, b, ia, ib = move
        if kind == "swap":
            sa, sb = teams[a].members[ia], teams[b].members[ib]
            teams[a].remove_member(sa.student_id)
            teams[b].remove_member(sb.student_id)
            teams[a].add_member(sb)
            teams[b].add_member(sa)
            return (kind, a, b, sa, sb)
        student = teams[a].members[ia]
        teams[a].remove_member(student.student_id)
        teams[b].add_member(student)
        return (kind, a, b, student, None)

    @staticmethod
    def _undo(teams: List[Team], undo) -> None:
        kind, a, b, sa, sb = undo
        if kind == "swap":
            teams[a].remove_member(sb.student_id)
            teams[b].remove_member(sa.student_id)
            teams[a].add_member(sa)
            teams[b].add_member(sb)
        else:
            teams[b].remove_member(sa.student_id)
            teams[a].add_member(sa)

    def _calibrate_temperature(self, teams, keys, evaluator, pair_counts, rng,
                               size_min, size_max, samples: int = 120) -> float:
        worsening: List[float] = []
        for _ in range(samples):
            move = self._propose(teams, rng, size_min, size_max)
            if move is None:
                break
            undo = self._apply(teams, move)
            _, a, b, _, _ = move
            ka = self.team_key(teams[a], evaluator, pair_counts)
            kb = self.team_key(teams[b], evaluator, pair_counts)
            self._undo(teams, undo)
            if ka[0] + kb[0] != keys[a][0] + keys[b][0]:
                continue
            delta = -(ka[1] + kb[1]) + (keys[a][1] + keys[b][1])
            if delta < -1e-9:
                worsening.append(-delta)
        if not worsening:
            return 1.0
        return statistics.median(worsening) / math.log(2.0)

    def _parallel_chains(
        self,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int],
        initial_temp: float,
        cooling_rate: float,
        max_iterations: int,
        num_cores: int,
        start_time: float,
        seed: int,
        auto_temperature: bool = True,
    ) -> AlgorithmResult:
        callback = self.progress_callback
        n_chains = num_cores
        workers = max_workers(num_cores)
        seeds = [seed] + [derive_seed(seed, "chain", i) for i in range(1, n_chains)]
        error: Optional[str] = None
        best_result: Optional[AlgorithmResult] = None
        futures: Dict[Any, int] = {}

        try:
            executor = get_process_executor(workers)
            chain_func = partial(
                _run_sa_chain,
                students=students,
                config=config,
                evaluator=evaluator,
                pair_counts=dict(pair_counts),
                initial_temp=initial_temp,
                cooling_rate=cooling_rate,
                max_iterations=max_iterations,
                min_temp=self.min_temp,
                strategy=SimulatedAnnealingStrategy(min_temp=self.min_temp),
                auto_temperature=auto_temperature,
            )
            futures = {executor.submit(chain_func, s): i for i, s in enumerate(seeds)}
            results: List[Optional[AlgorithmResult]] = [None] * n_chains
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if self.better(best_result, results[futures[future]]):
                    best_result = results[futures[future]]
                if callback:
                    callback.update(done, n_chains, best_result.soft_score)
            best_result = None
            for result in results:
                if self.better(best_result, result):
                    best_result = result
        except FormationCancelled:
            for future in futures:
                future.cancel()
            raise
        except Exception as exc:
            handle_pool_failure(workers, exc)
            error = f"{type(exc).__name__}: {exc}"
            best_result = self._run_single_chain(
                students, config, evaluator, pair_counts,
                initial_temp, cooling_rate, max_iterations,
                rng=random.Random(seed), auto_temperature=auto_temperature,
            )

        success = error is None
        best_result.execution_time = time.time() - start_time
        best_result.stats.update({
            "seed": seed,
            "num_chains": n_chains if success else 1,
            "parallel_execution": success,
            "multiprocessing_successful": success,
            "actual_cores_used": workers if success else 1,
            "requested_cores": num_cores,
        })
        if error:
            best_result.stats["multiprocessing_error"] = error
        return best_result

    def _run_single_chain(
        self,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int],
        initial_temp: float,
        cooling_rate: float,
        max_iterations: int,
        rng: Optional[random.Random] = None,
        auto_temperature: bool = True,
    ) -> AlgorithmResult:
        start_time = time.time()
        rng = rng or random.Random(derive_seed(
            getattr(config, "random_seed", SA_DEFAULT_SEED), "anneal-direct"))
        pair_counts = pair_counts or {}
        size_min, size_max = int(config.team_size_min), int(config.team_size_max)
        max_iterations = max(0, int(max_iterations))

        teams = self._generate_initial_solution(students, config, rng)
        keys = [self.team_key(t, evaluator, pair_counts) for t in teams]
        units = sum(k[0] for k in keys)
        neg_soft = sum(k[1] for k in keys)
        initial_key = (units, neg_soft)

        best_teams = self._copy_teams(teams)
        best_key = initial_key

        if auto_temperature:
            temperature = self._calibrate_temperature(
                teams, keys, evaluator, pair_counts, rng, size_min, size_max)
            start_temp = temperature
            end_temp = start_temp * 1e-3
            cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iterations))
            floor = 0.0
        else:
            temperature = start_temp = float(initial_temp)
            cooling = float(cooling_rate)
            floor = float(self.min_temp)

        cold_below = start_temp * self.COLD_FRACTION
        patience = max(500, max_iterations // 5)
        history = [penalised_value(best_key)]
        since_improvement = 0
        accepted = 0
        iteration = 0

        while iteration < max_iterations:
            move = self._propose(teams, rng, size_min, size_max)
            if move is None:
                break
            _, a, b, _, _ = move
            undo = self._apply(teams, move)
            ka = self.team_key(teams[a], evaluator, pair_counts)
            kb = self.team_key(teams[b], evaluator, pair_counts)
            new_units = units - keys[a][0] - keys[b][0] + ka[0] + kb[0]
            new_neg_soft = neg_soft - keys[a][1] - keys[b][1] + ka[1] + kb[1]
            delta = (-new_neg_soft - LEGACY_HARD_PENALTY * new_units) - (
                -neg_soft - LEGACY_HARD_PENALTY * units)
            if delta >= 0:
                accept = True
            elif temperature > 0:
                accept = rng.random() < math.exp(max(-700.0, delta / temperature))
            else:
                accept = False

            if accept:
                keys[a], keys[b] = ka, kb
                units, neg_soft = new_units, new_neg_soft
                accepted += 1
                if (units, neg_soft) < (best_key[0], best_key[1] - 1e-9):
                    best_key = (units, neg_soft)
                    best_teams = self._copy_teams(teams)
                    since_improvement = 0
                else:
                    since_improvement += 1
            else:
                self._undo(teams, undo)
                since_improvement += 1

            temperature = max(floor, temperature * cooling)
            iteration += 1

            if iteration % 100 == 0:
                history.append(penalised_value(best_key))
                if self.progress_callback:
                    self.report_progress(iteration, max_iterations, -best_key[1])

            if since_improvement > patience and temperature <= cold_below:
                break

        swap_stats = {"swaps": 0, "evaluations": 0, "passes": 0}
        if getattr(config, "local_search", True) and len(best_teams) > 1:
            swap_stats = self.local_search_swaps(
                best_teams, evaluator, pair_counts,
                max_evaluations=20000, max_passes=8,
            )

        result = self.build_result(
            best_teams, evaluator, pair_counts,
            execution_time=time.time() - start_time,
            iterations=iteration,
            converged=iteration < max_iterations or temperature <= cold_below,
            stats={
                "initial_temperature": start_temp,
                "final_temperature": temperature,
                "cooling_rate": cooling,
                "auto_temperature": auto_temperature,
                "acceptance_rate": accepted / iteration if iteration else 0.0,
                "initial_score": -initial_key[1],
                "initial_hard_violations": initial_key[0],
                "no_improvement_count": since_improvement,
                "local_search": swap_stats,
                "size_notes": plan_team_sizes(len(students), config).notes,
            },
            history=history,
        )
        result.history.append(penalised_value((result.hard_violations, -result.soft_score)))
        result.stats["improvement"] = result.soft_score - (-initial_key[1])
        return result
