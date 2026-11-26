from __future__ import annotations

import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .base import (
    FormationCancelled,
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


GA_DEFAULT_SEED = 42


def _evaluate_fitness_parallel(
    individual: "Individual",
    students: List["Student"],
    config: "Configuration",
    evaluator: "ConstraintEvaluator",
    pair_counts: Dict[tuple, int],
    strategy: "GeneticAlgorithmStrategy"
) -> Tuple[float, float]:
    return strategy._solution_key_of(individual, students, evaluator, pair_counts)


@dataclass
class Individual:
    assignment: List[int]
    fitness: float = 0.0
    units: float = 0.0
    soft: float = 0.0
    evaluated: bool = False

    def copy(self) -> "Individual":
        return Individual(assignment=self.assignment.copy(), fitness=self.fitness,
                          units=self.units, soft=self.soft, evaluated=self.evaluated)

    def set_key(self, key: Tuple[float, float]) -> None:
        self.units, self.soft = key[0], -key[1]
        self.fitness = penalised_value(key)
        self.evaluated = True

    def invalidate(self) -> None:
        self.evaluated = False

    @property
    def rank(self) -> Tuple[float, float]:
        return (-self.units, self.soft)


class GeneticAlgorithmStrategy(TeamFormationStrategy):
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 5,
        elitism: int = 2
    ):
        super().__init__("Genetic Algorithm")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self._rng = random.Random(GA_DEFAULT_SEED)

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

        population_size = max(2, int(kwargs.get("population_size", config.ga_population_size)))
        generations = max(0, int(kwargs.get("generations", config.ga_generations)))
        mutation_rate = float(kwargs.get("mutation_rate", config.ga_mutation_rate))
        num_cores = int(kwargs.get("num_cores", config.num_cores) or 1)

        n_students = len(students)
        plan = plan_team_sizes(n_students, config)
        team_sizes = plan.sizes
        n_teams = len(team_sizes)

        seed = derive_seed(getattr(config, "random_seed", GA_DEFAULT_SEED), "genetic",
                           input_fingerprint(students, pair_counts, team_sizes))
        self._rng = random.Random(seed)
        rng = self._rng

        population = self._initialize_population(n_students, team_sizes, population_size)

        fitness_func = partial(
            _evaluate_fitness_parallel,
            students=students,
            config=config,
            evaluator=evaluator,
            pair_counts=pair_counts,
            strategy=GeneticAlgorithmStrategy(),
        )

        workers = max_workers(num_cores)
        pool = None
        multiprocessing_successful = False
        mp_error: Optional[str] = None
        if num_cores > 1 and n_students > 1:
            try:
                pool = get_process_executor(workers)
                self._evaluate_batch(population, fitness_func, pool, workers)
                multiprocessing_successful = True
            except Exception as exc:
                handle_pool_failure(workers, exc)
                mp_error = f"{type(exc).__name__}: {exc}"
                pool = None

        self._evaluate_batch(population, None, None, 1, students, evaluator, pair_counts)

        best_individual = max(population, key=lambda ind: ind.rank).copy()
        initial_best = best_individual.copy()
        score_history = [best_individual.fitness]

        stagnation_count = 0
        patience = max(10, min(generations // 4, 100))
        generation = -1

        for generation in range(generations):
            self.check_cancelled()
            population.sort(key=lambda ind: ind.rank, reverse=True)
            new_population = [ind.copy() for ind in population[:self.elitism]]

            while len(new_population) < population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                if rng.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, n_teams, config)
                    self._repair(child1, team_sizes)
                    self._repair(child2, team_sizes)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                for child in (child1, child2):
                    if rng.random() < mutation_rate:
                        self._mutate(child, n_teams, config)
                        child.invalidate()

                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)

            population = new_population
            try:
                self._evaluate_batch(population, fitness_func,
                                     pool if multiprocessing_successful else None, workers,
                                     students, evaluator, pair_counts)
            except FormationCancelled:
                raise
            except Exception as exc:
                handle_pool_failure(workers, exc)
                mp_error = f"{type(exc).__name__}: {exc}"
                multiprocessing_successful = False
                pool = None
                self._evaluate_batch(population, None, None, 1, students, evaluator, pair_counts)

            current_best = max(population, key=lambda ind: ind.rank)
            if (current_best.units, -current_best.soft) < (
                    best_individual.units, -best_individual.soft - 1e-9):
                best_individual = current_best.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            score_history.append(best_individual.fitness)
            if generation % 10 == 0:
                self.report_progress(generation, generations, best_individual.soft)
            if stagnation_count > patience:
                break

        teams = self._decode_individual(best_individual, students, config)

        swap_stats = {"swaps": 0, "evaluations": 0, "passes": 0}
        if getattr(config, "local_search", True) and len(teams) > 1:
            swap_stats = self.local_search_swaps(
                teams, evaluator, pair_counts,
                max_evaluations=20000, max_passes=8,
            )

        result = self.build_result(
            teams, evaluator, pair_counts,
            execution_time=time.time() - start_time,
            iterations=generation + 1,
            converged=stagnation_count > patience,
            stats={
                "seed": seed,
                "population_size": population_size,
                "final_generation": generation + 1,
                "stagnation_count": stagnation_count,
                "initial_best": initial_best.soft,
                "initial_hard_violations": initial_best.units,
                "local_search": swap_stats,
                "size_notes": plan.notes,
                "multiprocessing_enabled": num_cores > 1,
                "multiprocessing_successful": multiprocessing_successful,
                "actual_cores_used": workers if multiprocessing_successful else 1,
                "requested_cores": num_cores,
            },
            history=score_history,
        )
        result.history.append(penalised_value((result.hard_violations, -result.soft_score)))
        result.stats["improvement"] = result.soft_score - initial_best.soft
        if mp_error:
            result.stats["multiprocessing_error"] = mp_error
        return result

    def _evaluate_batch(self, population: List[Individual], fitness_func, pool, workers: int,
                        students=None, evaluator=None, pair_counts=None) -> None:
        pending = [ind for ind in population if not ind.evaluated]
        if not pending:
            return
        if pool is not None and fitness_func is not None:
            chunksize = max(1, len(pending) // (workers * 2) or 1)
            for individual, key in zip(pending, pool.map(fitness_func, pending, chunksize=chunksize)):
                individual.set_key(key)
            return
        if students is None:
            return
        for k, individual in enumerate(pending):
            if k % 32 == 0:
                self.check_cancelled()
            individual.set_key(self._solution_key_of(individual, students, evaluator, pair_counts))

    def _initialize_population(
        self,
        n_students: int,
        team_sizes: List[int],
        population_size: int,
    ) -> List[Individual]:
        population = []
        student_order = list(range(n_students))
        for _ in range(population_size):
            self._rng.shuffle(student_order)
            assignment = [0] * n_students
            pos = 0
            for team_idx, size in enumerate(team_sizes):
                for student_idx in student_order[pos:pos + size]:
                    assignment[student_idx] = team_idx
                pos += size
            population.append(Individual(assignment=assignment))
        return population

    def _solution_key_of(self, individual: Individual, students: List["Student"],
                         evaluator: "ConstraintEvaluator",
                         pair_counts: Dict[tuple, int]) -> Tuple[float, float]:
        teams = self._decode_individual(individual, students, None)
        return self.solution_key(teams, evaluator, pair_counts)

    def _evaluate_fitness(
        self,
        individual: Individual,
        students: List["Student"],
        config: "Configuration",
        evaluator: "ConstraintEvaluator",
        pair_counts: Dict[tuple, int]
    ) -> float:
        individual.set_key(self._solution_key_of(individual, students, evaluator, pair_counts))
        return individual.fitness

    def _decode_individual(
        self,
        individual: Individual,
        students: List["Student"],
        config: Optional["Configuration"]
    ) -> List[Team]:
        team_members: Dict[int, List] = {}
        for student_idx, team_idx in enumerate(individual.assignment):
            team_members.setdefault(team_idx, []).append(students[student_idx])
        return [
            Team(team_id=f"team_{team_idx + 1}", members=list(team_members[team_idx]))
            for team_idx in sorted(team_members)
        ]

    def _tournament_select(self, population: List[Individual]) -> Individual:
        tournament = self._rng.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.rank)

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        n_teams: int,
        config: "Configuration"
    ) -> Tuple[Individual, Individual]:
        child1, child2 = [], []
        for gene1, gene2 in zip(parent1.assignment, parent2.assignment):
            if self._rng.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)
        return Individual(assignment=child1), Individual(assignment=child2)

    def _mutate(
        self,
        individual: Individual,
        n_teams: int,
        config: "Configuration"
    ) -> None:
        assignment = individual.assignment
        n = len(assignment)
        if n < 2 or len(set(assignment)) < 2:
            return
        for _ in range(self._rng.randint(1, min(3, n))):
            for _attempt in range(10):
                i, j = self._rng.randrange(n), self._rng.randrange(n)
                if assignment[i] != assignment[j]:
                    assignment[i], assignment[j] = assignment[j], assignment[i]
                    break

    def _repair(
        self,
        individual: Individual,
        team_sizes: List[int]
    ) -> None:
        n_teams = len(team_sizes)
        assignment = individual.assignment
        members: Dict[int, List[int]] = {t: [] for t in range(n_teams)}
        homeless: List[int] = []
        for i, team_idx in enumerate(assignment):
            if 0 <= team_idx < n_teams:
                members[team_idx].append(i)
            else:
                homeless.append(i)
        for team_idx in range(n_teams):
            surplus = len(members[team_idx]) - team_sizes[team_idx]
            if surplus > 0:
                movers = self._rng.sample(members[team_idx], surplus)
                homeless.extend(movers)
                mover_set = set(movers)
                members[team_idx] = [i for i in members[team_idx] if i not in mover_set]
        self._rng.shuffle(homeless)
        open_slots = [t for t in range(n_teams)
                      for _ in range(team_sizes[t] - len(members[t]))]
        self._rng.shuffle(open_slots)
        for student_idx, team_idx in zip(homeless, open_slots):
            assignment[student_idx] = team_idx
        if len(homeless) != len(open_slots):
            for student_idx in homeless[len(open_slots):]:
                assignment[student_idx] = min(range(n_teams), key=lambda t: len(members[t]))
        individual.invalidate()
