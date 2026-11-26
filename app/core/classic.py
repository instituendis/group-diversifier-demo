from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..algorithms import GeneticAlgorithmStrategy, GreedyStrategy, SimulatedAnnealingStrategy
from ..algorithms.base import FormationCancelled, ProgressCallback
from ..algorithms.pool_utils import estimate_time
from ..constraints import ConstraintFactory
from ..engine.sizing import project_config
from ..utils.serialization import json_safe
from .assignment_import import build_imported_projects, seed_pair_counts
from .diversity_metrics import DiversityMetrics
from .ids import pair_key
from .models import Configuration, FormationResult, Project, Student
from .job_runner import run_slot
from .session_state import resolve_state
from .. import demo, hosting

CancelFn = Callable[[], bool]

ProgressFn = Callable[["ClassicProgress"], None]

_CLASSIC_LABELS = {"greedy": "Greedy", "simulated_annealing": "Simulated Annealing",
                   "genetic": "Genetic Algorithm"}


@dataclass
class ClassicRun:
    projects: List[Project]
    result: FormationResult
    stats: Dict[str, Any]


@dataclass
class ClassicProgress:
    stage: str
    project_idx: int = 0
    num_projects: int = 0
    elapsed: float = 0.0
    elapsed_project: float = 0.0
    iteration: int = 0
    total: int = 0
    score: float = 0.0
    updated: bool = False
    iterations: int = 0
    remaining: Optional[float] = None
    last_project_time: Optional[float] = None
    algo_label: str = ""
    num_cores: int = 1
    n_students: int = 0
    n_generated: int = 0
    estimate: str = ""
    message: str = ""


class ClassicRunError(RuntimeError):
    def __init__(self, project_idx: int, cause: BaseException):
        super().__init__(cause)
        self.project_idx = project_idx
        self.cause = cause


def _plain_label(algorithm: str) -> str:
    return _CLASSIC_LABELS.get(algorithm, algorithm.replace("_", " ").title())


def classic_progress(update: Callable[[int, int, float], None],
                     cancel: CancelFn) -> ProgressCallback:
    def guarded(iteration: int, total: int, score: float) -> None:
        if cancel():
            raise FormationCancelled("Team formation was stopped")
        update(iteration, total, score)

    return ProgressCallback(guarded, cancel=cancel)


def estimate(config: Configuration, n_students: int,
             n_projects: Optional[int] = None) -> str:
    if n_projects is None:
        n_projects = config.num_projects
    return estimate_time(n_students, n_projects, config.algorithm, int(config.num_cores))


def run(config: Configuration, students: List[Student], *, imported_projects: Any = (),
        participants: Any = None, progress: Optional[ProgressFn] = None,
        cancel: Optional[CancelFn] = None) -> ClassicRun:
    if demo.classic_hidden():
        # INVARIANT: the pages never offer a classic algorithm on a hosted server
        # (not the demo), and the configuration is coerced to the rules engine. Refuse anyway.
        raise ClassicRunError(-1, RuntimeError(
            f"The classic algorithms are not available on {demo.classic_unavailable_where()}; "
            "use the rules engine."))
    events = progress if progress is not None else (lambda event: None)
    if int(config.num_cores) != hosting.clamp_workers(config.num_cores):
        config = copy.deepcopy(config)
        config.num_cores = hosting.clamp_workers(config.num_cores)
    with run_slot(cancel=cancel,
                  on_wait=lambda text: events(ClassicProgress(stage="waiting", message=text))):
        return _run(config, students, events, cancel, imported_projects=imported_projects,
                    participants=participants)


def _run(config: Configuration, students: List[Student], events: ProgressFn,
         cancel: Optional[CancelFn], *, imported_projects: Any,
         participants: Any) -> ClassicRun:
    imported_count = imported_projects.project_count if imported_projects else 0
    n_generated_projects = config.num_projects - imported_count
    events(ClassicProgress(stage="start", algo_label=_plain_label(config.algorithm),
                           num_cores=int(config.num_cores), n_students=len(students),
                           n_generated=n_generated_projects,
                           estimate=estimate_time(len(students), n_generated_projects,
                                                  config.algorithm, config.num_cores)))

    project_times: List[float] = []
    live_progress = {"iteration": 0, "total": 0, "score": 0.0, "updated": False}

    cancelled = threading.Event()
    time_cap = hosting.max_classic_time()
    project_deadline: List[Optional[float]] = [None]
    timed_out = threading.Event()

    def update_progress(iteration, total, score):
        live_progress["iteration"] = iteration
        live_progress["total"] = total
        live_progress["score"] = score
        live_progress["updated"] = True

    def stop_requested() -> bool:
        if cancelled.is_set() or bool(cancel and cancel()):
            return True
        limit = project_deadline[0]
        if limit is not None and time.monotonic() >= limit:
            timed_out.set()
            return True
        return False

    callback = classic_progress(update_progress, stop_requested)
    if config.algorithm == "simulated_annealing":
        strategy = SimulatedAnnealingStrategy(
            initial_temp=config.sa_initial_temp,
            cooling_rate=config.sa_cooling_rate,
            max_iterations=config.sa_iterations,
        )
    elif config.algorithm == "genetic":
        strategy = GeneticAlgorithmStrategy(
            population_size=config.ga_population_size,
            generations=config.ga_generations,
            mutation_rate=config.ga_mutation_rate,
        )
    else:
        strategy = GreedyStrategy(randomize_start=True)
    strategy.set_progress_callback(callback)

    students_by_id = {s.student_id: s for s in students}
    projects: List[Project] = []
    pair_counts: Dict[Tuple[str, str], int] = {}
    result = None
    if imported_projects:
        projects.extend(build_imported_projects(imported_projects, students_by_id))
        pair_counts.update(seed_pair_counts(imported_projects))
    violations: List[str] = []
    per_project_stats: Dict[str, Dict[str, Any]] = {}
    total_start = time.time()

    for project_idx in range(imported_count, config.num_projects):
        project_start = time.time()
        project_id = str(project_idx + 1)
        pool_ids = participants.pool_for(project_id) if participants else None
        if pool_ids is not None:
            pool_set = set(pool_ids)
            project_students = [s for s in students if s.student_id in pool_set]
        else:
            project_students = students
        p_config = project_config(config, project_idx)
        evaluator = ConstraintFactory.create_from_config(p_config, students=project_students)
        prior_counts = dict(pair_counts)

        events(ClassicProgress(stage="project_start", project_idx=project_idx,
                               num_projects=config.num_projects,
                               elapsed=time.time() - total_start,
                               remaining=(sum(project_times) / len(project_times))
                                         * (config.num_projects - project_idx)
                               if project_times else None))

        live_progress.update({"iteration": 0, "total": 0, "score": 0.0, "updated": False})
        result_holder = [None]
        error_holder = [None]
        project_deadline[0] = None if time_cap is None else time.monotonic() + time_cap
        timed_out.clear()

        def run_algorithm():
            try:
                result_holder[0] = strategy.form_teams(
                    students=project_students,
                    config=p_config,
                    evaluator=evaluator,
                    existing_pair_counts=prior_counts,
                )
            except FormationCancelled:
                pass
            except Exception as exc:
                error_holder[0] = exc

        algo_thread = threading.Thread(target=run_algorithm, daemon=True,
                                       name="classic-formation-worker")
        algo_thread.start()

        try:
            while algo_thread.is_alive():
                if project_deadline[0] is not None and time.monotonic() >= project_deadline[0]:
                    timed_out.set()
                    cancelled.set()
                elapsed_project = time.time() - project_start
                events(ClassicProgress(stage="tick", elapsed=time.time() - total_start,
                                       elapsed_project=elapsed_project,
                                       iteration=live_progress["iteration"],
                                       total=live_progress["total"],
                                       score=live_progress["score"],
                                       updated=live_progress["updated"],
                                       last_project_time=project_times[-1] if project_times else None))
                algo_thread.join(timeout=0.3)
        finally:
            if algo_thread.is_alive():
                cancelled.set()
        algo_thread.join()

        if error_holder[0]:
            raise ClassicRunError(project_idx, error_holder[0])
        if result_holder[0] is None:
            if timed_out.is_set():
                raise ClassicRunError(project_idx, TimeoutError(
                    f"this hosted server stops one classic project after {time_cap} s "
                    "(a limit its operator sets); use the rules engine or fewer students"))
            raise ClassicRunError(project_idx, FormationCancelled("Team formation was stopped"))

        result = result_holder[0]
        project_elapsed = time.time() - project_start
        project_times.append(project_elapsed)

        project = Project(project_id=project_id, name=f"Project {project_id}", teams=result.teams)
        for team in project.teams:
            team.project_id = project_id
            for violation in evaluator.get_violations(team, pair_counts=prior_counts):
                violations.append(f"{project.name} · {team.team_id}: {violation}")
        projects.append(project)

        stats = json_safe(result.stats or {})
        per_project_stats[project_id] = {
            "score": float(result.score) if math.isfinite(result.score) else None,
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "execution_time": round(project_elapsed, 3),
            "history": json_safe(list(result.history or stats.get("history", []))),
            "requested_cores": stats.get("requested_cores", config.num_cores),
            "multiprocessing_successful": stats.get("multiprocessing_successful", False),
            "actual_cores_used": stats.get("actual_cores_used", 1),
        }

        events(ClassicProgress(stage="project_done", project_idx=project_idx,
                               score=result.score, iterations=int(result.iterations),
                               elapsed_project=project_elapsed))

        for team in result.teams:
            for a, b in team.get_pairs():
                key = pair_key(a, b)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    events(ClassicProgress(stage="done", num_projects=config.num_projects,
                           elapsed=time.time() - total_start))
    total_time = time.time() - total_start

    metrics = DiversityMetrics(config, cohort_students=students)
    all_diversity_scores: Dict[str, List[float]] = {}
    for project in projects:
        for attr, score in metrics.calculate_project_diversity(project).get("average_scores", {}).items():
            all_diversity_scores.setdefault(attr, []).append(score)
    avg_diversity = {attr: sum(s) / len(s) for attr, s in all_diversity_scores.items()}
    total_score = sum(
        metrics.calculate_weighted_score(metrics.calculate_team_diversity(team))
        for project in projects
        for team in project.teams
    )

    algorithm_stats = json_safe(dict(result.stats)) if result else {}
    algorithm_stats["per_project"] = per_project_stats
    formation_result = FormationResult(
        projects=projects,
        config=config,
        total_score=total_score,
        diversity_scores=avg_diversity,
        constraint_violations=violations,
        overlap_matrix=pair_counts,
        execution_time=total_time,
        algorithm_stats=algorithm_stats,
    )
    return ClassicRun(projects=projects, result=formation_result, stats=algorithm_stats)


def activate(config: Configuration, students: List[Student], run_data: ClassicRun,
             state: Any = None) -> None:
    if state is None:
        state = resolve_state()
    state.formation_result = run_data.result
    state.projects = run_data.projects


def parameter_echo(config: Configuration) -> str:
    lines: List[str] = []
    if config.algorithm != "engine":
        lines.append(f"- **Max Repeat Pairings (classic algorithms):** "
                     f"{config.max_repeat_pairings}")
    if config.algorithm == "simulated_annealing":
        lines.append(f"- **Initial Temp:** {config.sa_initial_temp}")
        lines.append(f"- **Cooling Rate:** {config.sa_cooling_rate}")
        lines.append(f"- **Max Iterations:** {config.sa_iterations}")
    elif config.algorithm == "genetic":
        lines.append(f"- **Population:** {config.ga_population_size}")
        lines.append(f"- **Generations:** {config.ga_generations}")
        lines.append(f"- **Mutation Rate:** {config.ga_mutation_rate}")
    return "\n".join(lines)
