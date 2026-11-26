import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import exceptions, page, run_page  # noqa: E402

DEMO_COLUMNS = ["favourite_colour", "ice_cream", "city", "hobbies", "pet", "coffees_per_day"]


@pytest.mark.export
@pytest.mark.parametrize("algorithm", ["greedy", "simulated_annealing", "genetic"])
def test_each_classic_algorithm_forms_the_demo_teams(export_root, config_dir, demo_run,
                                                     monkeypatch, algorithm):
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    from app.core import classic
    from app.core.models import Configuration  # noqa: F401  (the class the config is built on)

    cfg = copy.deepcopy(demo_run["config"])
    cfg.algorithm = algorithm
    cfg.num_projects = 3
    cfg.team_size_min, cfg.team_size_max, cfg.team_size_target = 3, 5, 4
    cfg.num_cores = 1
    cfg.random_seed = 42
    cfg.sa_iterations = 2000
    cfg.ga_population_size = 10
    cfg.ga_generations = 5
    loader = demo_run["_data_loader"]
    students = loader.get_students()

    run_data = classic.run(cfg, students, imported_projects=demo_run["imported_assignments"])
    assert [p.project_id for p in run_data.projects] == ["1", "2", "3"]
    for project in run_data.projects[1:]:
        placed = [sid for team in project.teams for sid in team.member_ids]
        assert len(placed) == len(set(placed)) == len(students), project.project_id
        assert all(3 <= team.size <= 5 for team in project.teams), project.project_id
    assert math.isfinite(run_data.result.total_score)
    assert set(run_data.result.diversity_scores) == set(DEMO_COLUMNS)

    from types import SimpleNamespace
    state = SimpleNamespace()
    classic.activate(cfg, students, run_data, state=state)
    assert state.formation_result is run_data.result


@pytest.mark.export
def test_team_formation_runs_the_greedy_classic_algorithm(export_root, config_dir, monkeypatch):
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    at = run_page(page(export_root, 3))
    at.session_state["config"].algorithm = "greedy"
    at.run()
    [run] = [b for b in at.button if "Run Team Formation" in b.label]
    run.click().run()
    assert not exceptions(at), exceptions(at)
    assert at.session_state["config"].algorithm == "greedy"
    assert at.session_state["formation_result"] is not None
