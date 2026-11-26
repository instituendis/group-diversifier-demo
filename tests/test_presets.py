import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import exceptions, page, run_page  # noqa: E402


def _preset_stems(root: Path):
    return sorted(p.stem for p in (root / "config" / "presets").glob("*.json"))


@pytest.mark.export
@pytest.mark.parametrize("name", ["colour_and_city_first", "equal_weights", "mirror_city_shares"])
def test_each_generic_preset_loads_and_a_greedy_run_succeeds(export_root, config_dir, demo_run,
                                                             monkeypatch, name):
    assert name in _preset_stems(export_root), name
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    from app.core import classic

    at = run_page(page(export_root, 2), state=demo_run)
    at.selectbox(key="config_preset_choice").set_value(name)
    [load] = [b for b in at.button if b.label == "Load Preset"]
    load.click().run()
    assert not exceptions(at), exceptions(at)
    cfg = copy.deepcopy(at.session_state["config"])
    cfg.algorithm = "greedy"
    cfg.num_projects = 3
    cfg.team_size_min, cfg.team_size_max, cfg.team_size_target = 3, 5, 4
    cfg.num_cores = 1
    cfg.random_seed = 42
    students = demo_run["_data_loader"].get_students()
    run_data = classic.run(cfg, students, imported_projects=demo_run["imported_assignments"])
    assert len(run_data.projects) == 3
