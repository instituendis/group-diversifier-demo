import importlib
import os
import re
import shutil
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _default_root() -> str:
    root = _HERE.parent
    return str(root) if _HERE.name == "tests" and (root / "DEMO_MODE").is_file() else ""


_EXPORT_ROOT = os.environ.get("GD_EXPORT_ROOT", "").strip() or _default_root()

collect_ignore_glob = [] if _EXPORT_ROOT else ["*.py"]

HOSTING_ENV = ("GROUP_DIVERSIFIER_HOSTED", "GROUP_DIVERSIFIER_SOURCE_URL",
               "GROUP_DIVERSIFIER_PRIVACY_NOTICE", "GROUP_DIVERSIFIER_PRIVACY_URL",
               "GROUP_DIVERSIFIER_MAX_WORKERS", "GROUP_DIVERSIFIER_MAX_CONCURRENT_RUNS",
               "GROUP_DIVERSIFIER_MAX_TIME_LIMIT", "GROUP_DIVERSIFIER_MAX_TABLE_UPLOAD_MB",
               "GROUP_DIVERSIFIER_MAX_CLASSIC_TIME")

DEMO_COLUMNS = ["favourite_colour", "ice_cream", "city", "hobbies", "pet", "coffees_per_day"]


def pytest_configure(config):
    config.addinivalue_line("markers", "export: runs against the exported tree")


def pytest_addoption(parser):
    try:
        parser.addoption("--run-slow", action="store_true", default=False,
                         help="run tests marked @pytest.mark.slow (equivalent to RUN_SLOW=1)")
    except ValueError:
        pass


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow", default=False) or \
            os.environ.get("RUN_SLOW", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    skip_slow = pytest.mark.skip(reason="slow test: pass --run-slow or set RUN_SLOW=1")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def export_root() -> Path:
    assert _EXPORT_ROOT, "GD_EXPORT_ROOT must name the exported tree"
    root = Path(_EXPORT_ROOT).resolve()
    app_file = Path(importlib.import_module("app").__file__).resolve()
    assert root in app_file.parents, f"app was imported from {app_file}, not from the export {root}"
    from app import extensions
    assert extensions.installed() is False
    return root


RAW_PYTHON_TEXT = (r"Traceback \(most recent call last\)|\bobject at 0x[0-9a-fA-F]+"
                   r"|<(?:function|bound method) \w")


@pytest.fixture(scope="session")
def scan_re() -> "re.Pattern":
    extra = os.environ.get("GD_EXTRA_SCAN_RE", "")
    return re.compile(f"{extra}|{RAW_PYTHON_TEXT}" if extra else RAW_PYTHON_TEXT)


@pytest.fixture(autouse=True)
def clean_hosting(monkeypatch):
    for name in HOSTING_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config_dir(export_root, tmp_path, monkeypatch):
    target = tmp_path / "config"
    shutil.copytree(export_root / "config", target)
    monkeypatch.setenv("GROUP_DIVERSIFIER_CONFIG_DIR", str(target))
    return target


@pytest.fixture(scope="session")
def demo_run(export_root, tmp_path_factory):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from support import exceptions, page, run_page

    patch = pytest.MonkeyPatch()
    target = tmp_path_factory.mktemp("seeded") / "config"
    shutil.copytree(export_root / "config", target)
    patch.setenv("GROUP_DIVERSIFIER_CONFIG_DIR", str(target))
    patch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    try:
        at = run_page(page(export_root, 3))
        at.session_state["config"].policy["solver"].update(
            {"iterations": 3000, "restarts": 1, "time_limit_s": 10})
        at.run()
        [run] = [b for b in at.button if "Run Team Formation" in b.label]
        run.click().run()
        assert not exceptions(at), exceptions(at)
        keys = ["config", "formation_result", "projects", "imported_assignments", "_import_sources",
                "engine_reports", "engine_scored_policy", "engine_result_meta", "run_history",
                "active_run_id", "_data_loader", "_demo_history_seeded", "uploaded_csv"]
        seeded = {k: at.session_state[k] for k in keys if k in at.session_state}
    finally:
        patch.undo()
    yield seeded
