import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import scan_hits  # noqa: E402

DATA = "data/demo/demo_cohort.csv"
HISTORY = "data/demo/demo_prior_projects.csv"
SOLVER = ["--iterations", "3000", "--restarts", "1"]


def _cli(root: Path, args):
    site = str(Path(pytest.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), site]))
    return subprocess.run([sys.executable, "-S", "-m", "app.cli", *args], cwd=root, env=env,
                          capture_output=True, text=True, timeout=600)


def _ok(root, args):
    result = _cli(root, args)
    assert result.returncode == 0, (args, result.stderr[-2000:])
    return result


def _subcommands(root):
    import argparse
    from app.cli import build_parser
    parser = build_parser()
    [sub] = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return list(sub.choices)


@pytest.mark.export
def test_every_subcommand_has_help(export_root, scan_re):
    names = _subcommands(export_root)
    assert {"attributes", "presets", "suggest-policy", "feasibility", "allocate",
            "evaluate", "compare"} <= set(names), names
    for name in names:
        result = _ok(export_root, [name, "--help"])
        assert not scan_hits(result.stdout, scan_re), name


@pytest.mark.export
def test_cli_commands_run_on_the_demo_data(export_root, scan_re, tmp_path):
    from app.engine.defaults import PRESETS
    seen = []

    seen.append(_ok(export_root, ["attributes", "--data", DATA]).stdout)
    seen.append(_ok(export_root, ["presets"]).stdout)
    for preset in PRESETS:
        out = tmp_path / f"suggest_{preset}.json"
        seen.append(_ok(export_root, ["suggest-policy", "--data", DATA, "--preset", preset,
                                      "--out", str(out), *SOLVER]).stdout)
        seen.append(out.read_text(encoding="utf-8"))
    policy = "config/policies/demo_fun.json"
    seen.append(_ok(export_root, ["feasibility", "--data", DATA, "--policy", policy]).stdout)

    alloc = tmp_path / "allocation.csv"
    report = tmp_path / "report.json"
    html = tmp_path / "report.html"
    seen.append(_ok(export_root, ["allocate", "--data", DATA, "--history", HISTORY, "--projects", "3",
                                  "--policy", policy, "--out", str(alloc), "--report", str(report),
                                  "--report-html", str(html), *SOLVER]).stdout)
    alloc2 = tmp_path / "allocation2.csv"
    seen.append(_ok(export_root, ["allocate", "--data", DATA, "--history", HISTORY, "--projects", "3",
                                  "--policy", policy, "--out", str(alloc2), "--seed", "2",
                                  *SOLVER]).stdout)
    seen.append(_ok(export_root, ["evaluate", "--data", DATA, "--allocation", str(alloc),
                                  "--history", HISTORY]).stdout)
    seen.append(_ok(export_root, ["compare", "--data", DATA, "--a", str(alloc), "--b", str(alloc2),
                                  "--history", HISTORY, "--report", str(tmp_path / "compare.json")]).stdout)
    for path in (alloc, report, html):
        seen.append(path.read_text(encoding="utf-8"))
    hits = sorted({h for text in seen for h in scan_hits(text.replace(str(tmp_path), "<tmp>"), scan_re)})
    assert not hits, hits
