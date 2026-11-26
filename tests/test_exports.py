import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import scan_hits  # noqa: E402

DEMO_COLUMNS = ["favourite_colour", "ice_cream", "city", "hobbies", "pet", "coffees_per_day"]


@pytest.fixture
def export_blobs(demo_run, monkeypatch):
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    from app.core import engine_session as es
    from app.utils import lms_export
    from app.utils.exporters import DataExporter, export_configuration_to_json, plain_projects
    from app.utils.report_html import render_allocation_html

    loader = demo_run["_data_loader"]
    frame = es.roster_frame(loader.get_students(), loader.get_dataframe(), use_cache=False)
    exporter = DataExporter(roster=frame, reports=demo_run["engine_reports"],
                            raw_df=loader.get_dataframe())
    result = demo_run["formation_result"]
    projects = demo_run["projects"]
    sheets = pd.read_excel(io.BytesIO(exporter.export_teams_to_excel(result)), sheet_name=None)
    blobs = {
        "csv": exporter.export_projects_to_csv(projects),
        "json": exporter.export_to_json(result),
        "markdown": exporter.generate_report(result),
        "html": render_allocation_html(plain_projects(projects), frame, demo_run["engine_reports"],
                                       title="Demo", attributes=None),
        "config": export_configuration_to_json(demo_run["config"]),
    }
    blobs["excel_headers"] = "\n".join(" ".join(map(str, df.columns)) for df in sheets.values())
    for name in ("canvas", "brightspace"):
        blobs[name] = lms_export.lms_group_csv(name, plain_projects(projects))
    blobs["moodle"] = lms_export.lms_group_csv("moodle", plain_projects(projects),
                                              course_shortname="demo")
    return blobs, sheets


@pytest.mark.export
def test_the_exports_hold_no_raw_text(export_blobs, scan_re):
    blobs, _sheets = export_blobs
    hits = {name: scan_hits(text, scan_re) for name, text in blobs.items()}
    assert not {k: v for k, v in hits.items() if v}, hits


@pytest.mark.export
def test_the_excel_export_carries_the_classic_metrics_on_the_demo_dimensions(export_blobs):
    _blobs, sheets = export_blobs
    classic = sheets["Classic_Metrics"]
    assert {m[len("Diversity: "):] for m in classic["Metric"] if m.startswith("Diversity: ")} \
        == set(DEMO_COLUMNS)


@pytest.mark.export
def test_the_session_file_round_trips(demo_run, monkeypatch):
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    from app.core import session_store

    blob = session_store.snapshot_json(dict(demo_run))
    payload = json.loads(blob)
    restored = {}
    session_store.restore_session(restored, payload)
    assert restored["config"].algorithm == demo_run["config"].algorithm
    assert len(restored["projects"]) == len(demo_run["projects"])
