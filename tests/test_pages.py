import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import (PAGE_NUMBERS, all_text, exceptions, page, scan_hits,  # noqa: E402
                     rendered_text, run_page)

DEMO_COLUMNS = ["favourite_colour", "ice_cream", "city", "hobbies", "pet", "coffees_per_day"]


@pytest.mark.export
@pytest.mark.parametrize("seeded", [False, True], ids=["fresh", "seeded"])
@pytest.mark.parametrize("number", PAGE_NUMBERS, ids=str)
def test_every_page_renders_without_exceptions_or_raw_text(
        export_root, config_dir, demo_run, scan_re, number, seeded):
    at = run_page(page(export_root, number), state=demo_run if seeded else None)
    assert not exceptions(at), exceptions(at)
    text, labels = rendered_text(at)
    assert not scan_hits(text, scan_re), (number, seeded, scan_hits(text, scan_re))
    assert not scan_hits("\n".join(sorted(labels)), scan_re), (number, seeded)


@pytest.mark.export
def test_configuration_shows_the_demo_dimensions_and_the_generic_presets(
        export_root, config_dir, monkeypatch):
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    at = run_page(page(export_root, 2))
    assert not exceptions(at), exceptions(at)
    assert [t.label for t in at.tabs] == ["👥 Team Settings", "🔧 Algorithm",
                                          "⚖️ Diversity Weights (classic)",
                                          "🚫 Constraints (classic)", "📋 Summary"]
    slider_keys = {s.key for s in at.slider if s.key}
    assert {k for k in slider_keys if k.startswith("dim_weight_")} == {
        f"dim_weight_{c}" for c in DEMO_COLUMNS}
    [preset_box] = [s for s in at.selectbox if s.key == "config_preset_choice"]
    stems = sorted(p.stem for p in (export_root / "config" / "presets").glob("*.json"))
    assert list(preset_box.options) == stems
    assert "Select Algorithm" in [s.label for s in at.selectbox]


@pytest.mark.export
def test_profile_writes_work_in_demo_and_are_read_only_when_hosted(
        export_root, config_dir, monkeypatch, tmp_path):
    from app.core.config_loader import ConfigurationLoader, ProfilesReadOnly

    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    profile = ConfigurationLoader(config_dir).load_profile("demo")
    saved = ConfigurationLoader(tmp_path / "written").save_profile(profile, "demo_copy")
    assert saved.is_file()
    monkeypatch.setenv("GROUP_DIVERSIFIER_HOSTED", "1")
    with pytest.raises(ProfilesReadOnly):
        ConfigurationLoader(tmp_path / "nothing").save_profile(profile, "demo_copy")
    assert not (tmp_path / "nothing").exists()


@pytest.mark.export
def test_the_demo_profile_describes_exactly_the_six_demo_attributes(export_root, config_dir):
    from app.core.config_loader import ConfigurationLoader
    profile = ConfigurationLoader(config_dir).load_profile("demo")
    names = sorted(d.name for d in profile.dimensions)
    assert names == sorted(DEMO_COLUMNS)


@pytest.mark.export
def test_profile_manager_lists_the_demo_profile(export_root, config_dir, monkeypatch):
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", "1")
    at = run_page(page(export_root, 8))
    assert not exceptions(at), exceptions(at)
    shown = all_text(at) + "\n" + "\n".join(str(o) for s in at.selectbox for o in s.options)
    assert "demo" in shown


@pytest.mark.export
@pytest.mark.parametrize("demo", ["1", "0"])
def test_every_sample_control_has_its_file(export_root, config_dir, demo_run, monkeypatch, demo):
    monkeypatch.setenv("GROUP_DIVERSIFIER_DEMO", demo)
    wanted = {1: {"⬇️ Download sample cohort", "⬇️ Download sample preferences"},
              3: {"⬇️ Download sample earlier groups", "⬇️ Download sample participant lists"}}
    for number, labels in wanted.items():
        at = run_page(page(export_root, number))
        assert not exceptions(at), exceptions(at)
        assert labels <= {d.label for d in at.get("download_button")}, (number, demo)
    at = run_page(page(export_root, 12), state=demo_run)
    assert not exceptions(at), exceptions(at)
    [source] = [r for r in at.radio if r.key == "compare_ext_source"]
    assert "Bundled sample reference groups" in source.options


@pytest.mark.export
def test_the_demo_sample_files_fit_the_demo_roster(export_root):
    from app import demo
    from app.core.assignment_import import (parse_assignments_csv, parse_participants_csv,
                                            validate_against_roster, validate_participants)

    ids = {line.split(",", 1)[0] for line in
           (demo.DEMO_DIR / "demo_cohort.csv").read_text(encoding="utf-8").splitlines()[1:]}
    assert ids == {str(i) for i in range(1, 101)}
    for kind in ("earlier_groups", "reference_groups"):
        imp, errors = parse_assignments_csv((demo.DEMO_DIR / demo.DEMO_SAMPLE_FILES[kind][0]).read_bytes())
        assert errors == [], kind
        validate_against_roster(imp, ids)
        assert not imp.unknown_students, kind
    spec, errors = parse_participants_csv((demo.DEMO_DIR / "demo_participants.csv").read_bytes())
    assert errors == []
    validate_participants(spec, ids, 1, 3)
    assert spec.warnings == []
    rows = (demo.DEMO_DIR / "demo_preferences.csv").read_text(encoding="utf-8").splitlines()[1:]
    named = {sid for row in rows for cell in row.split(",") for sid in cell.split("|") if sid}
    assert named <= ids


@pytest.mark.export
def test_the_local_launchers_listen_on_this_computer_only(export_root):
    import tomllib
    config = tomllib.loads((export_root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["address"] == "127.0.0.1"
