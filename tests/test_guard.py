import pytest


@pytest.mark.export
def test_the_suite_tests_the_export(export_root):
    assert export_root.is_dir()
