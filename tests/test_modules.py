import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.mark.export
def test_every_module_imports_and_no_extension_is_installed(export_root):
    import app
    from app import extensions

    seen = []
    for info in pkgutil.walk_packages(app.__path__, prefix="app."):
        package = getattr(extensions, "PACKAGE", None)
        if (info.name.endswith("__main__") or ".pages" in info.name
                or (package and info.name.startswith(package))):
            continue
        importlib.import_module(info.name)
        seen.append(info.name)
    assert extensions.installed() is False
    assert {"app.core.models", "app.engine.runner", "app.core.classic", "app.utils.exporters"} <= set(seen)
