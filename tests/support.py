import re
from typing import List, Set, Tuple

from streamlit.testing.v1 import AppTest

TIMEOUT = 180
SKIP_PROTOS = ("Block", "ArrowDataFrame", "ArrowTable")


def exceptions(at: AppTest) -> List[str]:
    return [str(e.value) for e in at.exception]


def all_text(at: AppTest) -> str:
    parts = [str(m.value) for m in at.markdown]
    for kind in ("info", "caption", "warning", "success", "error", "title", "subheader", "header"):
        parts.extend(str(e.value) for e in getattr(at, kind))
    return "\n".join(parts)


def rendered_text(at: AppTest) -> Tuple[str, Set[str]]:
    from google.protobuf import text_format

    elements: list = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            elements.append(child)
            walk(child)

    walk(at.main)
    walk(at.sidebar)
    text, labels = [], set()
    for el in elements:
        proto = getattr(el, "proto", None)
        if proto is not None and not type(proto).__name__.startswith(SKIP_PROTOS):
            dumped = text_format.MessageToString(proto, as_utf8=True)
            text.append(re.sub(r'id: "\$\$ID-[0-9a-f-]+[^"]*"', "", dumped))
            for field in ("label", "options", "default", "body"):
                value = getattr(proto, field, None)
                if isinstance(value, str):
                    labels.add(value.strip().lower())
                elif value is not None and not isinstance(value, (int, float, bool)):
                    try:
                        labels.update(str(v).strip().lower() for v in value if isinstance(v, str))
                    except TypeError:
                        pass
        frame = getattr(el, "value", None)
        if type(el).__name__ in ("Dataframe", "Table") and hasattr(frame, "columns"):
            labels.update(str(c).strip().lower() for c in frame.columns)
            text.append(" ".join(str(c) for c in frame.columns))
    return "\n".join(text), labels


def scan_hits(text: str, scan_re) -> List[str]:
    return sorted({m.group(0) for m in scan_re.finditer(text)})


def run_page(path, *, state=None, timeout: int = TIMEOUT) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=timeout)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    return at.run()


def page(root, number):
    from pathlib import Path
    if number == "main":
        return Path(root) / "app" / "main.py"
    [found] = sorted((Path(root) / "app" / "pages").glob(f"{number}_*.py"))
    return found


PAGE_NUMBERS = ["main", 0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12]
