#!/usr/bin/env python3
"""Write ``THIRD_PARTY_LICENSES.md`` from the installed package metadata.

The table lists every package pinned in ``requirements.lock.txt``: the set the
container image installs. Version and licence are read from the metadata of the
interpreter that runs this script, so run it with the project virtualenv that
holds exactly the locked versions (``uv pip sync requirements-dev.lock.txt``
puts them there; ``make lock`` only writes the lock files). A licence the metadata
does not declare is listed as "not determined"; it is never guessed.

Usage::

    python scripts/make_third_party_notices.py            # rewrite the file
    python scripts/make_third_party_notices.py --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock.txt"
NOTICE = ROOT / "THIRD_PARTY_LICENSES.md"
UNDETERMINED = "not determined"
PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")

HEADER = """\
# Third-party licences

Group Diversifier is AGPL-3.0-or-later (see `LICENSE`). The container image
installs the packages below, which keep their own licences. The list is every
package pinned in `requirements.lock.txt`, with the version the lock pins and
the licence each package declares in its installed metadata. The table is
generated from that metadata by `scripts/make_third_party_notices.py`, not
written by hand.

A licence that the metadata does not declare is listed as "not determined".
Check that package's own project before relying on its licence.

| Package | Version | Declared licence | Project |
|---|---|---|---|
"""

FOOTER = """
The development lock adds pytest and ruff for development. They are not in the
image and are not listed.
"""


def pins(path: Path) -> list[tuple[str, str]]:
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN_RE.match(line)
        if m:
            found.append((m.group(1), m.group(2)))
    return found


def declared_licence(meta) -> str:
    expr = (meta.get("License-Expression") or "").strip()
    if expr:
        return expr
    lic = (meta.get("License") or "").strip()
    if lic and len(lic) < 120 and "\n" not in lic:
        return lic
    classifiers = sorted({
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    })
    return "; ".join(classifiers) or UNDETERMINED


def _repo_root(url: str) -> str:
    parts = url.split("/")
    if url.startswith("https://github.com/") and len(parts) > 5:
        return "/".join(parts[:5])
    return url


def project_url(meta) -> str:
    home = (meta.get("Home-page") or "").strip()
    if home:
        return _repo_root(home)
    for value in meta.get_all("Project-URL") or []:
        return _repo_root(value.split(",", 1)[-1].strip())
    return ""


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render() -> str:
    rows = []
    problems = []
    for name, version in pins(LOCK):
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            problems.append(f"not installed: {name}=={version}")
            continue
        if dist.version != version:
            problems.append(f"{name} is {dist.version} here, the lock pins {version}")
            continue
        meta = dist.metadata
        rows.append((name, version, declared_licence(meta), project_url(meta)))
    if problems:
        raise SystemExit(
            "make_third_party_notices: run it with the project virtualenv that holds "
            "the locked versions; " + "; ".join(problems)
        )
    rows.sort(key=lambda r: r[0].lower())
    table = "".join(
        f"| {cell(n)} | {cell(v)} | {cell(lic)} | {cell(url)} |\n"
        for n, v, lic, url in rows
    )
    return HEADER + table + FOOTER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="fail if THIRD_PARTY_LICENSES.md differs from what would be written",
    )
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = NOTICE.read_text(encoding="utf-8") if NOTICE.exists() else ""
        if current != text:
            print(f"{NOTICE.name} is stale: run scripts/make_third_party_notices.py",
                  file=sys.stderr)
            return 1
        print(f"{NOTICE.name} is up to date")
        return 0
    NOTICE.write_text(text, encoding="utf-8")
    print(f"wrote {NOTICE.name}: {text.count(chr(10) + '| ') - 1} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
