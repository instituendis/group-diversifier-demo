# Group Diversifier — demo

Group Diversifier forms **diverse, balanced project teams** across a
sequence of projects, keeping earlier teammates apart. This repository is its
**demo**: the app runs in demo mode on a cohort of 100 entirely fictional
people (favourite colour, ice cream, city, hobbies, pet, coffees per day),
with playful allocation rules and one earlier project already loaded.

## Run it

```bash
./run.sh                    # creates .venv, installs requirements, starts the app (run.bat on Windows)
podman compose up --build   # or in a container
```

Then open http://localhost:8501 (the app listens on this computer only). The
sidebar shows "🎈 Demo mode — the bundled sample data is fictional". You can
also upload your own CSV: its columns become attributes, except the id,
preference and earlier-group columns (the user guide lists them).

Run it from a checkout like this (or `pip install -e .`): the demo marker, the
profiles in `config/` and the data in `data/demo/` live next to `app/`, and a
plain wheel install does not carry them.

## What to try

1. **Data Overview**: the fictional roster and its attributes.
2. **Allocation Rules**: the rules (spread colours, cities, hobbies and
   pets; balance coffees; at most one Chocolate fan per team; never repeat
   a teammate). Try a preset or "✨ Suggest from data".
3. **Team Formation**: form the next projects' teams.
4. **Team Editor**, **Analytics**, **Cross Project**, **Export**.

The [user guide](USER_GUIDE.md) walks through every page in order, from the
roster to the exported files.

`data/demo/` holds the roster, the earlier project and small sample files for
the other uploads; `scripts/make_demo_cohort.py` regenerates the roster.

Repeat avoidance is a rule like the others: the rules set how often a pair may
share a team, and whether that cap is hard. Team Formation and the CLI report
the pairs over the cap (`--strict` makes the CLI fail when a hard rule is
still broken), so check that report before you use an allocation.

`app/extensions.py` is an extension point: other code can install hooks there
that change defaults, columns and wording. This repository ships no extension,
so every hook returns its default.

## Tests

```bash
uv venv --python 3.13 && uv pip sync --require-hashes requirements-dev.lock.txt
.venv/bin/python -m pytest      # the smoke suite in tests/: pages, CLI, exports, presets
```

The suite renders every page, runs the CLI and the classic algorithms on the
demo data and checks the exported files. It takes a few minutes.

## Licence

GNU Affero General Public License v3 or later (`LICENSE`, `NOTICE`).
`THIRD_PARTY_LICENSES.md` lists the licence each third-party package of the
container image declares.

This repository is a generated export: every change is made upstream and
exported here, so pull requests here are not merged. Bug reports are welcome
as issues; report a vulnerability privately (see [SECURITY.md](SECURITY.md)).
