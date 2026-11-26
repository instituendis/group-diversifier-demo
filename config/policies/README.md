# Example allocation policies

A **policy** is the complete rule set the allocation engine optimises:
attribute goals, per-team head counts, "no lone member" rules, numeric
bounds, shared values, keep-together/keep-apart groups, repeat pairings
across projects, student preferences, fairness and solver settings. It is
plain JSON (`app/engine/policy.py`, `Policy.to_dict()` /
`Policy.from_dict()`), stored on `Configuration.policy`, so it travels with
configuration exports and session files.

## The files

| File | What it is |
| --- | --- |
| `demo_fun.json` | "Demo — Fun mix", for the fictional roster `data/demo/demo_cohort.csv` (100 people, teams of 4). Hand-written: diversify favourite colour, ice cream and city (weight 2), hobby and pet (1.5), balance the average coffees per day (1), at most 1 chocolate fan per team (soft), never repeat a teammate (hard). It is the default policy of a demo session. See [README.md](../../README.md) for demo mode. |


## Use one

From the command line, the file is the `--policy` of the CLI (it also accepts
a CLI `--report` file or a configuration export):

```bash
python -m app.cli feasibility --data data/demo/demo_cohort.csv --policy config/policies/demo_fun.json
python -m app.cli allocate --data data/demo/demo_cohort.csv --policy config/policies/demo_fun.json \
    --projects 1 --team-size 4 --out exports/allocation.csv
```

In the app, the Allocation Rules page lists the files of this folder under
"Example rule sets" and loads the one you pick.

In Python:

```python
from pathlib import Path

from app.engine.policy import Policy

policy = Policy.from_json(Path("config/policies/demo_fun.json").read_text(encoding="utf-8"))
```

## Write your own

For another roster, generate a fresh suggestion from its columns instead of
editing a copy of these files:

```bash
python -m app.cli suggest-policy --data your.csv --out my_policy.json
python -m app.cli feasibility --data your.csv --policy my_policy.json
```

The attribute names in a policy are roster columns, so a policy only fits the
roster it was written for. Unknown keys are ignored and missing keys take
their defaults, so older or hand-written files keep loading.

## What each field sets

The fields and their defaults are defined in `app/engine/policy.py`.

| Field | What it sets |
| --- | --- |
| `name`, `description`, `version` | the policy's name, description and version |
| `attributes` | the goal per roster column: `balance`, `diversify`, `cluster`, `spread` or `ignore` |
| `counts` | head-count bounds per team, `mode` `any_of` or `each`, and `numeric_range` |
| `isolation` | "no lone member" |
| `numeric_bounds` | team mean and range limits |
| `shared` | teammates share at least k values |
| `pairs` | keep-together and keep-apart groups; `projects: null` means every generated project |
| `history` | repeat cap, contact weighting, `ignore_groups_larger_than` and section-mates |
| `preferences` | student prefer and avoid lists |
| `fairness` | total versus worst team |
| `solver` | `method`, `time_limit_s`, `restarts`, `seed`, `iterations`, `flexible_sizes`, `polish`, `joint_projects`, `mip_gap`, `workers`, `scale_time_limit` |

The presets of the Configuration page are in `config/presets/`. The rules
engine that scores the penalties is in `app/engine/`.
