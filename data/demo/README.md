# Demo data

Everything in this folder is **entirely fictional**: the 100 "students" are
made-up rows with playful attributes, not real people. It belongs to the demo
mode of the app (see [README.md](../../README.md)).

| File | What it is |
| --- | --- |
| `demo_cohort.csv` | 100 people (made by [`scripts/make_demo_cohort.py`](../../scripts/make_demo_cohort.py), seed 2026, byte for byte): `identifier`, `favourite_colour`, `ice_cream`, `city`, `hobbies` (several values separated by `\|`), `pet` (the literal `None` is a category) and `coffees_per_day` (a whole number from 0 to 5). |
| `demo_prior_projects.csv` | One earlier project: the 100 people in 25 teams of 4, in the app's import format (`team_id,project_id,student_id`). Import it on Team Formation to see "never repeat a teammate" at work. |
| `demo_participants.csv` | Who takes part in projects 2 and 3 (`project_id,student_id`): people 97-100 sit out project 2. Upload it on Team Formation. |
| `demo_preferences.csv` | A few prefer/avoid requests (`student_id,prefer_teammates,avoid_teammates`, several ids separated by `\|`). Upload it on Data Overview. |
| `demo_reference_groups.csv` | A baseline for Compare Runs: project 1 is the earlier project above, then projects 2 and 3 split by hand: project 2 in id order (1-4, 5-8, ...), project 3 by counting off (1, 26, 51, 76, ...). |

The three small files above were written by hand. `demo_prior_projects.csv` was made with the command-line tool, not by hand,
from the demo roster and the demo rules
([config/policies/demo_fun.json](../../config/policies/demo_fun.json)):

```bash
python -m app.cli allocate --data data/demo/demo_cohort.csv \
    --policy config/policies/demo_fun.json --projects 1 --team-size 4 \
    --iterations 3000 --restarts 1 --seed 42 --out data/demo/demo_prior_projects.csv
```
