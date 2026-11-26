# Group Diversifier demo: user guide

This guide takes you through the demo app from start to finish: check the
roster, set the rules, form the teams, change them by hand, review the result
and export it. It describes the app as it behaves in demo mode. Every person,
value and number in the demo is made up.

## 1. What the demo does

Group Diversifier forms diverse, balanced project teams across a sequence of
projects. Across those projects it can stop the same two people from sharing a
team twice.

The demo runs the same app on a fictional roster of 100 people. Each one has a
favourite colour, an ice cream, a city, one or two hobbies, a pet and a number
of coffees a day. The rules are playful, and the app starts with one earlier
project already loaded, so you can see the "never repeat a teammate" rule at
work straight away. Three projects are planned in total: the earlier one, and
two that the app forms.

Demo mode changes only the starting point: the roster, the rules, the team size
and the number of projects. Every page, the rules engine, the Team Editor, the
exports and the session files work as they do in the full app.

Please do not upload real people's details to a shared demo. Uploads stay on the
server, in its memory or its temporary files, until the server restarts, so a
demo is not the place for real records.

## 2. Opening the app

Start the app as the README describes, then open http://localhost:8501 in a
browser. If someone has given you a hosted demo address, open that instead.

The sidebar shows the banner "🎈 Demo mode — the bundled sample data is
fictional" when demo mode is on. It speaks of the files that come with the app:
a roster you upload is your own data, and the banner says nothing about it. If the banner is missing, start the app again with the environment
variable `GROUP_DIVERSIFIER_DEMO` set to `1`.

## 3. The fictional roster

The roster is the file `data/demo/demo_cohort.csv`. Its columns are:

| Column | Values |
| --- | --- |
| `identifier` | a number for each person |
| `favourite_colour` | Blue, Green, Purple, Red or Yellow |
| `ice_cream` | Chocolate, Mango, Pistachio, Strawberry or Vanilla |
| `city` | Kyoto, Lima, Lisbon, Nairobi, Oslo or Toronto |
| `hobbies` | one or two of Baking, Chess, Gaming, Hiking, Karaoke, Painting or Surfing (the profile allows up to three) |
| `pet` | Cat, Dog, Fish, None or Parrot |
| `coffees_per_day` | a whole number from 0 to 5 |

The columns other than the identifier are attributes that the rules can act on.
The earlier project, in `data/demo/demo_prior_projects.csv`, puts the 100 people
in 25 teams of 4.

## 4. The pages

The sidebar groups the pages. **Start** holds Home. **Setup** holds Data
Overview, Configuration and Allocation Rules. **Allocate** holds Team Formation
and Team Editor. **Review** holds Analytics, Cross-Project and Compare Runs.
**Share** holds Export. **Advanced** holds Profile Manager.

The first time through, follow the order above. Home shows a checklist of the
steps and the state of each one, and it lists the run history.

### Data Overview

The **Student Table** tab lists the roster. **Attribute Insights** shows how the
app reads each column and which attributes overlap. **Distributions** shows how
often each value occurs. **Edit Preferences** changes one person's preferences.
Those edits last for this browser session, until the roster changes.

To use your own roster, upload a CSV or Excel file (`.csv`, `.xlsx` or `.xlsm`)
at the top of the page. Every column becomes an attribute except the reserved
input columns: the id (`student_id` or `identifier`), `prefer_teammates`,
`avoid_teammates`, `group_number`, `team_id`, `project_id` and columns that start
with `team_project_`. **Restore default
dataset** brings the fictional roster back. The Data Validation section at the
bottom lists warnings about the roster.

### Configuration

Set the team sizes (the demo starts with 3 to 5 people per team, aiming for 4),
the number of projects (the demo starts with 3, the earlier one included) and
the search settings. The bundled presets (under **Import / Presets**) set the
weights of the classic algorithms only; the rules engine does not read them.
The description of the chosen preset appears under the selector:

- **Equal weights**: every demo attribute weighs the same.
- **Colour and city first**: favourite colour and city weigh most, then ice
  cream and hobbies, then pet and coffees.
- **Mirror city shares**: the demo weights, plus proportional city
  representation, so that each team mirrors the city mix of the roster.

This guide follows the rules engine, which Team Formation runs with the rules
you set on Allocation Rules.

### Allocation Rules

This page says what a good allocation looks like. The demo starts with its
playful rule set:

- favourite colour, ice cream, city, hobbies and pet are spread across the
  teams, so that teammates rarely share a value (hobbies and pet count a little
  less than the others);
- coffees per day are balanced, so that every team has about the same average;
- at most one Chocolate fan per team (a soft rule: the engine tries to keep it,
  and the violations list names any team where it fails);
- never repeat a teammate (a hard rule): the solver keeps two people who shared
  a team before apart. When the team sizes or the other rules leave no way to
  do that, it still returns teams and lists the repeated pairs; Team Formation
  shows them, and the CLI's `--strict` fails the run while a hard rule is broken.

The page has these parts:

- **Start from a preset**, or pick an example rule set and load it.
- **Import / export**: rules as a JSON file, so you can keep and reuse them.
- **The attribute table**, with one row per roster column. Choose a goal and a
  weight from 0 to 10, or untick **On** to switch the rule off. The goals are
  *balance* (each team mirrors the mix of the whole roster, or, for numbers,
  has equal averages), *diversify* (as few teammates as possible share a
  value), *spread* (for numbers: equal averages and a range inside each team
  like the roster's), *cluster* (similar students together) and *ignore*.
- **Head-count rules**: the minimum or maximum number of students with a given
  value in each team.
- **No lone members**: a team that has a student with a chosen value must have
  at least a set number of students with that value.
- **Numeric bounds**: limits on the average of a numeric attribute in each team.
- **Shared values**: teammates must share at least a set number of values of a
  multi-value attribute such as hobbies.
- **Keep together / keep apart**: named groups that must share a team, or must
  not.
- **Repeat pairings across projects**: whether the project history counts, and
  an optional cap on how often a pair may share a team.

**✨ Suggest from data** proposes a starting rule set from the roster.

### Team Formation

Press **Run Team Formation**. The button shows an estimate of the time. The app
forms every project that is not locked history, then shows:

- the **Results Summary**, with the score of the run (engine scores are penalty
  points, so lower is better) and a badge per project that says whether all the
  rules are met;
- the teams of each project;
- **Constraint Violations**, which lists each rule that is not met and where;
- the **Validation Summary**.

**Generate alternatives** produces several good allocations of one project that
differ noticeably from each other. They keep to the history of the other
projects. **Clear Results** discards the run.

The **Import prior projects (locked history)** section takes team assignments
of earlier projects, in the format of **Download sample earlier groups**. The
demo has already loaded its earlier project. **Clear imported history** removes
it.

### Team Editor

After a run, change the teams by hand. The tabs are:

- **Move / swap**: choose the student or students, and the page shows the effect
  before anything changes.
- **Suggestions**: **Suggest improvements** proposes changes that lower the
  score.
- **Re-optimise**: a full search for one project, which leaves locked students
  where they are.
- **Locks & teams**: lock students in place, or release the locks.

If students have left or joined the roster, **Repair automatically** fixes the
teams. You can also fix them by hand. Edits are kept in the run history, so they
can be compared later.

### Review: Analytics, Cross-Project and Compare Runs

**Analytics** shows the diversity and fairness of the result: whether any
student is consistently in low- or high-diversity teams, and who works with whom
across the projects.

**Cross-Project** shows which team each student is in, in every project, and
which pairs have already been teammates.

**Compare Runs** lists every run in the session: formation runs, saved edits and
imported runs. Pick two to four to compare them side by side, with the headline
numbers, rule compliance and how similar the allocations are project by project.
You can also score an allocation made elsewhere, given as a long CSV file, with
the current rules.

### Export

The **Export** page offers these downloads:

- **CSV Export**: the team assignments in one long file, which can be imported
  back into the app.
- **Printable Report**: one self-contained HTML file. Open it in a browser; it
  needs no internet connection.
- **LMS Groups**: files that create the teams as groups in a learning platform.
- **Roster & Composition**: a wide roster (one row per person, with the team in
  every project) and a table of the head counts per team.
- **Rule Report**: how each rule was met, as CSV and JSON.
- **Excel Export**: the complete results in one workbook with several sheets.
- **JSON**: a snapshot of the configuration.

## 5. Things to try

1. Run Team Formation with the defaults. Read the Results Summary and the
   Constraint Violations.
2. On Allocation Rules, choose another rule preset (for example "Maximum mix")
   and run again. Compare the two runs on Compare Runs.
3. On Allocation Rules, raise the weight of city to 10, or untick the pet rule,
   and run again.
4. Generate alternatives for project 2 and see how different they are.
5. In Team Editor, pick a student and move them to another team. Read the
   preview before you apply it.
6. Export the printable report and open it in a browser.
7. Upload a short CSV of invented rows, with a column for each attribute you
   want to try, and check the attribute insights on Data Overview.

## 6. Saving your work

The sidebar has **Download session**, which saves a JSON snapshot of the whole
session: the roster, the rules, the runs and the edits. **Restore session**
loads a snapshot back. The app saves no sessions on the server, so a page reload
starts a fresh session. Download the session first if you want to keep your
work.

## 7. Profiles

The **Advanced** section holds the Profile Manager. Profiles are the dimension
sets that the older classic algorithms and the legacy diversity metrics use. The
rules engine does not read them, so you can leave them alone. In demo mode the
profile is called "demo" and holds the six attributes in section 3.

## 8. Licence and source

The app is free software under the GNU Affero General Public License, version 3
or later. The licence text is in the file `LICENSE`. The footer links to the
source of the running version; in the demo, that is the demo repository.

## 9. Hosting it for others

The app has no login, no TLS and no rate limits of its own. For one person on
one computer, use `compose.yaml` or `run.sh`; both listen on this computer only
(127.0.0.1). To serve other people, run the
container in hosted mode behind a reverse proxy that adds those. `deploy/`
holds three example files. They are examples, not turnkey files: read each one
before you use it.

| File | What it is |
| --- | --- |
| `deploy/compose.hosted.yaml` | The container in hosted mode, with Podman or Docker Compose |
| `deploy/quadlet/group-diversifier.container` | The same container as a Podman Quadlet (systemd) unit |
| `deploy/nginx/group-diversifier.conf.example` | nginx in front: login, TLS, rate limits, cache headers |

### What to change

* **The image tag**: `image:` (compose) or `Image=` (Quadlet), the tag you
  built.
* **`STREAMLIT_BROWSER_SERVER_ADDRESS`**: the public host name users type, with
  no scheme and no port.
* **`GROUP_DIVERSIFIER_SOURCE_URL`**: required if you run a modified version or
  your users cannot open the default link. It is the footer's "Source code"
  link, the source offer of AGPL-3.0 section 13.
* **`GROUP_DIVERSIFIER_PRIVACY_URL` / `GROUP_DIVERSIFIER_PRIVACY_NOTICE`**: your
  privacy statement (recommended, both optional). Empty keeps the built-in
  notice; the URL must be http(s).
* **Memory and upload size**: `mem_limit` (compose) or `Memory=` (Quadlet), and
  `STREAMLIT_SERVER_MAX_UPLOAD_SIZE`. To give the container less memory, lower
  the upload cap too (see "Memory" below).
* **In the nginx file**: your host name (twice, plus the certificate paths),
  `ssl_certificate` and `ssl_certificate_key`, the password file, the port the
  container is published on, `client_max_body_size` (a little above
  `STREAMLIT_SERVER_MAX_UPLOAD_SIZE`) and the rate limits (keep the burst above
  the number of files a browser fetches at once when it opens the app).

### Build and start

Build the image with the tag the example files name: `group-diversifier:`
followed by the version in `app/__init__.py` (written `VERSION` below).

```bash
podman build --format docker -t group-diversifier:VERSION -f Containerfile .
podman compose -f deploy/compose.hosted.yaml up -d
```

With Quadlet instead (rootless, as the user that owns the service):

```bash
cp deploy/quadlet/group-diversifier.container ~/.config/containers/systemd/
systemctl --user daemon-reload
systemctl --user start group-diversifier.service
loginctl enable-linger "$USER"      # keep the service running after logout
```

For a system-wide unit use `/etc/containers/systemd/` and `systemctl` without
`--user`. `/usr/libexec/podman/quadlet -dryrun -user` shows what Quadlet
generates.

nginx: install the example as `/etc/nginx/conf.d/group-diversifier.conf` (it is
included inside the `http {}` block). Create the password file with
`htpasswd -B -c /etc/nginx/group-diversifier.htpasswd <user>`. Uncomment the
`Strict-Transport-Security` header once HTTPS works for good.

### Caps

Hosted mode caps workers, concurrent runs, the solver time limit (seconds per
project, before it is scaled to the roster size), the classic algorithms' time
and the largest table file read (roster or import, in MB). To change one, add
its variable to the `environment:` section of `compose.hosted.yaml` or as an
`Environment=` line of the Quadlet unit:

| Variable | Hosted default |
| --- | --- |
| `GROUP_DIVERSIFIER_MAX_WORKERS` | 1 |
| `GROUP_DIVERSIFIER_MAX_CONCURRENT_RUNS` | 1 |
| `GROUP_DIVERSIFIER_MAX_TIME_LIMIT` | 120 (seconds per project) |
| `GROUP_DIVERSIFIER_MAX_CLASSIC_TIME` | 120 (seconds per project) |
| `GROUP_DIVERSIFIER_MAX_TABLE_UPLOAD_MB` | 5 (MB) |

### Restart on a schedule

Uploaded data lives only in the process's memory, but Streamlit does not end
the session of a closed tab, so the data stays there until the process
restarts. Restart the container on a schedule (for example nightly, with
`podman restart group-diversifier` from cron or a systemd timer), or stop it
when nobody uses it. A user timer for the Quadlet unit, next to it in
`~/.config/systemd/user/`:

```ini
# group-diversifier-restart.service
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl --user restart group-diversifier.service

# group-diversifier-restart.timer
[Timer]
OnCalendar=*-*-* 04:00:00
Persistent=true
[Install]
WantedBy=timers.target
```

Then: `systemctl --user enable --now group-diversifier-restart.timer`.

### Memory

Uploads stay in the server's memory, and the `/tmp` buffer is RAM, so the
memory a server needs grows with the upload cap and the number of sessions
since the last restart. The examples' 1.5 GB limit is a starting point, not a
measured requirement: watch the container's memory on your own host with your
own rosters. To give it less memory, lower `STREAMLIT_SERVER_MAX_UPLOAD_SIZE`
along with the memory limit; that also lowers the largest session file that can
be restored.
