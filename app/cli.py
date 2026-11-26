from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from app import __version__  # noqa: E402
from app import extensions  # noqa: E402
from app.core.allocation_similarity import allocation_similarity  # noqa: E402
from app.core.ids import trailing_number_key  # noqa: E402
from app.core.assignment_import import (  # noqa: E402
    AssignmentImport,
    ParticipantSpec,
    parse_assignments_csv,
    parse_participants_csv,
    validate_against_roster,
    validate_participants,
)
from app.core.config_loader import ConfigurationLoader  # noqa: E402
from app.core.data_loader import DataLoader  # noqa: E402
from app.core.engine_session import (  # noqa: E402
    attribute_kinds,
    project_specs,
    run_engine,
    section_reunions,
    summarize_reports,
)
from app.core.models import Configuration  # noqa: E402
from app.core.policy_files import (  # noqa: E402
    PolicyFileError,
    numbers_as_text,
    read_policy_text,
)
from app.engine.attributes import attribute_overview, count_rule_targets, redundant_pairs  # noqa: E402
from app.engine.defaults import PRESETS, suggest_policy  # noqa: E402
from app.engine.feasibility import FeasibilityReport, check_feasibility  # noqa: E402
from app.engine.history import PriorProject  # noqa: E402
from app.engine.policy import SOLVER_METHODS, Policy  # noqa: E402
from app.engine.policy_text import describe_policy  # noqa: E402
from app.engine.report_views import (  # noqa: E402
    history_section,
    overall,
    preferences_section,
    stats_view,
    teams,
)
from app.engine.roster import build_roster_frame  # noqa: E402
from app.engine.runner import ProjectSpec, evaluate_allocation  # noqa: E402
from app.engine.sizing import (  # noqa: E402
    all_sizes_listed,
    describe_sizes,
    size_problems,
)
from app.utils.exporters import (  # noqa: E402
    ALLOCATION_COLUMNS,
    allocation_long_rows,
    report_envelope,
)
from app.utils.fmt import fmt  # noqa: E402
from app.utils.headline import SECTION_MATES_HELP, headline_rows, repeat_cap_text  # noqa: E402
from app.utils.serialization import json_safe  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

CONFIG_DIR_ENV = "GROUP_DIVERSIFIER_CONFIG_DIR"
_GAP_ERROR_PREFIX = "Imported projects must form a contiguous prefix"
_REPORT_FORMAT = "group-diversifier-cli-report"
_EXCEL_SUFFIXES = (".xlsx", ".xlsm")
_TARGET_ROWS = 8


class CliError(ValueError): ...


def _say(text: str = "") -> None:
    print(text, file=sys.stdout)


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _note(args: argparse.Namespace, text: str) -> None:
    if not getattr(args, "quiet", False):
        _err(text)


def _warn(args: argparse.Namespace, text: str) -> None:
    if not getattr(args, "quiet", False):
        _err(_wrap(f"warning: {text}", indent="  "))


def _width() -> int:
    return max(60, min(120, shutil.get_terminal_size((100, 24)).columns))


_KEEP_TOGETHER = re.compile(r"\d+ x \d+")


def _wrap(text: str, indent: str = "    ", first: str = "") -> str:
    text = _KEEP_TOGETHER.sub(lambda m: m.group(0).replace(" ", "\u00a0"), text)
    wrapped = textwrap.fill(text, width=_width(), initial_indent=first,
                            subsequent_indent=indent, break_long_words=False,
                            break_on_hyphens=False)
    return wrapped.replace("\u00a0", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
           right: Iterable[int] = ()) -> str:
    right = set(right)
    head = [str(h).split("\n") for h in headers]
    depth = max((len(h) for h in head), default=1)
    head = [[""] * (depth - len(h)) + h for h in head]
    body = [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(text) for text in head[k] + [row[k] for row in body])
              for k in range(len(headers))]

    def line(row: Sequence[str]) -> str:
        parts = [cell.rjust(widths[k]) if k in right else cell.ljust(widths[k])
                 for k, cell in enumerate(row)]
        return "  ".join(parts).rstrip()

    rule = "  ".join("-" * w for w in widths)
    titles = [line([head[k][d] for k in range(len(headers))]) for d in range(depth)]
    return "\n".join([*titles, rule, *(line(r) for r in body)])


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return fmt(value, digits=digits, missing="n/a")


def _fmt_quality(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def _fmt_sizes(sizes: Sequence[int]) -> str:
    counts: Dict[int, int] = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    return " + ".join(f"{c}x{s}" for s, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _team_size(text: str) -> int:
    value = _positive_int(text)
    if value < 2:
        raise argparse.ArgumentTypeError(f"a team needs at least 2 members, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text!r}")
    return value


def _seed(text: str) -> Optional[int]:
    if text.strip().lower() in ("none", "random", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a whole number or 'none', got {text!r}") from None


@dataclass
class Roster:
    path: Path
    students: List[Any]
    raw: pd.DataFrame
    frame: pd.DataFrame
    warnings: List[str] = field(default_factory=list)

    @property
    def ids(self) -> List[str]:
        return [str(sid) for sid in self.frame["student_id"].tolist()]


def _config_dir(override: Optional[str]) -> Path:
    path = Path(override) if override else Path(os.environ.get(CONFIG_DIR_ENV) or ROOT_DIR / "config")
    if not path.is_dir():
        raise CliError(f"profile directory not found: {path} (see --config-dir)")
    return path


def load_roster(path: str, sheet: Optional[str] = None, *, config_dir: Optional[str] = None) -> Roster:
    source = Path(path)
    if not source.is_file():
        raise CliError(f"data file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".xls":
        raise CliError(f"{source.name}: legacy .xls workbooks are not supported; "
                       "save the sheet as .xlsx or .csv")
    if sheet and suffix not in _EXCEL_SUFFIXES:
        raise CliError("--sheet only applies to Excel workbooks (.xlsx/.xlsm)")
    loader = DataLoader(config_loader=ConfigurationLoader(_config_dir(config_dir)),
                        profile_name="default")
    try:
        if suffix in _EXCEL_SUFFIXES:
            students = loader.load_excel(source, sheet_name=sheet)
        else:
            students = loader.load_csv(source)
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        if suffix in _EXCEL_SUFFIXES and sheet:
            try:
                message += f" (sheets: {', '.join(DataLoader.excel_sheet_names(source))})"
            except Exception:
                pass
        raise CliError(f"cannot load {source.name}: {message}") from None
    if not students:
        raise CliError(f"{source.name}: no students found")
    frame = build_roster_frame(students, loader.get_dataframe())
    return Roster(path=source, students=students, raw=loader.get_dataframe(), frame=frame,
                  warnings=list(loader.warnings))


def _roster_from_args(args: argparse.Namespace) -> Roster:
    roster = load_roster(args.data, args.sheet, config_dir=args.config_dir)
    for warning in roster.warnings:
        _warn(args, f"{roster.path.name}: {warning}")
    return roster


def _read_text(path: str, what: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise CliError(f"{what} not found: {source}")
    try:
        return source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise CliError(f"cannot read {what} {source}: {exc}") from None


def _read_bytes(path: str, what: str) -> bytes:
    source = Path(path)
    if not source.is_file():
        raise CliError(f"{what} not found: {source}")
    try:
        return source.read_bytes()
    except OSError as exc:
        raise CliError(f"cannot read {what} {source}: {exc}") from None


def load_policy_file(path: str) -> Policy:
    text = _read_text(path, "policy file")
    name = Path(path).name
    try:
        loaded = read_policy_text(text)
    except PolicyFileError as exc:
        raise CliError(f"{name}: {exc}") from None
    if loaded.problems:
        shown = loaded.problems[:8] + ([f"... and {len(loaded.problems) - 8} more"]
                                       if len(loaded.problems) > 8 else [])
        raise CliError(f"{name}: not a valid policy:\n  - " + "\n  - ".join(shown))
    try:
        return Policy.from_dict(numbers_as_text(loaded.policy))
    except Exception as exc:
        raise CliError(f"{name}: not a valid policy ({exc})") from None


def _apply_solver_overrides(policy: Policy, args: argparse.Namespace) -> None:
    solver = policy.solver
    if getattr(args, "method", None):
        solver.method = args.method
    if getattr(args, "time_limit", None) is not None:
        solver.time_limit_s = float(args.time_limit)
    if getattr(args, "restarts", None) is not None:
        solver.restarts = int(args.restarts)
    if getattr(args, "seed", "keep") != "keep":
        solver.seed = args.seed
    if getattr(args, "workers", None) is not None:
        solver.workers = int(args.workers)
    if getattr(args, "iterations", None) is not None:
        solver.iterations = int(args.iterations)
    if getattr(args, "joint_projects", False):
        solver.joint_projects = True


def resolve_policy(args: argparse.Namespace, roster: Roster) -> Policy:
    if getattr(args, "policy", None):
        policy = load_policy_file(args.policy)
    else:
        preset = getattr(args, "preset", None) or "balanced"
        policy = suggest_policy(roster.frame, preset=preset, team_size=args.team_size,
                                num_teams=args.num_teams or None)
    _apply_solver_overrides(policy, args)
    try:
        problems = policy.validate(attribute_kinds(roster.frame))
    except (TypeError, ValueError) as exc:
        raise CliError(f"the policy has a field of the wrong type ({exc})") from None
    if problems:
        raise CliError("the policy does not fit this roster:\n  - " + "\n  - ".join(problems))
    return policy


def load_assignments(path: str, roster: Roster, *, allow_gaps: bool,
                     what: str = "allocation file") -> Tuple[AssignmentImport, List[str]]:
    imp, errors = parse_assignments_csv(_read_bytes(path, what))
    notes: List[str] = []
    if allow_gaps:
        gaps = [e for e in errors if e.startswith(_GAP_ERROR_PREFIX)]
        errors = [e for e in errors if not e.startswith(_GAP_ERROR_PREFIX)]
        if gaps and not errors:
            notes.append(f"{Path(path).name}: project ids {', '.join(imp.imported_project_ids)} "
                         "are not 1..K; fine for evaluation, but the app can only continue "
                         "from a 1..K timeline")
    if errors:
        shown = "; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else "")
        raise CliError(f"{Path(path).name}: {shown}")
    validate_against_roster(imp, set(roster.ids))
    notes.extend(f"{Path(path).name}: {w}" for w in imp.warnings)
    return imp, notes


def plain_projects(imp: AssignmentImport, *, imported: bool) -> List[Dict[str, Any]]:
    out = []
    for pid in imp.imported_project_ids:
        teams = imp.teams_by_project.get(pid, {})
        names = sorted(teams, key=trailing_number_key)
        out.append({"project_id": str(pid), "name": f"Project {pid}", "imported": imported,
                    "teams": [list(teams[name]) for name in names], "team_names": names})
    return out


def write_allocation_csv(path: str, projects: Sequence[Dict[str, Any]]) -> int:
    target = Path(path)
    rows = allocation_long_rows(projects)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(ALLOCATION_COLUMNS)
            for row in rows:
                writer.writerow([row["team_id"], row["project_id"], row["student_id"]])
    except OSError as exc:
        raise CliError(f"cannot write {target}: {exc}") from None
    return len(rows)


def check_output_paths(args: argparse.Namespace,
                       options: Sequence[Tuple[str, str]]) -> None:
    seen: Dict[Path, str] = {}
    for attr, what in options:
        value = getattr(args, attr, None)
        if not value:
            continue
        target = Path(value)
        flag = "--" + attr.replace("_", "-")
        key = target.resolve()
        if key in seen:
            raise CliError(f"{seen[key]} and {flag} both point to {target}; give each output "
                           "its own file")
        seen[key] = flag
        if target.is_dir():
            raise CliError(f"cannot write {what} {target}: it is a directory")
        existed = target.exists()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8"):
                pass
        except OSError as exc:
            raise CliError(f"cannot write {what} {target}: {exc}") from None
        if not existed:
            try:
                target.unlink()
            except OSError:
                pass


def _write_text(path: str, text: str, what: str) -> None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot write {what} {target}: {exc}") from None


def _write_json(path: str, data: Dict[str, Any], what: str = "report") -> None:
    _write_text(path, json.dumps(json_safe(data), indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n", what)


@dataclass
class Plan:
    config: Configuration
    history: List[Dict[str, Any]]
    specs: List[ProjectSpec]
    total_projects: int
    participants: Optional[ParticipantSpec] = None
    notes: List[str] = field(default_factory=list)

    @property
    def prior(self) -> List[PriorProject]:
        return [PriorProject(p["project_id"], p["teams"]) for p in self.history]


def pool_text(roster: "Roster", specs: Sequence[ProjectSpec], bracket: bool = True) -> str:
    n = len(roster.frame)
    pools = [(spec.project_id, len(spec.student_ids)) for spec in specs]
    if all(size == n for _, size in pools):
        return f"{n} students"
    listed = "participants: " + ", ".join(f"{size} in project {pid}" for pid, size in pools)
    return f"{n} students ({listed})" if bracket else f"{n} students, {listed}"


def _size_settings(args: argparse.Namespace) -> Tuple[int, int, int, int]:
    target = int(args.team_size)
    lo = int(args.min_size) if args.min_size is not None else max(2, target - 1)
    hi = int(args.max_size) if args.max_size is not None else target + 1
    if lo > hi:
        raise CliError(f"--min-size {lo} is larger than --max-size {hi}")
    if not lo <= target <= hi:
        raise CliError(f"--team-size {target} must lie between --min-size {lo} and --max-size {hi}")
    return lo, hi, target, int(args.num_teams or 0)


def build_plan(args: argparse.Namespace, roster: Roster) -> Plan:
    history: List[Dict[str, Any]] = []
    notes: List[str] = []
    if getattr(args, "history", None):
        imp, load_notes = load_assignments(args.history, roster, allow_gaps=False,
                                           what="history file")
        history = plain_projects(imp, imported=True)
        notes.extend(load_notes)
    k = len(history)
    total = int(args.projects) if args.projects is not None else k + 1
    if total <= k:
        raise CliError(f"--projects {total}: the history already holds projects 1..{k}; "
                       f"ask for at least {k + 1} projects")
    participants = None
    if getattr(args, "participants", None):
        participants, errors = parse_participants_csv(_read_bytes(args.participants,
                                                                  "participants file"))
        if errors:
            raise CliError(f"{Path(args.participants).name}: " + "; ".join(errors[:5]))
        validate_participants(participants, set(roster.ids), k, total)
        notes.extend(f"{Path(args.participants).name}: {w}" for w in participants.warnings)
    lo, hi, target, num_teams = _size_settings(args)
    config = Configuration(team_size_min=lo, team_size_max=hi, team_size_target=target,
                           num_teams=num_teams, num_projects=total)
    specs = project_specs(config, roster.ids, participants=participants,
                          imported_count=k, name_teams=True)
    return Plan(config=config, history=history, specs=specs, total_projects=total,
                participants=participants, notes=notes)


def evaluate_plain(roster: Roster, policy: Policy, projects: Sequence[Dict[str, Any]],
                   size_settings: Tuple[int, int, int, int],
                   only: Optional[Set[str]] = None
                   ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    lo, hi, target, _ = size_settings
    reports: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    history: List[PriorProject] = []
    for entry in projects:
        pid = str(entry["project_id"])
        teams = [list(t) for t in entry.get("teams", []) if t]
        skip = entry.get("imported") or (only is not None and pid not in only)
        if not skip and not teams:
            warnings.append(f"Project {pid}: no team has students from the roster; not evaluated")
            skip = True
        if not skip:
            pool = [sid for team in teams for sid in team]
            spec = ProjectSpec(project_id=pid, student_ids=pool,
                               team_sizes=[len(t) for t in teams], size_min=lo, size_max=hi,
                               size_target=target, team_names=list(entry.get("team_names") or []))
            try:
                result = evaluate_allocation(roster.frame, policy, spec, teams,
                                             prior_projects=history)
            except ValueError as exc:
                raise CliError(str(exc)) from None
            reports[pid] = result.report
            warnings.extend(w for w in result.warnings if w not in warnings)
        history.append(PriorProject(pid, teams))
    return reports, warnings


def _id_list(ids: Sequence[str], limit: int = 10) -> str:
    shown = sorted(ids, key=trailing_number_key)
    return ", ".join(shown[:limit]) + (f" … (+{len(shown) - limit} more)"
                                       if len(shown) > limit else "")


def coverage_problems(roster: Roster, projects: Sequence[Dict[str, Any]],
                      size_settings: Tuple[int, int, int, int],
                      only: Optional[Set[str]] = None,
                      participants: Optional[ParticipantSpec] = None) -> List[str]:
    lo, hi, _, _ = size_settings
    problems: List[str] = []
    for entry in projects:
        pid = str(entry["project_id"])
        teams = [list(t) for t in entry.get("teams", []) if t]
        if entry.get("imported") or (only is not None and pid not in only):
            continue
        where = f"Project {pid}"
        placed = {sid for team in teams for sid in team}
        pool = participants.pool_for(pid) if participants is not None else None
        expected = list(pool) if pool is not None else roster.ids
        who = "listed participant(s)" if pool is not None else "roster student(s)"
        if not teams:
            problems.append(f"{where}: {len(expected)} {who} not in any team (no team has "
                            "students from the roster)")
            continue
        missing = [sid for sid in expected if sid not in placed]
        if missing:
            problems.append(f"{where}: {len(missing)} {who} not in any team: {_id_list(missing)}")
        if pool is not None:
            extra = sorted(placed - set(pool))
            if extra:
                problems.append(f"{where}: {len(extra)} student(s) in a team are not on the "
                                f"participant list: {_id_list(extra)}")
        sizes = [len(team) for team in teams]
        breakdown = "" if all_sizes_listed(sizes, lo, hi) else f" ({describe_sizes(sizes)})"
        problems.extend(f"{where}: {problem.replace(' would have ', ' have ')}{breakdown}"
                        for problem in size_problems(sizes, lo, hi))
    return problems


def _worst_team(report: Dict[str, Any]) -> str:
    team_views = teams(report)
    if not team_views:
        return "n/a"
    worst = max(team_views, key=lambda t: float(t.penalty or 0.0))
    return f"{worst.name} ({_fmt(worst.penalty)})"


def _has_history(reports: Dict[str, Dict[str, Any]],
                 projects: Optional[Sequence[Dict[str, Any]]] = None) -> bool:
    if projects is None:
        return section_reunions(None, reports) > 0
    earlier = False
    for entry in projects:
        if earlier and str(entry["project_id"]) in reports:
            return True
        earlier = earlier or any(entry.get("teams") or [])
    return False


def _run_totals(reports: Dict[str, Dict[str, Any]],
                stats: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    return summarize_reports(reports, stats) if reports else {}


def _restarts_text(stats: Optional[Dict[str, Any]]) -> str:
    view = stats_view(stats)
    planned = view.restarts_planned
    if not planned:
        return "-"
    return f"{int(view.restarts_completed or 0)}/{int(planned)}"


def _prefer_text(report: Dict[str, Any]) -> str:
    prefs = preferences_section(report)
    total = int(prefs.prefer_total or 0)
    return f"{int(prefs.prefer_satisfied or 0)}/{total}" if total else "-"


def summary_table(reports: Dict[str, Dict[str, Any]],
                  elapsed: Optional[Dict[str, Optional[float]]] = None,
                  stats: Optional[Dict[str, Dict[str, Any]]] = None,
                  projects: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    sections = _has_history(reports, projects)
    headers = ["Project", "Teams", "Sizes", "Objective", "Quality\nvs random", "Hard\nviolations",
               "Repeat\npairs"] + (["Section-mates\nreunited"] if sections else []) + [
               "Worst team", "Prefers\nmet"]
    if stats is not None:
        headers.append("Restarts")
    if elapsed is not None:
        headers.append("Time")
    rows = []
    for pid, report in reports.items():
        ova = overall(report)
        team_views = teams(report)
        sizes = [int(t.size or 0) for t in team_views]
        headline = {row.key: row for row in headline_rows(
            summarize_reports({str(pid): report}), {str(pid): report})}
        row = [pid, ova.n_teams if ova.n_teams is not None else len(sizes), _fmt_sizes(sizes),
               _fmt(ova.objective), _fmt_quality(ova.quality_index_soft),
               headline["hard"].text, headline["repeats"].text]
        if sections:
            section = headline.get("section_mates")
            row.append(section.text if section is not None else "-")
        row += [_worst_team(report), _prefer_text(report)]
        if stats is not None:
            row.append(_restarts_text(stats.get(pid)))
        if elapsed is not None:
            seconds = elapsed.get(pid)
            row.append("-" if seconds is None else f"{seconds:.2f}s")
        rows.append(row)
    left = ("Project", "Sizes", "Worst team")
    return _table(headers, rows, right=[k for k, h in enumerate(headers) if h not in left])


def top_violations(reports: Dict[str, Dict[str, Any]], limit: int) -> Tuple[List[str], int]:
    hard: List[str] = []
    soft: List[str] = []
    for pid, report in reports.items():
        for team in teams(report):
            for issue in team.issues or []:
                tag = "hard" if issue.get("hard") else "soft"
                line = f"Project {pid} · {team.name}: {issue.get('text', '')} [{tag}]"
                (hard if issue.get("hard") else soft).append(line)
    lines = hard + soft
    return lines[:max(0, limit)], len(lines)


def _print_results(args: argparse.Namespace, reports: Dict[str, Dict[str, Any]],
                   elapsed: Optional[Dict[str, Optional[float]]] = None,
                   projects: Optional[Sequence[Dict[str, Any]]] = None,
                   stats: Optional[Dict[str, Any]] = None) -> None:
    if not reports:
        _say("No projects were evaluated.")
        return
    history = _has_history(reports, projects)
    _say(summary_table(reports, elapsed, stats, projects))
    total = _run_totals(reports, stats)
    if len(reports) > 1:
        quality = total.get("quality_index_soft")
        sections = (f", {total['section_reunions']} section-mate pair(s) reunited"
                    if history else "")
        _say(f"\nAll {total['projects']} projects: objective {_fmt(total['objective'])}, "
             f"mean quality vs random {_fmt_quality(quality)}, "
             f"{total['hard_units']} hard violation(s), {total['repeated_pairs']} repeat pair(s)"
             + sections)
    lines, count = top_violations(reports, getattr(args, "top", 10))
    if count:
        _say(f"\nRule violations ({count}; hard first):")
        for line in lines:
            _say(_wrap(line, first="  ", indent="      "))
        if count > len(lines):
            _say(f"  … and {count - len(lines)} more (see --report, or raise --top)")
    else:
        _say("\nNo rule violations.")
    if getattr(args, "quiet", False):
        return
    _say("Quality vs random: how much lower the soft penalty is than the mean of 200 random "
         "allocations with the same team sizes (100% = perfect, 0% = random).")
    reunited = int(total.get("section_reunions") or 0)
    if reunited and history:
        _say(_wrap(f"Section-mates reunited: {reunited} teammate pair(s). {SECTION_MATES_HELP}",
                   indent=""))


def _hard_units(reports: Dict[str, Dict[str, Any]]) -> int:
    return int(sum(int(overall(r).hard_units or 0) for r in reports.values()))


def _report_payload(command: str, args: argparse.Namespace, roster: Roster, policy: Policy,
                    projects: Sequence[Dict[str, Any]], reports: Dict[str, Dict[str, Any]],
                    warnings: Sequence[str], **extra: Any) -> Dict[str, Any]:
    return report_envelope(
        _REPORT_FORMAT, policy=policy, projects=projects, reports=reports,
        summary=_run_totals(reports, extra.get("stats")), warnings=warnings,
        command=command, data=str(roster.path), **extra)


def _write_outputs(args: argparse.Namespace, payload: Dict[str, Any], roster: Roster,
                   projects: Sequence[Dict[str, Any]], reports: Dict[str, Dict[str, Any]]) -> None:
    if getattr(args, "report", None):
        _write_json(args.report, payload)
        _note(args, f"Wrote report {args.report}")
    if getattr(args, "report_html", None):
        write_html_report(args, roster, projects, reports)


def write_html_report(args: argparse.Namespace, roster: Roster,
                      projects: Sequence[Dict[str, Any]],
                      reports: Dict[str, Dict[str, Any]]) -> bool:
    try:
        from app.utils.report_html import render_allocation_html
    except ImportError as exc:
        _err(f"warning: HTML report not written: app.utils.report_html is not available "
             f"in this build ({exc})")
        return False
    title = args.title or f"Team allocation — {roster.path.stem}"
    try:
        html = render_allocation_html(list(projects), roster.frame, reports, title=title,
                                      anonymize=bool(args.anonymize))
    except Exception as exc:
        _err(f"warning: HTML report not written: rendering failed ({exc})")
        return False
    _write_text(args.report_html, html, "HTML report")
    _note(args, f"Wrote HTML report {args.report_html}")
    return True


class _Progress:
    def __init__(self, enabled: bool) -> None:
        stream = sys.stderr
        self.stream = stream
        self.enabled = enabled and hasattr(stream, "isatty") and stream.isatty()
        self.last = 0.0
        self.dirty = False

    def __call__(self, fraction: float, best: float, message: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self.last < 0.2 and fraction < 1.0:
            return
        self.last = now
        filled = int(round(20 * max(0.0, min(1.0, fraction))))
        best_text = _fmt(best) if best is not None and math.isfinite(best) else "-"
        line = f"[{'#' * filled}{'.' * (20 - filled)}] {fraction:4.0%}  best {best_text}  {message}"
        self.stream.write("\r" + _clip(line, _width() - 1).ljust(_width() - 1))
        self.stream.flush()
        self.dirty = True

    def close(self) -> None:
        if self.dirty:
            self.stream.write("\r" + " " * (_width() - 1) + "\r")
            self.stream.flush()
            self.dirty = False


class _Interrupt:
    def __init__(self) -> None:
        self.requested = False
        self.previous: Any = None
        self.installed = False

    def __enter__(self) -> "_Interrupt":
        try:
            self.previous = signal.signal(signal.SIGINT, self._handler)
            self.installed = True
        except ValueError:
            self.installed = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.installed:
            signal.signal(signal.SIGINT, self.previous)

    def _handler(self, signum: int, frame: Any) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        _err("\nStopping: keeping the best allocation found so far (Ctrl+C again to abort)")

    def __call__(self) -> bool:
        return self.requested


def _parse_target(spec: str) -> Tuple[str, Optional[Tuple[Optional[float], Optional[float]]]]:
    name, sep, rest = spec.partition(":")
    if not sep:
        return spec, None
    lo_text, _, hi_text = rest.partition(":")
    try:
        lo = float(lo_text) if lo_text.strip() else None
        hi = float(hi_text) if hi_text.strip() else None
    except ValueError:
        raise CliError(f"--targets {spec!r}: use NAME, or NAME:LO:HI for a numeric range "
                       "(either end may be empty)") from None
    return name, (lo, hi)


def cmd_attributes(args: argparse.Namespace) -> int:
    roster = _roster_from_args(args)
    rows = attribute_overview(roster.frame, team_size=args.team_size)
    redundancy = redundant_pairs(roster.frame)
    targets = []
    for spec in args.targets or []:
        name, rng = _parse_target(spec)
        if name not in roster.frame.columns:
            raise CliError(f"--targets: attribute '{name}' is not in the roster "
                           f"(attributes: {', '.join(r['name'] for r in rows)})")
        kind = next((r["kind"] for r in rows if r["name"] == name), None)
        if rng is not None and kind != "numeric":
            raise CliError(f"--targets {spec!r}: a LO:HI range needs a numeric attribute, but "
                           f"'{name}' is {kind or 'not an attribute'}; use --targets {name}")
        try:
            found = count_rule_targets(roster.frame, name, args.team_size,
                                       num_teams=args.num_teams or None, numeric_range=rng)
        except ValueError as exc:
            raise CliError(f"--targets {spec!r}: {exc}; e.g. --targets {name}:26: for "
                           f"'{name} >= 26'") from None
        targets.append({"attribute": name, "range": rng, "rows": found})
    if args.json:
        _say(json.dumps(json_safe({"data": str(roster.path), "students": len(roster.frame),
                                   "attributes": rows, "redundancy": redundancy,
                                   "targets": targets}), indent=2, ensure_ascii=False))
        return EXIT_OK

    _say(f"{roster.path.name}: {len(roster.frame)} students, {len(rows)} attributes "
         f"(team size {args.team_size})\n")
    table_rows = []
    for row in rows:
        values = row["stats"] or row["top_values"]
        goal = row["suggested_goal"]
        if goal != "ignore":
            goal = f"{goal} (w{row['suggested_weight']:g})"
        table_rows.append([row["name"], row["kind"], f"{100 * row['coverage']:.0f}%",
                           row["n_unique"], _clip(values, 56), goal])
    _say(_table(["Attribute", "Kind", "Coverage", "Unique", "Values", "Suggested goal"],
                table_rows, right=(2, 3)))
    if args.verbose:
        _say("\nWhy these suggestions:")
        for row in rows:
            _say(_wrap(f"{row['name']}: {row['why']}", first="  - ", indent="    "))
    if redundancy:
        _say("\nAttributes that carry the same information (weighting both double-counts it):")
        for item in redundancy:
            _say(_wrap(f"{item['a']} ~ {item['b']} ({item['relation']}): {item['explanation']}",
                       first="  - ", indent="    "))
    for target in targets:
        _say(f"\nHead-count targets for {target['attribute']}:")
        shown = target["rows"][:_TARGET_ROWS]
        for row in shown:
            _say(_wrap(row["hint"], first="  - ", indent="    "))
        if len(target["rows"]) > len(shown):
            _say(f"  … and {len(target['rows']) - len(shown)} rarer values (see --json)")
    return EXIT_OK


def cmd_presets(args: argparse.Namespace) -> int:
    for name, preset in PRESETS.items():
        _say(f"{name:<18} {preset.title}")
        _say(_wrap(preset.description, first=" " * 19, indent=" " * 19))
    return EXIT_OK


def cmd_suggest_policy(args: argparse.Namespace) -> int:
    roster = _roster_from_args(args)
    policy = suggest_policy(roster.frame, preset=args.preset, team_size=args.team_size,
                            num_teams=args.num_teams or None)
    _apply_solver_overrides(policy, args)
    problems = policy.validate(attribute_kinds(roster.frame))
    if problems:  # pragma: no cover - suggestions are built from the roster
        raise CliError("the suggested policy is invalid:\n  - " + "\n  - ".join(problems))
    text = policy.to_json()
    lines = describe_policy(policy, attribute_kinds(roster.frame))
    if args.out:
        _write_text(args.out, text + "\n", "policy")
        _say(f"Wrote policy '{policy.name}' to {args.out}")
        for line in lines:
            _say(_wrap(line, first="  - ", indent="    "))
    else:
        _say(text)
        _note(args, f"Policy '{policy.name}':")
        for line in lines:
            _note(args, _wrap(line, first="  - ", indent="    "))
    return EXIT_OK


_LEVEL_TAGS = {"error": "ERROR", "warning": "WARNING", "info": "note"}


def _print_findings(report: FeasibilityReport, include_notes: bool = True,
                    printer: Callable[[str], None] = _say) -> None:
    for finding in report.findings:
        if finding.level == "info" and not include_notes:
            continue
        tag = f"{_LEVEL_TAGS.get(finding.level, finding.level):<8}"
        where = f"[{finding.area}{': ' + finding.rule_label if finding.rule_label else ''}] "
        printer(_wrap(where + finding.message, first="  " + tag + " ", indent=" " * 11))


def cmd_feasibility(args: argparse.Namespace) -> int:
    roster = _roster_from_args(args)
    policy = resolve_policy(args, roster)
    plan = build_plan(args, roster)
    for note in plan.notes:
        _warn(args, note)
    report = check_feasibility(roster.frame, policy, plan.specs, plan.prior,
                               default_team_size=args.team_size)
    if args.json:
        _say(json.dumps(json_safe(report.to_dict()), indent=2, ensure_ascii=False))
    else:
        ids = ", ".join(s.project_id for s in plan.specs)
        _say(f"Feasibility of '{policy.name}' for project(s) {ids} "
             f"({pool_text(roster, plan.specs, bracket=False)}; {len(plan.history)} imported "
             f"project(s)): "
             f"{report.summary()}")
        _print_findings(report)
        if report.ok:
            _say("\nNo errors: every hard rule can be met as far as the checks can tell.")
        else:
            _say("\nErrors mean some rule cannot be met by any allocation; fix them or make "
                 "the rule soft.")
    return EXIT_OK if report.ok else EXIT_FAILED


def cmd_allocate(args: argparse.Namespace) -> int:
    roster = _roster_from_args(args)
    policy = resolve_policy(args, roster)
    plan = build_plan(args, roster)
    for note in plan.notes:
        _warn(args, note)
    s = policy.solver
    ids = ", ".join(spec.project_id for spec in plan.specs)
    check_output_paths(args, (("out", "allocation"), ("report", "report"),
                              ("report_html", "HTML report")))
    if not args.no_check:
        check = check_feasibility(roster.frame, policy, plan.specs, plan.prior,
                                  default_team_size=args.team_size, probe_time_s=0.2)
        if check.errors and not args.force:
            _err(f"Pre-flight check: {check.summary()}")
            _print_findings(check, include_notes=False, printer=_err)
            raise CliError(f"the pre-flight check found {len(check.errors)} error(s); nothing "
                           "was allocated. Fix the data, sizes or rules, or pass --force to "
                           "allocate anyway (--no-check skips the check)")
        if check.errors or (check.warnings and not args.quiet):
            _err(f"Pre-flight check: {check.summary()}")
            _print_findings(check, include_notes=False, printer=_err)
            if check.errors:
                _err("--force: allocating despite the errors above")

    _note(args, f"Allocating project(s) {ids} for {pool_text(roster, plan.specs)} with policy "
                f"'{policy.name}' (method {s.method}, {s.restarts} restart(s), seed {s.seed}, "
                f"time limit {s.time_limit_s:g} s per project)")
    if plan.history:
        _note(args, f"Continuing from {len(plan.history)} imported project(s): "
                    + ", ".join(p["project_id"] for p in plan.history))
    progress = _Progress(not args.quiet)
    try:
        with _Interrupt() as interrupt:
            result, _ = run_engine(plan.config, roster.students, raw_df=roster.raw,
                                   frame=roster.frame, prior=plan.prior,
                                   participants=plan.participants, policy=policy,
                                   name_teams=True, progress=progress, cancel=interrupt,
                                   apply_server_caps=False)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    finally:
        progress.close()

    generated = [{"project_id": pr.project_id, "name": f"Project {pr.project_id}",
                  "imported": False, "teams": pr.teams, "team_names": list(pr.team_names)}
                 for pr in result.projects]
    projects = plan.history + generated
    reports = {pr.project_id: pr.report for pr in result.projects}
    elapsed = {pr.project_id: _solve_seconds(pr.stats) for pr in result.projects}

    if args.out:
        rows = write_allocation_csv(args.out, projects)
        _note(args, f"Wrote {args.out} ({rows} rows, projects "
                    f"{', '.join(p['project_id'] for p in projects)})")
    payload = _report_payload(
        "allocate", args, roster, policy, projects, reports, result.warnings,
        method=result.method, elapsed_s=result.elapsed_s,
        stats={pr.project_id: {k: v for k, v in pr.stats.items() if k != "best_history"}
               for pr in result.projects})
    _write_outputs(args, payload, roster, projects, reports)

    _say("")
    _print_results(args, reports, elapsed, projects,
                   {pr.project_id: pr.stats for pr in result.projects})
    for pr in result.projects:
        note = _milp_note(pr.project_id, pr.stats)
        if note:
            _note(args, note)
    for warning in result.warnings:
        _warn(args, warning)
    _note(args, f"Done in {result.elapsed_s:.2f} s.")
    if not args.out:
        _note(args, "Tip: --out FILE writes the allocation (team_id,project_id,student_id) for "
                    "the app's import.")
    hard = _hard_units(reports)
    if args.strict and hard:
        _err(f"strict: {hard} hard violation(s) remain")
        return EXIT_FAILED
    return EXIT_OK


def _solve_seconds(stats: Dict[str, Any]) -> Optional[float]:
    view = stats_view(stats)
    parts = [view.elapsed_s, view.milp.elapsed_s]
    known = [float(p) for p in parts if p is not None]
    return sum(known) if known else None


def _milp_note(project_id: str, stats: Dict[str, Any]) -> str:
    view = stats_view(stats)
    if not view.milp.raw:
        return ""
    info = view.milp
    status = info.status or "unknown"
    gap = info.gap
    gap_text = f", gap {100 * float(gap):.1f}%" if gap is not None else ""
    line = f"Project {project_id}: MILP {status}{gap_text} in {_fmt(info.elapsed_s, 1)} s"
    if status != "optimal":
        line += " - not proven optimal"
    used = view.method_used
    if used == "milp+anneal" and view.method == "milp":
        line += "; polished by annealing automatically"
    elif used == "milp+anneal":
        line += ("; the restart from its solution won" if view.warm_start_won
                 else "; an annealing restart beat its solution")
    elif used == "anneal" and status != "cancelled":
        line += "; no usable allocation, annealing used instead"
    return line


def _projects_for_evaluation(args: argparse.Namespace, roster: Roster, path: str,
                             history: List[Dict[str, Any]], label: str = ""
                             ) -> List[Dict[str, Any]]:
    imp, notes = load_assignments(path, roster, allow_gaps=True)
    for note in notes:
        _warn(args, label + note)
    known = {p["project_id"]: p for p in history}
    projects = list(history)
    for entry in plain_projects(imp, imported=False):
        prior = known.get(entry["project_id"])
        if prior is None:
            projects.append(entry)
            continue
        same = ({frozenset(t) for t in prior["teams"]} == {frozenset(t) for t in entry["teams"]})
        if not same:
            _warn(args, f"{label}project {entry['project_id']} differs between "
                        f"{Path(path).name} and the --history file; the history version is used")
    return projects


def _history_projects(args: argparse.Namespace, roster: Roster) -> List[Dict[str, Any]]:
    if not getattr(args, "history", None):
        return []
    imp, notes = load_assignments(args.history, roster, allow_gaps=True, what="history file")
    for note in notes:
        _warn(args, note)
    return plain_projects(imp, imported=True)


def _evaluation_participants(args: argparse.Namespace, roster: Roster,
                             projects: Sequence[Dict[str, Any]]) -> Optional[ParticipantSpec]:
    if not getattr(args, "participants", None):
        return None
    name = Path(args.participants).name
    spec, errors = parse_participants_csv(_read_bytes(args.participants, "participants file"))
    if errors:
        raise CliError(f"{name}: " + "; ".join(errors[:5]))
    numbers = [int(pid) for pid in [p["project_id"] for p in projects] + list(spec.participants)
               if str(pid).isdigit()]
    validate_participants(spec, set(roster.ids), 0, max(numbers, default=0))
    for warning in spec.warnings:
        _warn(args, f"{name}: {warning}")
    return spec


def cmd_evaluate(args: argparse.Namespace) -> int:
    roster = _roster_from_args(args)
    policy = resolve_policy(args, roster)
    sizes = _size_settings(args)
    check_output_paths(args, (("report", "report"), ("report_html", "HTML report")))
    history = _history_projects(args, roster)
    projects = _projects_for_evaluation(args, roster, args.allocation, history)
    only = {str(p) for p in args.project} if args.project else None
    if only:
        missing = only - {p["project_id"] for p in projects if not p.get("imported")}
        if missing:
            raise CliError(f"--project {', '.join(sorted(missing, key=trailing_number_key))}: not in "
                           f"{Path(args.allocation).name} (or taken from --history)")
    participants = _evaluation_participants(args, roster, projects)
    reports, warnings = evaluate_plain(roster, policy, projects, sizes, only)
    problems = coverage_problems(roster, projects, sizes, only, participants)
    warnings = list(warnings) + [p for p in problems if p not in warnings]
    _note(args, f"Evaluating {Path(args.allocation).name} with policy '{policy.name}'"
                + (f" (history: {', '.join(p['project_id'] for p in history)})" if history else ""))
    payload = _report_payload("evaluate", args, roster, policy, projects, reports, warnings,
                              allocation=str(args.allocation))
    _write_outputs(args, payload, roster, projects, reports)
    _say("")
    _print_results(args, reports, projects=projects)
    for warning in warnings:
        _warn(args, warning)
    hard = _hard_units(reports)
    if args.strict and (hard or problems or not reports):
        if not reports and not problems:
            _err("strict: no project was evaluated")
        if hard:
            _err(f"strict: {hard} hard violation(s)")
        for problem in problems:
            _err(f"strict: {problem}")
        return EXIT_FAILED
    return EXIT_OK


def _delta(a: Optional[float], b: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if a is None or b is None:
        return ""
    d = float(b) - float(a)
    if abs(d) < 0.5 * 10 ** -digits:
        return "="
    return f"{d:+.{digits}f}{suffix}"


def _comparison_rows(project_id: str, ra: Dict[str, Any],
                     rb: Dict[str, Any]) -> List[List[str]]:
    oa, ob = overall(ra), overall(rb)
    ha, hb = history_section(ra), history_section(rb)
    pa, pb = preferences_section(ra), preferences_section(rb)

    def num(value: Optional[float]) -> Optional[float]:
        return None if value is None else float(value)

    sizes_a = [int(t.size or 0) for t in teams(ra)]
    sizes_b = [int(t.size or 0) for t in teams(rb)]
    rows = [
        ["Teams", _fmt_sizes(sizes_a), _fmt_sizes(sizes_b), ""],
        ["Objective (lower is better)", _fmt(num(oa.objective)), _fmt(num(ob.objective)),
         _delta(num(oa.objective), num(ob.objective))],
        ["Quality vs random", _fmt_quality(num(oa.quality_index_soft)),
         _fmt_quality(num(ob.quality_index_soft)),
         _delta(num(oa.quality_index_soft), num(ob.quality_index_soft), 1, " pts")],
        ["Hard violations", str(oa.hard_units or 0), str(ob.hard_units or 0),
         _delta(num(oa.hard_units), num(ob.hard_units), 0)],
        ["Worst team penalty", _fmt(num(oa.worst_team_penalty)),
         _fmt(num(ob.worst_team_penalty)),
         _delta(num(oa.worst_team_penalty), num(ob.worst_team_penalty))],
        ["Mean team penalty", _fmt(num(oa.mean_team_penalty)),
         _fmt(num(ob.mean_team_penalty)),
         _delta(num(oa.mean_team_penalty), num(ob.mean_team_penalty))],
        ["Repeat pairs", repeat_cap_text(int(ha.repeated_pairs or 0),
                                         int(ha.over_cap_pairs or 0)),
         repeat_cap_text(int(hb.repeated_pairs or 0), int(hb.over_cap_pairs or 0)),
         _delta(num(ha.repeated_pairs), num(hb.repeated_pairs), 0)],
        ["Section-mates reunited", str(section_reunions({}, {project_id: ra})),
         str(section_reunions({}, {project_id: rb})),
         _delta(section_reunions({}, {project_id: ra}),
                section_reunions({}, {project_id: rb}), 0)],
        ["Prefers met", _prefer_text(ra), _prefer_text(rb),
         _delta(num(pa.prefer_satisfied), num(pb.prefer_satisfied), 0)],
        ["Avoids violated", str(pa.avoid_violated or 0), str(pb.avoid_violated or 0),
         _delta(num(pa.avoid_violated), num(pb.avoid_violated), 0)],
    ]
    violations_a = sum(len(t.violations or []) for t in teams(ra))
    violations_b = sum(len(t.violations or []) for t in teams(rb))
    rows.append(["Rule violations (all)", str(violations_a), str(violations_b),
                 _delta(violations_a, violations_b, 0)])
    return rows


def _verdict(ra: Dict[str, Any], rb: Dict[str, Any]) -> str:
    oa, ob = overall(ra), overall(rb)
    ha, hb = int(oa.hard_units or 0), int(ob.hard_units or 0)
    if ha != hb:
        return f"{'A' if ha < hb else 'B'} is better: fewer hard violations ({min(ha, hb)} vs {max(ha, hb)})"
    a, b = float(oa.objective or 0.0), float(ob.objective or 0.0)
    if abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b)):
        return "A and B score the same under this policy"
    better = "A" if a < b else "B"
    return f"{better} is better under this policy (objective {_fmt(min(a, b))} vs {_fmt(max(a, b))})"


def cmd_compare(args: argparse.Namespace) -> int:
    roster = _roster_from_args(args)
    policy = resolve_policy(args, roster)
    sizes = _size_settings(args)
    check_output_paths(args, (("report", "report"),))
    history = _history_projects(args, roster)
    projects_a = _projects_for_evaluation(args, roster, args.a, history, label="A: ")
    projects_b = _projects_for_evaluation(args, roster, args.b, history, label="B: ")
    reports_a, warnings_a = evaluate_plain(roster, policy, projects_a, sizes)
    reports_b, warnings_b = evaluate_plain(roster, policy, projects_b, sizes)
    teams_a = {p["project_id"]: p["teams"] for p in projects_a if not p.get("imported")}
    teams_b = {p["project_id"]: p["teams"] for p in projects_b if not p.get("imported")}
    names_a = {p["project_id"]: p.get("team_names") for p in projects_a if not p.get("imported")}
    names_b = {p["project_id"]: p.get("team_names") for p in projects_b if not p.get("imported")}
    common = [pid for pid in reports_a if pid in reports_b]
    only_a = [pid for pid in reports_a if pid not in reports_b]
    only_b = [pid for pid in reports_b if pid not in reports_a]

    _say(f"A = {args.a}\nB = {args.b}\nPolicy: '{policy.name}'"
         + (f"; history: {', '.join(p['project_id'] for p in history)}" if history else ""))
    similarity: Dict[str, Dict[str, Any]] = {}
    for pid in common:
        overlap = allocation_similarity(teams_a[pid], teams_b[pid],
                                        names_a=names_a.get(pid), names_b=names_b.get(pid))
        similarity[pid] = overlap
        _say(f"\nProject {pid}")
        if overlap["identical_teams"] == overlap["teams_a"] == overlap["teams_b"]:
            _say(f"  identical allocations ({overlap['teams_a']} teams)")
            continue
        _say(_table(["Metric", "A", "B", "B - A"],
                    _comparison_rows(pid, reports_a[pid], reports_b[pid]),
                    right=(1, 2, 3)))
        if overlap["common_students"] == 0:
            _say("  no student is placed in both allocations")
        else:
            _say(f"Similarity: {overlap['shared_pairs']} shared teammate pairs "
                 f"({100 * overlap['share_of_a']:.0f}% of A's {overlap['pairs_a']}, "
                 f"{100 * overlap['share_of_b']:.0f}% of B's {overlap['pairs_b']}; "
                 f"Jaccard {100 * overlap['jaccard']:.0f}%), "
                 f"{overlap['identical_teams']} identical team(s)")
        _say(_verdict(reports_a[pid], reports_b[pid]))
    if only_a:
        _say(f"\nOnly in A: project(s) {', '.join(only_a)}")
    if only_b:
        _say(f"\nOnly in B: project(s) {', '.join(only_b)}")
    if not common:
        _say("\nNo project appears in both allocations; nothing to compare.")
    for warning in dict.fromkeys(warnings_a + warnings_b):
        _warn(args, warning)
    if getattr(args, "report", None):
        payload = report_envelope(
            _REPORT_FORMAT, policy=policy,
            warnings=list(dict.fromkeys(warnings_a + warnings_b)),
            command="compare", data=str(roster.path),
            a={"file": str(args.a), "projects": projects_a, "reports": reports_a,
               "summary": _run_totals(reports_a)},
            b={"file": str(args.b), "projects": projects_b, "reports": reports_b,
               "summary": _run_totals(reports_b)},
            similarity=similarity,
        )
        _write_json(args.report, payload)
        _note(args, f"Wrote report {args.report}")
    return EXIT_OK


_EPILOG = extensions.text('cli.epilog', """\
examples (from the repository root):
  python -m app.cli attributes --data data/demo/demo_cohort.csv --targets city --targets coffees_per_day:3:
  python -m app.cli suggest-policy --data data/demo/demo_cohort.csv --preset balanced --out policy.json
  python -m app.cli feasibility --data data/demo/demo_cohort.csv --policy policy.json --history data/demo/demo_prior_projects.csv --projects 3
  python -m app.cli allocate --data data/demo/demo_cohort.csv --history data/demo/demo_prior_projects.csv --projects 3 \\
      --out exports/allocation.csv --report exports/report.json --report-html exports/report.html
  python -m app.cli evaluate --data data/demo/demo_cohort.csv --allocation exports/allocation.csv --history data/demo/demo_prior_projects.csv
  python -m app.cli compare --data data/demo/demo_cohort.csv --a exports/run_a.csv --b exports/run_b.csv

exit codes: 0 ok; 1 rules cannot be met (feasibility), or with --strict: hard violations
remain (allocate, evaluate), or evaluate finds students in no team, a team size outside the
bounds or no project it could evaluate; 2 usage/input errors, also allocate stopped by a
pre-flight ERROR before solving (--force overrides; --no-check skips the check).
Every option of a command: python -m app.cli COMMAND --help
""")


def _data_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("data")
    g.add_argument("--data", required=True, metavar="FILE",
                   help=extensions.text('cli.data_help', 'roster CSV or Excel (.xlsx/.xlsm) file, e.g. data/demo/demo_cohort.csv'))
    g.add_argument("--sheet", metavar="NAME", help="worksheet of an Excel file (default: first)")
    g.add_argument("--config-dir", metavar="DIR",
                   help=f"profile directory (default: ${CONFIG_DIR_ENV} or config/)")
    g.add_argument("-q", "--quiet", action="store_true",
                   help="print only results (tables, violation list, JSON) and errors: no "
                        "progress, notes, warnings or explanations")
    return p


def _sizes_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("team sizes")
    g.add_argument("--team-size", type=_team_size, default=5, metavar="N",
                   help="target team size (default: 5)")
    g.add_argument("--min-size", type=_positive_int, metavar="N",
                   help="smallest allowed team (default: team size - 1)")
    g.add_argument("--max-size", type=_positive_int, metavar="N",
                   help="largest allowed team (default: team size + 1)")
    g.add_argument("--num-teams", type=_non_negative_int, default=0, metavar="N",
                   help="exact number of teams per project (default: from the sizes)")
    return p


def _policy_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("policy (one of; default: the suggested 'balanced' policy)")
    x = g.add_mutually_exclusive_group()
    x.add_argument("--policy", metavar="FILE",
                   help="policy JSON (also accepts a CLI --report or an app configuration "
                        "export, which carry the policy they used)")
    x.add_argument("--preset", choices=list(PRESETS), metavar="NAME",
                   help="suggest a policy from the data with this preset: "
                        + ", ".join(PRESETS))
    return p


def _plan_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("projects")
    g.add_argument("--history", metavar="FILE",
                   help="earlier projects 1..K (team_id,project_id,student_id); new projects "
                        "start at K+1 and avoid repeating those pairings")
    g.add_argument("--participants", metavar="FILE",
                   help="per-project participant lists (project_id,student_id)")
    g.add_argument("--projects", type=_positive_int, metavar="N",
                   help="total number of projects including the history (default: K+1)")
    return p


def _solver_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("solver (override the policy's settings)")
    g.add_argument("--method", choices=list(SOLVER_METHODS),
                   help="anneal (default), milp or milp+anneal")
    g.add_argument("--time-limit", type=_positive_float, metavar="S",
                   help="time budget per project in seconds")
    g.add_argument("--restarts", type=_positive_int, metavar="N",
                   help="independent annealing restarts (best kept)")
    g.add_argument("--iterations", type=_positive_int, metavar="N",
                   help="annealing moves per restart (default: automatic)")
    g.add_argument("--seed", type=_seed, default=argparse.SUPPRESS, metavar="N|none",
                   help="random seed; 'none' = different every run")
    g.add_argument("--workers", type=_positive_int, metavar="N",
                   help="worker processes for restarts (results do not depend on it)")
    g.add_argument("--joint-projects", action="store_true",
                   help="optimise all generated projects together")
    return p


def _report_parent(html: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("reports")
    g.add_argument("--report", metavar="FILE",
                   help="write a JSON report (policy, projects, per-project reports, warnings)")
    if html:
        g.add_argument("--report-html", metavar="FILE",
                       help="write a printable HTML report")
        g.add_argument("--title", help="title of the HTML report")
        g.add_argument("--anonymize", action="store_true",
                       help="hide student ids in the HTML report")
    g.add_argument("--top", type=_non_negative_int, default=10, metavar="N",
                   help="rule violations to list, also with --quiet (default: 10; 0 = only "
                        "the count)")
    g.add_argument("--strict", action="store_true",
                   help="exit with code 1 when hard rule violations remain (evaluate: also "
                        "when students are in no team, a team size is outside the bounds or "
                        "no project could be evaluated)")
    return p


def build_parser(prog: str = "python -m app.cli") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Group Diversifier: form diverse project teams from a roster, headless.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    fmt = argparse.RawDescriptionHelpFormatter

    p = sub.add_parser("attributes", parents=[_data_parent()], formatter_class=fmt,
                       help="what is in the roster: kinds, coverage, suggested goals",
                       description="Summarise the roster's attributes (kind, coverage, top "
                                   "values, suggested goal) and columns that carry the same "
                                   "information.")
    p.add_argument("--team-size", type=_team_size, default=5, metavar="N",
                   help="team size used for suggestions and targets (default: 5)")
    p.add_argument("--num-teams", type=_non_negative_int, default=0, metavar="N",
                   help="exact team count for --targets (default: from the team size)")
    p.add_argument("--targets", action="append", metavar="ATTR[:LO:HI]",
                   help=extensions.text('cli.targets_help', 'realistic per-team head-count limits for an attribute; numeric attributes need a range, e.g. coffees_per_day:3: (repeatable)'))
    p.add_argument("--verbose", action="store_true", help="explain every suggestion")
    p.add_argument("--json", action="store_true", help="print JSON instead of tables")
    p.set_defaults(handler=cmd_attributes)

    p = sub.add_parser("presets", help="list the policy presets",
                       description="List the presets suggest-policy and --preset accept.")
    p.set_defaults(handler=cmd_presets)

    p = sub.add_parser("suggest-policy", parents=[_data_parent(), _solver_parent()],
                       formatter_class=fmt, help="write a policy suggested from the data",
                       description="Suggest a complete policy from the roster (attribute goals, "
                                   "head-count and isolation rules, history, fairness) and "
                                   "write it as JSON.")
    p.add_argument("--preset", choices=list(PRESETS), default="balanced", metavar="NAME",
                   help="preset applied to the suggestion: " + ", ".join(PRESETS)
                        + " (default: balanced)")
    p.add_argument("--team-size", type=_team_size, default=5, metavar="N",
                   help="team size the suggestion is sized for (default: 5)")
    p.add_argument("--num-teams", type=_non_negative_int, default=0, metavar="N",
                   help="exact team count the suggestion is sized for")
    p.add_argument("--out", metavar="FILE", help="write the JSON here (default: print it)")
    p.set_defaults(handler=cmd_suggest_policy)

    p = sub.add_parser("feasibility",
                       parents=[_data_parent(), _policy_parent(), _plan_parent(), _sizes_parent()],
                       formatter_class=fmt, help="check whether the rules can be met",
                       description="Pre-flight check of a policy against the roster, team "
                                   "sizes, history and participants. Exit code 1 when a rule "
                                   "cannot be met by any allocation.")
    p.add_argument("--json", action="store_true", help="print the findings as JSON")
    p.set_defaults(handler=cmd_feasibility)

    p = sub.add_parser("allocate",
                       parents=[_data_parent(), _policy_parent(), _plan_parent(), _sizes_parent(),
                                _solver_parent(), _report_parent()],
                       formatter_class=fmt, help="form teams for one or more projects",
                       description="Form teams for projects K+1..N (K = projects in --history) "
                                   "and write them in the app's import format.")
    p.add_argument("--out", metavar="FILE",
                   help="allocation CSV team_id,project_id,student_id (history included, so "
                        "the file imports into the app as projects 1..N)")
    p.add_argument("--no-check", action="store_true", help="skip the pre-flight feasibility check")
    p.add_argument("--force", action="store_true",
                   help="allocate even when the pre-flight check reports ERRORs (a rule or the "
                        "team sizes cannot be met); without it such a run stops with exit "
                        "code 2 before solving. Not the same as --strict, which judges the "
                        "result afterwards (exit 1 when hard violations remain)")
    p.set_defaults(handler=cmd_allocate)

    p = sub.add_parser("evaluate",
                       parents=[_data_parent(), _policy_parent(), _sizes_parent(),
                                _report_parent()],
                       formatter_class=fmt, help="score an existing allocation",
                       description="Score every project of an allocation file with the engine "
                                   "(history = --history plus earlier projects in the file).")
    p.add_argument("--allocation", required=True, metavar="FILE",
                   help="allocation CSV team_id,project_id,student_id")
    p.add_argument("--history", metavar="FILE",
                   help="earlier projects; projects also in the allocation file are taken from "
                        "here and not evaluated")
    p.add_argument("--participants", metavar="FILE",
                   help="per-project participant lists (project_id,student_id): who should be "
                        "in each project (default: the whole roster)")
    p.add_argument("--project", action="append", metavar="ID",
                   help="only evaluate this project id (repeatable)")
    p.set_defaults(handler=cmd_evaluate)

    p = sub.add_parser("compare",
                       parents=[_data_parent(), _policy_parent(), _sizes_parent()],
                       formatter_class=fmt, help="compare two allocations side by side",
                       description="Evaluate two allocation files under the same policy and "
                                   "compare them project by project, with their similarity.")
    p.add_argument("--a", required=True, metavar="FILE", help="first allocation CSV")
    p.add_argument("--b", required=True, metavar="FILE", help="second allocation CSV")
    p.add_argument("--history", metavar="FILE",
                   help="earlier projects shared by both (not compared)")
    p.add_argument("--report", metavar="FILE", help="write the comparison as JSON")
    p.set_defaults(handler=cmd_compare)
    return parser


def _stdout_closed() -> None:
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)
    except (AttributeError, OSError, ValueError):
        pass


def _exit_code(code: Any) -> int:
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    return EXIT_USAGE


def main(argv: Optional[Sequence[str]] = None, *, prog: str = "python -m app.cli") -> int:
    try:
        return _dispatch(argv, prog)
    except BrokenPipeError:
        _stdout_closed()
        return EXIT_OK


def _dispatch(argv: Optional[Sequence[str]], prog: str) -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return _exit_code(exc.code)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return int(handler(args) or EXIT_OK)
    except CliError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    except KeyboardInterrupt:
        _err("aborted")
        return EXIT_INTERRUPTED


def run(prog: str = "python -m app.cli") -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    code = main(prog=prog)
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        _stdout_closed()
    sys.exit(code)


if __name__ == "__main__":
    run()
