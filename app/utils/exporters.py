from __future__ import annotations
from typing import List, Dict, Any, Optional, Sequence, Union
from datetime import datetime, timezone
import math
import re
import pandas as pd
import json
import io

from .. import extensions
from ..core.models import Team, Project, FormationResult, Configuration, Student
from ..core.engine_session.plain_form import PlainProject, plain_projects, team_names_for
from ..core.engine_session.staleness import report_matches_teams
from ..core.engine_session.summaries import summarize_reports
from ..engine.report_views import history_section, rule_entries
from .headline import headline_rows as shared_headline_rows
from .roster_view import MISSING_LABEL, RosterView
from .serialization import json_safe

_LEADING_COLUMNS = ("student_id",) + tuple(extensions.hook("export_leading_columns", ()))

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def neutralise_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES) \
            and not _NUMBER_RE.match(value.strip()):
        return "'" + value
    return value


def neutralise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object or pd.api.types.is_string_dtype(out[column]):
            out[column] = out[column].map(neutralise_cell)
    out.columns = [neutralise_cell(c) for c in out.columns]
    return out


def safe_csv(frame: pd.DataFrame, **kwargs: Any) -> str:
    kwargs.setdefault("index", False)
    return neutralise_frame(frame).to_csv(**kwargs)


def _tidy(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
    return value


def member_record(member: Student, roster: Optional["RosterView"] = None) -> Dict[str, Any]:
    record: Dict[str, Any] = {"student_id": member.student_id}
    if roster is not None and member.student_id in roster:
        for attribute in roster.attributes:
            value = _tidy(roster.raw(member.student_id, attribute))
            if value is not None and value != "":
                record[attribute] = value
    for key, value in member.to_dict().items():
        if value not in (None, "") and key not in record:
            record[key] = value
    return record


def _ordered_columns(rows: List[Dict[str, Any]], leading: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for column in leading:
        if any(column in r for r in rows):
            seen[column] = None
    for row in rows:
        for column in row:
            seen.setdefault(column, None)
    return list(seen)


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        series = out[column]
        if series.dtype != object:
            continue
        kinds = {type(v) for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))}
        if len(kinds) > 1 or kinds & {list, tuple, dict, set}:
            out[column] = series.map(
                lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v))
                else ("|".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v))
            )
    return out


_SEARCH_SUMMARY_KEYS = ("time_limited", "restarts_completed", "restarts_planned")


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "–"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "–"
    if number.is_integer() and digits == 0:
        return f"{int(number)}{suffix}"
    return f"{number:.{digits}f}{suffix}"


class DataExporter:
    def __init__(self, roster: Any = None, reports: Optional[Dict[str, Dict[str, Any]]] = None,
                 raw_df: Optional[pd.DataFrame] = None):
        self._roster_arg = roster
        self._reports_arg = reports
        self._raw_df = raw_df

    def roster_view(self) -> Optional[RosterView]:
        if self._roster_arg is None:
            return None
        if isinstance(self._roster_arg, RosterView):
            return self._roster_arg
        return RosterView(self._roster_arg, raw_df=self._raw_df)

    def engine_reports(self, projects: Sequence[Project] = ()) -> Dict[str, Dict[str, Any]]:
        reports = {str(k): v for k, v in (self._reports_arg or {}).items()}
        if not projects:
            return reports
        out = {}
        view = self.roster_view()
        for project in projects:
            pid = str(project.project_id)
            if pid in reports and report_matches_teams(project, reports[pid], view):
                out[pid] = reports[pid]
        return out

    @staticmethod
    def engine_summary(reports: Dict[str, Dict[str, Any]],
                       result: Optional[FormationResult] = None) -> Dict[str, Any]:
        stats = getattr(result, "algorithm_stats", None) or {}
        stored = dict(stats.get("engine_summary") or {})
        if not reports:
            return stored
        per_project = stats.get("per_project")
        summary = summarize_reports(reports, per_project if isinstance(per_project, dict) else None)
        for key in _SEARCH_SUMMARY_KEYS:
            if key not in summary and key in stored:
                summary[key] = stored[key]
        return summary

    @staticmethod
    def headline_rows(summary: Dict[str, Any]) -> List[Dict[str, str]]:
        if not summary or not summary.get("projects"):
            return []
        prefer_total = summary.get("prefer_total") or 0
        rows = [{"Metric": row.label, "Value": row.text}
                for row in shared_headline_rows(summary)]
        rows.append({"Metric": "Rule violations (all rules)",
                     "Value": _fmt(summary.get("violations"), 0)})
        if prefer_total:
            rows.append({"Metric": "Preferred teammates satisfied",
                         "Value": f"{int(summary.get('prefer_satisfied') or 0)}"
                                  f" of {int(prefer_total)}"})
        if summary.get("avoid_violated") is not None:
            rows.append({"Metric": "Avoid requests violated",
                         "Value": _fmt(summary.get("avoid_violated"), 0)})
        if summary.get("time_limited"):
            rows.append({"Metric": "Time limited",
                         "Value": "yes (results may vary between machines)"})
        return rows

    @staticmethod
    def repeat_rows(projects: Sequence[Project],
                    reports: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        names = {str(p.project_id): p.name for p in projects}
        team_ids = {str(p.project_id): [t.team_id for t in p.teams] for p in projects}
        rows = []
        for pid, report in reports.items():
            for pair in history_section(report).pairs:
                t = int(pair.team or 0)
                ids = team_ids.get(pid, [])
                rows.append({
                    "Project": names.get(pid, f"Project {pid}"),
                    "Team": ids[t] if t < len(ids) else f"team_{t + 1}",
                    "Student_1": pair.a,
                    "Student_2": pair.b,
                    "Times_Before": pair.times_before,
                    "Met_In": ", ".join(map(str, pair.met_in or [])),
                    "Over_Cap": bool(pair.over_cap),
                })
        return rows

    @staticmethod
    def _team_rows(team: Team, include_details: bool, project_id: str = "",
                   project_name: str = "", imported: Optional[bool] = None,
                   roster: Optional[RosterView] = None) -> List[Dict[str, Any]]:
        rows = allocation_long_rows([{"project_id": project_id,
                                      "teams": [[m.student_id for m in team.members]],
                                      "team_names": [team.team_id]}])
        for row, member in zip(rows, team.members):
            if include_details:
                row.update(member_record(member, roster))
            if project_name:
                row["project_name"] = project_name
            if imported is not None:
                row["imported"] = imported
        return rows

    def export_teams_to_csv(
        self,
        teams: List[Team],
        include_details: bool = True,
        project_id: Optional[str] = None,
    ) -> str:
        roster = self.roster_view() if include_details else None
        rows: List[Dict[str, Any]] = []
        for team in teams:
            pid = project_id if project_id is not None else (team.project_id or "")
            rows.extend(self._team_rows(team, include_details, str(pid), roster=roster))
        columns = _ordered_columns(rows, ALLOCATION_COLUMNS)
        return safe_csv(pd.DataFrame(rows, columns=columns))

    def export_projects_to_csv(self, projects: Sequence[Project], include_details: bool = True) -> str:
        roster = self.roster_view() if include_details else None
        rows: List[Dict[str, Any]] = []
        for project in projects:
            for team in project.teams:
                rows.extend(self._team_rows(team, include_details, str(project.project_id),
                                            project.name, bool(project.imported), roster=roster))
        columns = _ordered_columns(
            rows, ALLOCATION_COLUMNS + ("project_name", "imported"))
        return safe_csv(pd.DataFrame(rows, columns=columns))

    def project_dataframe(self, project: Project, include_details: bool = True,
                          roster: Optional[RosterView] = None) -> pd.DataFrame:
        if roster is None and include_details:
            roster = self.roster_view()
        rows: List[Dict[str, Any]] = []
        for team in project.teams:
            rows.extend(self._team_rows(team, include_details, str(project.project_id),
                                        roster=roster))
        columns = _ordered_columns(rows, ("project_id", "team_id") + _LEADING_COLUMNS)
        return arrow_safe(pd.DataFrame(rows, columns=columns))

    def export_teams_to_excel(self, result: FormationResult) -> bytes:
        output = io.BytesIO()
        roster = self.roster_view()
        reports = self.engine_reports(result.projects)
        summary = self.engine_summary(reports, result)
        headline = self.headline_rows(summary)

        def write(df: pd.DataFrame, sheet: str) -> None:
            neutralise_frame(df).to_excel(writer, sheet_name=sheet, index=False)

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            rows = list(headline)
            if not headline:
                rows.append({"Metric": "Total Score (classic metrics)",
                             "Value": _fmt(result.total_score, 2)})
            rows += [
                {"Metric": "Execution Time (s)", "Value": f"{result.execution_time:.2f}"},
                {"Metric": "Number of Projects", "Value": str(len(result.projects))},
                {"Metric": "Number of Teams",
                 "Value": str(sum(len(p.teams) for p in result.projects))},
                {"Metric": "Algorithm", "Value": str(result.config.algorithm)},
            ]
            write(pd.DataFrame(rows, columns=["Metric", "Value"]), "Summary")

            used_names = {"Summary"}
            for project in result.projects:
                df = self.project_dataframe(project, roster=roster).drop(
                    columns=["project_id"], errors="ignore")
                df = df.rename(columns={"team_id": "Team", "student_id": "Student ID"})
                base = f"Project_{project.project_id}"[:31]
                name, suffix = base, 2
                while name in used_names:
                    name = f"{base[:28]}_{suffix}"
                    suffix += 1
                used_names.add(name)
                write(df, name)

            write(self.export_team_summary(result.projects), "Team_Summary")

            if reports:
                names = {str(p.project_id): p.name for p in result.projects}
                compliance = rule_compliance_frame(reports, names)
                if not compliance.empty:
                    write(compliance, "Rule_Compliance")
                repeats = self.repeat_rows(result.projects, reports)
                if repeats:
                    write(pd.DataFrame(repeats), "Repeat_Pairings")
            elif result.overlap_matrix:
                overlap_rows = [
                    {"Student_1": pair[0], "Student_2": pair[1], "Times_Paired": count}
                    for pair, count in result.overlap_matrix.items()
                    if count > 1
                ]
                if overlap_rows:
                    write(pd.DataFrame(overlap_rows), "Repeat_Pairings")

            classic = [{"Metric": "Total Score (classic metrics)",
                        "Value": _fmt(result.total_score, 2)}]
            classic += [{"Metric": f"Diversity: {attr}", "Value": f"{score:.3f}"}
                        for attr, score in (extensions.without_inactive_metrics(
                            result.diversity_scores) or {}).items()]
            if headline or len(classic) > 1:
                write(pd.DataFrame(classic, columns=["Metric", "Value"]), "Classic_Metrics")

        output.seek(0)
        return output.read()

    def export_to_json(self, result: FormationResult) -> str:
        roster = self.roster_view()
        reports = self.engine_reports(result.projects)
        summary = self.engine_summary(reports, result)
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "execution_time": result.execution_time,
                "engine_summary": summary or None,
            },
            "classic_metrics": {
                "total_score": result.total_score,
                "diversity_scores": extensions.without_inactive_metrics(result.diversity_scores),
            },
            "configuration": extensions.without_inactive_settings(result.config.to_dict()),
            "constraint_violations": result.constraint_violations,
            "rule_compliance": (rule_compliance_frame(reports).to_dict(orient="records")
                                if reports else []),
            "algorithm_stats": result.algorithm_stats,
            "projects": []
        }

        for project in result.projects:
            data["projects"].append({
                "project_id": project.project_id,
                "name": project.name,
                "imported": bool(project.imported),
                "teams": [
                    {
                        "team_id": team.team_id,
                        "size": team.size,
                        "members": [member_record(m, roster) for m in team.members],
                    }
                    for team in project.teams
                ],
            })

        return json.dumps(json_safe(data), indent=2, allow_nan=False)

    def export_team_summary(
        self,
        teams: Union[Sequence[Team], Sequence[Project]],
        project_id: Optional[str] = None,
    ) -> pd.DataFrame:
        entries = []
        for item in teams:
            if isinstance(item, Project):
                entries.extend((str(item.project_id), team) for team in item.teams)
            else:
                pid = project_id if project_id is not None else (item.project_id or "")
                entries.append((str(pid), item))

        extension_row = (extensions.hook("export_team_summary_row")
                      if extensions.active() else None)
        roster = self.roster_view() if extension_row is None else None
        rows = []
        for pid, team in entries:
            if extension_row is None:
                row = {"Project": pid, "Team": team.team_id, "Size": team.size}
                ids = [m.student_id for m in team.members]
                for attribute in (roster.usable_attributes() if roster is not None else []):
                    if roster.kind(attribute) == "numeric":
                        mean = roster.mean(ids, attribute)
                        row[f"{attribute} (mean)"] = None if mean is None else round(mean, 2)
                    else:
                        row[f"{attribute} (distinct)"] = len(roster.counts(ids, attribute))
                row["Members"] = ", ".join(m.student_id for m in team.members)
                rows.append(row)
                continue
            rows.append(extension_row(pid, team))
        df = pd.DataFrame(rows)
        if not df.empty and not any(pid for pid, _ in entries):
            df = df.drop(columns=["Project"])
        return df

    def generate_report(self, result: FormationResult) -> str:
        reports = self.engine_reports(result.projects)
        summary = self.engine_summary(reports, result)
        headline = self.headline_rows(summary)
        lines = [
            "# Team Formation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
        ]
        lines += [f"- **{row['Metric']}:** {row['Value']}" for row in headline]
        if not headline:
            lines.append(f"- **Total Score (classic metrics):** {_fmt(result.total_score, 2)}")
        lines += [
            f"- **Execution Time:** {result.execution_time:.2f}s",
            f"- **Algorithm:** {result.config.algorithm}",
            f"- **Projects:** {len(result.projects)}",
            f"- **Total Teams:** {sum(len(p.teams) for p in result.projects)}",
        ]

        if reports:
            compliance = [row for row in rule_compliance_frame(reports).to_dict(orient="records")
                          if row.get("kind") != "attribute"]
            if compliance:
                lines.extend(["", "## Rule Compliance"])
                for row in compliance:
                    pct = row.get("compliance_pct")
                    lines.append(f"- {row['project']} · {row['rule']} ({row['hard']}): "
                                 f"{_fmt(pct, 1, '%')} of teams comply")

        if result.constraint_violations:
            lines.extend(["", "## Constraint Violations"])
            for violation in result.constraint_violations:
                lines.append(f"- ⚠️ {violation}")

        classic_scores = extensions.without_inactive_metrics(result.diversity_scores)
        if classic_scores:
            lines.extend(["", "## Classic Metrics (legacy diversity scores)"])
            if headline:
                lines.append("_Display only: the allocation was scored by the rules engine "
                             "above._")
            lines.append(f"- Total Score: {_fmt(result.total_score, 2)}")
            for attr, score in classic_scores.items():
                lines.append(f"- {attr}: {score:.3f}")

        lines.extend(["", "## Team Assignments"])
        for project in result.projects:
            label = project.name + (" (imported)" if project.imported else "")
            lines.append(f"\n### {label}")
            for team in project.teams:
                members = ", ".join(m.student_id for m in team.members)
                lines.append(f"- **{team.team_id}** ({team.size} members): {members}")

        return "\n".join(lines)


CONFIG_EXPORT_FORMAT = "group-diversifier-config"
CONFIG_EXPORT_FORMAT_VERSION = 1


def export_configuration_to_json(
    config: Configuration,
    name: str = "Group Diversifier Configuration",
    description: str = "",
) -> str:
    payload: Dict[str, Any] = {
        "format": CONFIG_EXPORT_FORMAT,
        "format_version": CONFIG_EXPORT_FORMAT_VERSION,
        "name": name,
        "description": description,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "configuration": extensions.without_inactive_settings(config.to_dict()),
    }
    return json.dumps(json_safe(payload), indent=2)


def import_configuration_from_json(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration file is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a JSON object.")

    fmt = payload.get("format")
    if fmt != CONFIG_EXPORT_FORMAT:
        raise ValueError(
            f"Unrecognized configuration format {fmt!r}; "
            f"expected {CONFIG_EXPORT_FORMAT!r}."
        )
    version = payload.get("format_version")
    if version != CONFIG_EXPORT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported configuration format_version {version!r}; "
            f"expected {CONFIG_EXPORT_FORMAT_VERSION}."
        )

    raw_config = payload.get("configuration")
    if not isinstance(raw_config, dict):
        raise ValueError("Configuration file is missing the 'configuration' object.")
    try:
        configuration = Configuration.from_dict(raw_config)
    except TypeError as exc:
        raise ValueError(f"Invalid configuration payload: {exc}") from exc

    return {"configuration": configuration}


ENGINE_REPORT_FORMAT = "group-diversifier-engine-report"
TEAM_COLUMN_PREFIX = "team_project_"

def pretty_team_name(name: str) -> str:
    text = str(name)
    head, _, tail = text.partition("_")
    if head.lower() == "team" and tail.isdigit():
        return f"Team {int(tail)}"
    return text


def project_label(project: PlainProject) -> str:
    return f"{project['name']} (imported)" if project.get("imported") else str(project["name"])


ALLOCATION_COLUMNS = ("team_id", "project_id", "student_id")


def allocation_long_rows(projects: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for project in plain_projects(projects):
        pid = project["project_id"]
        for team_id, members in zip(team_names_for(project), project["teams"]):
            for student_id in members:
                rows.append({"team_id": team_id, "project_id": pid,
                             "student_id": student_id})
    return rows


def wide_roster_frame(projects: Sequence[Any], roster: Any, *,
                      attributes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    view = RosterView(roster)
    plain = plain_projects(projects)
    if attributes is None:
        attrs = view.attributes
    else:
        attrs = [a for a in attributes if a in view.frame.columns and a != "student_id"]
    team_columns = [f"{TEAM_COLUMN_PREFIX}{p['project_id']}" for p in plain]

    rows: List[Dict[str, Any]] = [
        {"student_id": sid, **{a: _tidy(view.raw(sid, a)) for a in attrs}} for sid in view.ids
    ]
    extra: Dict[str, int] = {}
    for project, column in zip(plain, team_columns):
        for team_name, members in zip(team_names_for(project), project["teams"]):
            for sid in members:
                pos = view.position(sid)
                if pos is None:
                    pos = extra.get(sid)
                    if pos is None:
                        pos = extra[sid] = len(rows)
                        rows.append({"student_id": sid, **{a: None for a in attrs}})
                rows[pos][column] = team_name
    frame = pd.DataFrame(rows, columns=["student_id", *attrs, *team_columns])
    frame[team_columns] = frame[team_columns].astype(object).fillna("")
    return frame


def composition_frame(projects: Sequence[Any], roster: Any, attributes: Sequence[str], *,
                      project_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    view = RosterView(roster)
    attrs = [a for a in attributes if view.kind(a) != "empty"]
    wanted = None if project_ids is None else {str(p) for p in project_ids}
    plain = [p for p in plain_projects(projects) if wanted is None or p["project_id"] in wanted]

    value_columns: Dict[str, List[str]] = {}
    for attr in attrs:
        if view.kind(attr) == "numeric":
            value_columns[attr] = [f"{attr} (mean)"]
            continue
        cohort = [v for v in view.cohort_counts(attr) if v != MISSING_LABEL]
        value_columns[attr] = [f"{attr}: {v}" for v in cohort]

    rows = []
    for project in plain:
        for team_name, members in zip(team_names_for(project), project["teams"]):
            row: Dict[str, Any] = {
                "project_id": project["project_id"],
                "project": project["name"],
                "imported": bool(project.get("imported")),
                "team": team_name,
                "size": len(members),
            }
            for attr in attrs:
                if view.kind(attr) == "numeric":
                    mean = view.mean(members, attr)
                    row[f"{attr} (mean)"] = None if mean is None else round(mean, 3)
                    continue
                for value, count in view.counts(members, attr).items():
                    column = f"{attr}: {value}"
                    if column not in value_columns[attr]:
                        value_columns[attr].append(column)
                    row[column] = count
            rows.append(row)

    columns = ["project_id", "project", "imported", "team", "size"]
    count_columns = []
    for attr in attrs:
        columns.extend(value_columns[attr])
        if view.kind(attr) != "numeric":
            count_columns.extend(value_columns[attr])
    frame = pd.DataFrame(rows, columns=columns)
    if count_columns and not frame.empty:
        frame[count_columns] = frame[count_columns].fillna(0).astype(int)
    return frame


_RULE_COLUMNS = [
    "project_id", "project", "rule", "kind", "hard", "weight", "attribute", "goal",
    "teams_compliant", "teams_total", "compliance_pct", "quality_pct", "units", "points",
]


def rule_compliance_frame(reports: Dict[str, Dict[str, Any]],
                          project_names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    names = project_names or {}
    rows = []
    for pid, report in (reports or {}).items():
        for rule in rule_entries(report):
            is_attribute = rule.kind == "attribute"
            rows.append({
                "project_id": str(pid),
                "project": names.get(str(pid), f"Project {pid}"),
                "rule": rule.label or "",
                "kind": rule.kind or "",
                "hard": "hard" if rule.hard else "soft",
                "weight": rule.weight,
                "attribute": rule.attribute if is_attribute else None,
                "goal": rule.goal if is_attribute else None,
                "teams_compliant": None if is_attribute else rule.teams_compliant,
                "teams_total": None if is_attribute else rule.teams_total,
                "compliance_pct": None if is_attribute else rule.compliance_pct,
                "quality_pct": rule.quality_pct if is_attribute else None,
                "units": None if is_attribute else rule.units,
                "points": rule.points,
            })
    return pd.DataFrame(rows, columns=_RULE_COLUMNS)


REPORT_ENVELOPE_VERSION = 1


def report_envelope(format_id: str, *, policy: Any = None,
                    projects: Optional[Sequence[Any]] = None,
                    reports: Optional[Dict[str, Dict[str, Any]]] = None,
                    summary: Optional[Dict[str, Any]] = None,
                    warnings: Optional[Sequence[str]] = None,
                    **extra: Any) -> Dict[str, Any]:
    if policy is not None and hasattr(policy, "to_dict"):
        policy = policy.to_dict()
    payload: Dict[str, Any] = {
        "format": format_id,
        "version": REPORT_ENVELOPE_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": policy or {},
    }
    if reports is not None:
        payload["reports"] = reports
    if summary is not None:
        payload["summary"] = summary
    if projects is not None:
        payload["projects"] = plain_projects(projects)
    if warnings is not None:
        payload["warnings"] = list(warnings)
    payload.update(extra)
    return payload


def engine_report_json(reports: Dict[str, Dict[str, Any]], policy: Any,
                       summary: Optional[Dict[str, Any]] = None, *,
                       projects: Optional[Sequence[Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
    payload = report_envelope(
        ENGINE_REPORT_FORMAT, policy=policy, reports=reports or {},
        summary=dict(summary or {}), metadata=dict(metadata or {}),
        projects=projects,
    )
    return json.dumps(json_safe(payload), indent=2, allow_nan=False)
