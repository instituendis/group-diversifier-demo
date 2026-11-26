from __future__ import annotations

import hashlib
import html
import re
import secrets
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..core.engine_session.staleness import match_team_reports
from ..core.engine_session.summaries import summarize_reports
from ..engine.report_views import (
    TeamView,
    overall,
    rule_entries,
    teams,
)
from .exporters import plain_projects, pretty_team_name, team_names_for
from .headline import HeadlineRow, headline_rows
from .roster_view import MISSING_LABEL, RosterView

_DEFAULT_ATTRIBUTE_LIMIT = 6
_COMPACT_ATTRIBUTES = 3
_IDENTIFYING_TOKENS = {
    "name", "names", "firstname", "lastname", "surname", "fullname", "givenname",
    "email", "mail", "login", "username", "user", "userid", "phone", "mobile", "address",
    "sis", "identifier", "id", "orgdefinedid", "number", "snumber", "studentnumber",
}

_CSS = """
:root { --ink:#1d2433; --muted:#5b6475; --line:#d9dde5; --accent:#171c54;
        --good:#2f7d4f; --warn:#a86a00; --bad:#b3261e; --soft:#f4f6fa; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--ink); margin: 0 auto; max-width: 1100px; padding: 24px; font-size: 13px;
       line-height: 1.4; background: #fff; }
h1 { color: var(--accent); font-size: 24px; margin: 0 0 4px; }
h2 { color: var(--accent); font-size: 19px; margin: 0 0 8px; padding-bottom: 4px;
     border-bottom: 2px solid var(--accent); }
h3 { font-size: 14px; margin: 0 0 6px; }
.meta, .muted { color: var(--muted); }
.tag { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px;
       border-radius: 10px; background: var(--soft); border: 1px solid var(--line);
       color: var(--muted); vertical-align: middle; margin-left: 6px; }
.tag.hard { color: var(--bad); border-color: var(--bad); }
.cover { margin-bottom: 18px; }
.toc { margin: 8px 0 0; padding-left: 18px; }
.project { margin-top: 28px; }
.kpis { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }
.kpi { border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; min-width: 130px;
       background: var(--soft); }
.kpi .label { font-size: 11px; color: var(--muted); text-transform: uppercase;
              letter-spacing: .03em; }
.kpi .value { font-size: 17px; font-weight: 600; }
.kpi.bad .value { color: var(--bad); } .kpi.good .value { color: var(--good); }
table { border-collapse: collapse; width: 100%; margin: 4px 0 10px; }
th, td { border-bottom: 1px solid var(--line); padding: 3px 6px; text-align: left;
         vertical-align: top; }
th { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { display: inline-block; height: 8px; background: var(--accent); border-radius: 4px;
       vertical-align: middle; margin-right: 6px; }
.teams { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 10px; }
.teams.wide { grid-template-columns: 1fr; }
.team { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
        break-inside: avoid; page-break-inside: avoid; }
.team.flagged { border-left: 4px solid var(--warn); }
.team.hard { border-left: 4px solid var(--bad); }
.team table { font-size: 12px; margin-bottom: 6px; }
.composition { margin: 4px 0; padding: 0; list-style: none; font-size: 12px; }
.composition li { margin: 1px 0; }
.composition b { color: var(--muted); font-weight: 600; }
.violations { margin: 4px 0 0; padding-left: 16px; font-size: 12px; }
.violations li.hard { color: var(--bad); }
.note { background: var(--soft); border: 1px solid var(--line); border-radius: 6px;
        padding: 6px 10px; margin: 6px 0 10px; }
footer { margin-top: 30px; font-size: 11px; color: var(--muted); }
.sheets .team { margin-bottom: 14px; }
.context { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
@page { size: A4; margin: 14mm; }
@media print {
  body { max-width: none; padding: 0; font-size: 11px; }
  .project { margin-top: 0; }
  .project ~ .project { break-before: page; page-break-before: always; }
  .sheets .team { break-after: page; page-break-after: always; border: none; }
  .toc a { color: inherit; text-decoration: none; }
  .kpi, .note { background: none; }
}
"""


def pseudonym_map(student_ids: Iterable[Any], salt: str = "") -> Dict[str, str]:
    ids = sorted({str(s).strip() for s in student_ids if str(s).strip()})
    digests = {sid: hashlib.sha256(f"{salt}:{sid}".encode("utf-8")).hexdigest().upper()
               for sid in ids}
    length = 6
    while length < 64:
        codes = {sid: digest[:length] for sid, digest in digests.items()}
        if len(set(codes.values())) == len(codes):
            return {sid: f"S-{code}" for sid, code in codes.items()}
        length += 2
    return {sid: f"S-{digest}" for sid, digest in digests.items()}  # pragma: no cover


def _identifying(column: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(column).lower()) if t]
    joined = "".join(tokens)
    return any(t in _IDENTIFYING_TOKENS for t in tokens) or joined in _IDENTIFYING_TOKENS


_ID_SEPARATORS = r"\s,&→()"


def _sub_ids(text: str, alias: Dict[str, str]) -> str:
    if not alias or not text:
        return text
    choices = "|".join(re.escape(sid) for sid in sorted(alias, key=len, reverse=True) if sid)
    if not choices:
        return text
    pattern = rf"(?<![^{_ID_SEPARATORS}])(?:{choices})(?![^{_ID_SEPARATORS}])"
    return re.sub(pattern, lambda m: alias[m.group(0)], text)


_COUNT_FAMILIES = ("count", "isolation", "bounds")


def _anonymize_issue(issue: Dict[str, Any], alias: Dict[str, str]) -> str:
    text = str(issue.get("text", ""))
    family = issue.get("family")
    if family == "history":
        head, sep, tail = text.partition(" (met in ")
        return _sub_ids(head, alias) + sep + tail
    if family == "shared":
        head, sep, tail = text.rpartition(" for ")
        return head + sep + _sub_ids(tail, alias) if sep and "; no " in head else text
    if family in _COUNT_FAMILIES:
        return text
    return _sub_ids(text, alias)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "–"
    if number != number:
        return "–"
    return f"{number:,.{digits}f}"


def _units(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "–"
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


def _id_key(sid: str):
    return (0, int(sid), "") if sid.isdigit() else (1, 0, sid)


def _kpi(label: str, value: str, tone: str = "", title: str = "") -> str:
    attr = f' title="{_e(title)}"' if title else ""
    return (f'<div class="kpi {tone}"{attr}><div class="label">{_e(label)}</div>'
            f'<div class="value">{_e(value)}</div></div>')


def default_report_attributes(roster_frame: Any, reports: Dict[str, Dict[str, Any]]) -> List[str]:
    view = roster_frame if isinstance(roster_frame, RosterView) else RosterView(roster_frame)
    return _default_attributes(view, {str(k): v for k, v in (reports or {}).items()})


def _default_attributes(view: RosterView, reports: Dict[str, Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for report in reports.values():
        for team in teams(report)[:1]:
            for name in list(team.composition or {}) + list(team.numeric_means or {}):
                if name not in names:
                    names.append(name)
    names = [n for n in names if n in view.frame.columns]
    if names:
        return names[:_DEFAULT_ATTRIBUTE_LIMIT]
    return view.usable_attributes()[:_DEFAULT_ATTRIBUTE_LIMIT]


def _headline(rows: List[HeadlineRow], summary: Dict[str, Any],
              baseline: Any) -> str:
    parts = []
    for row in rows:
        if row.key == "over_cap":
            continue
        parts.append(_kpi(row.label, row.text, row.tone or "", row.help or ""))
        if row.key == "quality":
            parts.append(_kpi("Objective", _num(summary.get("objective"))))
    prefer_total = int(summary.get("prefer_total") or 0)
    avoid = int(summary.get("avoid_violated") or 0)
    if prefer_total or avoid:
        met = int(summary.get("prefer_satisfied") or 0)
        parts.append(_kpi("Preferences",
                          f"{met}/{prefer_total} met"
                          + (f", {avoid} avoid broken" if avoid else "")))
    html_parts = [f'<div class="kpis">{"".join(parts)}</div>']
    if baseline is not None and baseline.samples:
        html_parts.append(
            f'<p class="muted">Random baseline: mean penalty {_num(baseline.mean)} '
            f'± {_num(baseline.std)} over {int(baseline.samples)} random allocations; '
            f'{_num(baseline.percentile_of_solution, 0)}% of them score better.</p>')
    return "".join(html_parts)


def _rules_table(report: Dict[str, Any], alias: Optional[Dict[str, str]] = None) -> str:
    rules = rule_entries(report)
    if not rules:
        return '<p class="muted">No rules in the policy.</p>'
    group_rows, attr_rows = [], []
    for rule in rules:
        if rule.kind == "attribute":
            pct = float(rule.quality_pct or 0.0)
            attr_rows.append(
                f"<tr><td>{_e(rule.label)}</td><td class='num'>{_num(rule.weight)}</td>"
                f"<td><span class='bar' style='width:{max(0.0, min(pct, 100.0)) * 0.9:.0f}px'></span>"
                f"{pct:.0f}%</td><td class='num'>{_num(rule.points, 2)}</td></tr>")
        else:
            hard = "<span class='tag hard'>hard</span>" if rule.hard else ""
            label = str(rule.label or "")
            if alias and rule.kind == "pair":
                label = _sub_ids(label, alias)
            group_rows.append(
                f"<tr><td>{_e(label)}{hard}</td>"
                f"<td class='num'>{_num(rule.weight)}</td>"
                f"<td class='num'>{int(rule.teams_compliant or 0)}/{int(rule.teams_total or 0)}"
                f" ({_num(rule.compliance_pct, 0)}%)</td>"
                f"<td class='num'>{_units(rule.units)}</td>"
                f"<td class='num'>{_num(rule.points, 2)}</td></tr>")
    out = []
    if group_rows:
        out.append("<table class='rules'><thead><tr><th>Rule</th><th class='num'>Weight</th>"
                   "<th class='num'>Teams compliant</th><th class='num'>Violation units</th>"
                   "<th class='num'>Points</th></tr></thead><tbody>"
                   + "".join(group_rows) + "</tbody></table>")
    if attr_rows:
        out.append("<table class='rules'><thead><tr><th>Attribute goal</th><th class='num'>Weight</th>"
                   "<th>Quality (100% = perfect, 0% = random)</th><th class='num'>Points</th>"
                   "</tr></thead><tbody>" + "".join(attr_rows) + "</tbody></table>")
    return "".join(out)


def _team_card(name: str, members: Sequence[str], view: RosterView, attributes: Sequence[str],
               team_report: Optional[TeamView], alias: Optional[Dict[str, str]],
               context: str = "") -> str:
    def shown(sid: str) -> str:
        return alias.get(sid, sid) if alias else sid

    ordered = sorted(members, key=(lambda s: shown(s)) if alias else _id_key)
    issues = team_report.issues if team_report is not None else None
    if issues is None:
        issues = [{"text": v, "hard": False, "family": ""}
                  for v in (team_report.violations if team_report is not None else None) or []]
    hard = any(i.get("hard") for i in issues)
    classes = "team" + (" hard" if hard else " flagged" if issues else "")

    subtitle = [f"{len(members)} member{'s' if len(members) != 1 else ''}"]
    if team_report is not None and team_report.penalty is not None:
        subtitle.append(f"penalty {_num(team_report.penalty, 2)}")
    head = (f"<div class='context muted'>{_e(context)}</div>" if context else "") + (
        f"<h3>{_e(pretty_team_name(name))} <span class='muted'>· "
        f"{_e(' · '.join(subtitle))}</span></h3>")

    header = "".join(f"<th>{_e(a)}</th>" for a in attributes)
    rows = []
    for sid in ordered:
        known = sid in view
        cells = "".join(f"<td>{_e(view.text(sid, a))}</td>" for a in attributes)
        flag = "" if known else " <span class='muted'>(not in roster)</span>"
        rows.append(f"<tr><td>{_e(shown(sid))}{flag}</td>{cells}</tr>")
    table = (f"<table><thead><tr><th>Student</th>{header}</tr></thead>"
             f"<tbody>{''.join(rows)}</tbody></table>")

    comp_items = []
    for attr in attributes:
        summary = view.summary(members, attr)
        if summary and summary != MISSING_LABEL:
            comp_items.append(f"<li><b>{_e(attr)}</b> {_e(summary)}</li>")
    composition = f"<ul class='composition'>{''.join(comp_items)}</ul>" if comp_items else ""

    violations = ""
    if issues:
        items = []
        for issue in issues:
            text = _anonymize_issue(issue, alias) if alias else str(issue.get("text", ""))
            cls = " class='hard'" if issue.get("hard") else ""
            label = "hard: " if issue.get("hard") else ""
            items.append(f"<li{cls}>{_e(label + text)}</li>")
        violations = f"<ul class='violations'>{''.join(items)}</ul>"
    return f"<article class='{classes}'>{head}{table}{composition}{violations}</article>"


def _match_team_reports(project: Dict[str, Any], report: Optional[Dict[str, Any]],
                        view: Optional[RosterView] = None):
    if not report:
        return [None] * len(project["teams"]), False
    scored = teams(report)
    roster = view if view is not None and view.ids else None
    matched = match_team_reports(project["teams"], [t.members or [] for t in scored], roster)
    matched_teams = [scored[k] if k is not None else None for k in matched]
    every_scored_team_shown = len({k for k in matched if k is not None}) == len(scored)
    return matched_teams, every_scored_team_shown and all(
        team is not None or (roster is not None and not any(m in roster for m in members))
        for team, members in zip(matched_teams, project["teams"]))


def render_allocation_html(projects: List[Dict[str, Any]], roster_frame: pd.DataFrame,
                           reports: Dict[str, Dict[str, Any]], *,
                           title: str = "Team allocation",
                           attributes: Optional[List[str]] = None,
                           anonymize: bool = False,
                           team_sheets: bool = False,
                           salt: str = "",
                           generated_at: Optional[str] = None) -> str:
    view = RosterView(roster_frame)
    plain = plain_projects(projects)
    reports = {str(k): v for k, v in (reports or {}).items()}
    attrs = list(attributes) if attributes is not None else _default_attributes(view, reports)
    attrs = [a for a in attrs if a in view.frame.columns and a != "student_id"]
    dropped: List[str] = []
    if anonymize:
        dropped = [a for a in attrs if _identifying(a)]
        attrs = [a for a in attrs if a not in dropped]
    all_ids = {sid for p in plain for team in p["teams"] for sid in team}
    if anonymize and not salt:
        salt = secrets.token_hex(16)
    alias = pseudonym_map(all_ids | set(view.ids), salt) if anonymize else None
    stamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")

    n_teams = sum(len(p["teams"]) for p in plain)
    imported_tag = " <span class='muted'>(imported)</span>"
    toc = "".join(
        f"<li><a href='#project-{_e(p['project_id'])}'>{_e(p['name'])}</a>"
        f"{imported_tag if p.get('imported') else ''}"
        f" <span class='muted'>· {len(p['teams'])} teams</span></li>"
        for p in plain)
    notes = []
    if anonymize:
        notes.append("Student ids are replaced by pseudonyms.")
        if dropped:
            notes.append(f"Identifying columns omitted: {', '.join(dropped)}.")
    cover = (
        f"<header class='cover'><h1>{_e(title)}</h1>"
        f"<div class='meta'>Generated {_e(stamp)} · {len(plain)} project"
        f"{'s' if len(plain) != 1 else ''} · {n_teams} teams · {len(all_ids)} students</div>"
        + (f"<p class='note'>{_e(' '.join(notes))}</p>" if notes else "")
        + (f"<ol class='toc'>{toc}</ol>" if len(plain) > 1 else "")
        + "</header>")

    sections = []
    for project in plain:
        pid = project["project_id"]
        report = reports.get(pid)
        team_reports, all_matched = _match_team_reports(project, report, view)
        tags = "<span class='tag'>imported · locked history</span>" if project.get("imported") else ""
        parts = [f"<section class='project{' sheets' if team_sheets else ''}' id='project-{_e(pid)}'>"
                 f"<h2>{_e(project['name'])}{tags}</h2>"]
        if not team_sheets:
            if report:
                one = {pid: report}
                summary = summarize_reports(one)
                parts.append(_headline(headline_rows(summary, one), summary,
                                       overall(report).random_baseline_soft))
                if not all_matched:
                    parts.append("<p class='note'>Some teams differ from the scored allocation "
                                 "(edited after scoring?): their penalties are not shown.</p>")
                parts.append("<h3>Rule compliance</h3>" + _rules_table(report, alias))
            elif project.get("imported"):
                parts.append("<p class='note'>Imported project: recorded teams, used as pairing "
                             "history for later projects.</p>")
            else:
                parts.append("<p class='note'>This project has not been scored with the "
                             "allocation rules.</p>")
            parts.append("<h3>Teams</h3>")
        cards = [
            _team_card(name, members, view, attrs, team_report, alias,
                       context=project["name"] if team_sheets else "")
            for name, members, team_report in zip(team_names_for(project), project["teams"],
                                                  team_reports)
        ]
        layout = "teams wide" if len(attrs) > _COMPACT_ATTRIBUTES else "teams"
        parts.append(f"<div class='{layout}'>{''.join(cards)}</div></section>")
        sections.append("".join(parts))

    footer = ("<footer>Group Diversifier · penalties are rule points (lower is better); "
              "quality compares the soft penalty with random allocations of the same sizes."
              "</footer>")
    return ("<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
            f"{cover}{''.join(sections)}{footer}</body></html>\n")
