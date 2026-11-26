from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..engine.policy import Policy
from ..engine.policy_text import policy_diff
from ..engine.report_views import attribute_rules, rule_entries
from ..utils.fmt import fmt
from . import engine_session as es
from . import run_activation
from . import run_history as rh
from .allocation_similarity import allocation_similarity, teammate_pairs  # noqa: F401  teammate_pairs: re-exported (editor_ops)
from .edit_scoring import evaluate_plain
from .edit_session import EditError, EditSession
from .engine_session import as_policy


def run_metrics(record: rh.RunRecord) -> Dict[str, Any]:
    summary = dict(record.summary or {})
    if "objective" not in summary and record.reports:
        summary = {**es.summarize_reports(record.reports), **summary}
    return {
        "method": summary.get("method") or record.source,
        "quality": summary.get("quality_index_soft"),
        "objective": summary.get("objective"),
        "hard_units": summary.get("hard_units"),
        "repeated_pairs": summary.get("repeated_pairs"),
        "section_reunions": es.section_reunions(summary, record.reports),
        "show_section_reunions": es.show_section_reunions(summary, record.reports),
        "worst_team_penalty": summary.get("worst_team_penalty"),
        "prefer_satisfied": summary.get("prefer_satisfied"),
        "prefer_total": summary.get("prefer_total"),
        "avoid_violated": summary.get("avoid_violated"),
        "teams": summary.get("teams"),
        "projects": summary.get("projects"),
        "elapsed_s": summary.get("elapsed_s"),
        "time_limited": summary.get("time_limited"),
    }


def rule_compliance(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for report in reports.values():
        for rule in rule_entries(report):
            if rule.kind == "attribute":
                continue
            row = out.setdefault(rule.label, {"kind": rule.kind,
                                              "hard": bool(rule.hard),
                                              "teams_compliant": 0, "teams_total": 0,
                                              "units": 0.0})
            row["teams_compliant"] += int(rule.teams_compliant or 0)
            row["teams_total"] += int(rule.teams_total or 0)
            row["units"] += float(rule.units or 0.0)
    for row in out.values():
        total = row["teams_total"]
        row["compliance_pct"] = 100.0 * row["teams_compliant"] / total if total else 100.0
    return out


def attribute_quality(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    acc: Dict[str, Dict[str, Any]] = {}
    for report in reports.values():
        for rule in attribute_rules(report):
            row = acc.setdefault(rule.label, {"attribute": rule.attribute,
                                              "goal": rule.goal, "values": []})
            if rule.quality_pct is not None:
                row["values"].append(float(rule.quality_pct))
    return {label: {"attribute": row["attribute"], "goal": row["goal"],
                    "quality_pct": sum(row["values"]) / len(row["values"])
                    if row["values"] else None}
            for label, row in acc.items()}


def run_column(record: rh.RunRecord) -> str:
    return f"{record.run_id} · {record.label}"


def compare_rules(records: Sequence[rh.RunRecord]) -> List[Dict[str, Any]]:
    per_run = [(run_column(r), rule_compliance(r.reports)) for r in records]
    labels: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for _, rules in per_run:
        for label, row in rules.items():
            if label not in meta:
                labels.append(label)
                meta[label] = row
    rows = []
    for label in labels:
        row = {"rule": label, "kind": meta[label]["kind"],
               "hard": "hard" if meta[label]["hard"] else "soft"}
        for column, rules in per_run:
            row[column] = rules[label]["compliance_pct"] if label in rules else None
        rows.append(row)
    return rows


def compare_attributes(records: Sequence[rh.RunRecord]) -> List[Dict[str, Any]]:
    per_run = [(run_column(r), attribute_quality(r.reports)) for r in records]
    labels: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for _, rules in per_run:
        for label, row in rules.items():
            if label not in meta:
                labels.append(label)
                meta[label] = row
    rows = []
    for label in labels:
        row = {"rule": label, "attribute": meta[label]["attribute"], "goal": meta[label]["goal"]}
        for column, rules in per_run:
            row[column] = rules[label]["quality_pct"] if label in rules else None
        rows.append(row)
    return rows


def policy_differences(baseline: rh.RunRecord,
                       other: rh.RunRecord) -> Optional[List[str]]:
    if not baseline.policy or not other.policy:
        return None
    return policy_diff(Policy.from_dict(baseline.policy), Policy.from_dict(other.policy))


def common_project_ids(records: Sequence[rh.RunRecord]) -> List[str]:
    if not records:
        return []
    shared = set(p["project_id"] for p in records[0].projects)
    for record in records[1:]:
        shared &= {str(p["project_id"]) for p in record.projects}
    return [str(p["project_id"]) for p in records[0].projects if str(p["project_id"]) in shared]


def compare_allocations(baseline: rh.RunRecord, other: rh.RunRecord,
                        project_id: str) -> Dict[str, Any]:
    a, b = baseline.project(project_id), other.project(project_id)
    if a is None or b is None:
        raise ValueError(f"Project {project_id} is not in both runs")
    return allocation_similarity(a.get("teams", []), b.get("teams", []),
                                 names_a=a.get("team_names"), names_b=b.get("team_names"))


def activate_run(record: rh.RunRecord, config: Any, students: Sequence[Any], *,
                 raw_df: Optional[pd.DataFrame] = None,
                 policy: Any = None) -> List[str]:
    try:
        outcome = run_activation.activate_stored_projects(
            record.projects, config=config, students=students, raw_df=raw_df,
            reports=record.reports, scored=record.policy, fallback_policy=policy,
            run_id=record.run_id,
            stats={"run_id": record.run_id, "source": record.source,
                   "method": record.summary.get("method") or record.source},
            execution_time=float(record.summary.get("elapsed_s") or 0.0))
    except run_activation.ActivationError as exc:
        raise EditError(str(exc)) from exc
    if outcome.rescored_missing and not outcome.rescored_changed:
        record.reports = dict(outcome.reports)
        record.roster_signature = ""
        record.summary = {**record.summary, **es.summarize_reports(outcome.reports)}
    return outcome.warnings


def edited_record(session: EditSession, *, label: str, config: Any, policy: Any,
                  notes: str = "", roster: str = "") -> rh.RunRecord:
    reports = {pid: session.reports[pid] for pid in session.editable_project_ids()
               if pid in session.reports}
    summary = es.summarize_reports(reports)
    summary.update({"method": "edited", "edits": session.edit_count})
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    return rh.make_record(label=label, source="edited", config=config_dict,
                          policy=as_policy(policy).to_dict(),
                          projects=session.export_projects(), reports=reports,
                          summary=summary, notes=notes, parent_run_id=session.base_run_id,
                          roster_signature=roster)


def evaluated_record(projects: List[Dict[str, Any]], config: Any, students: Sequence[Any], *,
                     policy: Any, label: str, raw_df: Optional[pd.DataFrame] = None,
                     notes: str = "") -> rh.RunRecord:
    reports = evaluate_plain(projects, config, students, policy=policy, raw_df=raw_df)
    if not reports:
        raise EditError("No project of the allocation could be evaluated (no rostered "
                        "students in its teams)")
    summary = es.summarize_reports(reports)
    summary["method"] = "evaluated"
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    return rh.make_record(label=label, source="evaluated", config=config_dict,
                          policy=as_policy(policy).to_dict(), projects=projects,
                          reports=reports, summary=summary, notes=notes,
                          roster_signature=run_activation.roster_identity(students, raw_df))


def fmt_number(value: Any, digits: int = 2) -> str:
    return fmt(value, digits, compact="int")
