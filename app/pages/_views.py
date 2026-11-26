from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.engine_session import (
    history_rules_text,
    rules_of_active_reports,
    rules_source_text,
)
from app.utils.headline import headline_rows

CLASSIC_SCORE_SCALE = ["#BD0226", "#E1AE67", "#5D8972"]


def render_headline_row(summary: Dict[str, Any], reports: Any, *,
                        preferences: bool = False,
                        preferences_help: Optional[str] = None,
                        trailing: Sequence[Tuple[str, Any, Optional[str]]] = ()) -> None:
    rows = [r for r in headline_rows(summary, reports) if r.key != "over_cap"]
    cols = st.columns(len(rows) + (1 if preferences else 0) + len(trailing))
    for col, row in zip(cols, rows):
        col.metric(row.label, row.text, help=row.help)
    tail = iter(cols[len(rows):])
    if preferences:
        prefer_total = int(summary.get("prefer_total") or 0)
        next(tail).metric("Preferences met",
                          f"{int(summary.get('prefer_satisfied') or 0)}/{prefer_total}"
                          if prefer_total else "—",
                          help=preferences_help)
    for col, (label, value, help) in zip(tail, trailing):
        col.metric(label, value, help=help)


def render_scored_rules_caption(lead: str = "These scores use", middle: str = "", *,
                                extra: str = "") -> None:
    text = f"{lead} {rules_source_text()}{middle}: " \
           f"{history_rules_text(rules_of_active_reports())}."
    if extra:
        text = f"{text} {extra}"
    st.caption(text)


def classic_team_score(metrics: Any, team: Any) -> float:
    return metrics.calculate_weighted_score(metrics.calculate_team_diversity(team))


def classic_score_bar(frame: pd.DataFrame, *, x: str, y: str, title: str,
                      **px_kwargs: Any) -> go.Figure:
    fig = px.bar(frame, x=x, y=y, color=y, color_continuous_scale=CLASSIC_SCORE_SCALE,
                 title=title, **px_kwargs)
    fig.update_layout(yaxis_range=[0, 1])
    return fig


def render_roster_team_view(projects: Sequence[Any], frame: pd.DataFrame, *,
                            key: str = "roster_team_view") -> None:
    from app.engine.roster import infer_attributes

    names = [f"Project {p.project_id} (imported)" if p.imported else f"Project {p.project_id}"
             for p in projects]
    chosen = st.selectbox("Select Project", range(len(projects)), format_func=lambda i: names[i],
                          key=f"{key}_project")
    project = projects[chosen]
    infos = infer_attributes(frame)
    columns = [c for c, info in infos.items() if info.usable]
    indexed = frame.set_index("student_id")
    st.subheader(f"Teams for {project.name}")
    comparison = []
    for team in project.teams:
        ids = [str(m.student_id) for m in team.members]
        rows = indexed.reindex(ids)[columns].reset_index().rename(columns={"student_id": "ID"})
        with st.expander(f"**{team.team_id}** - {team.size} members", expanded=True):
            st.dataframe(rows.astype(str), width="stretch", hide_index=True)
        row: Dict[str, Any] = {"Team": team.team_id, "Size": team.size}
        for col in columns:
            kind = infos[col].kind
            label = col.replace("_", " ").title()
            if kind == "numeric":
                row[f"{label} (mean)"] = round(float(pd.to_numeric(rows[col], errors="coerce").mean()), 2)
            else:
                values = {part for cell in rows[col].dropna() for part in str(cell).split("|") if part}
                row[f"{label} (distinct)"] = len(values)
        comparison.append(row)
    st.subheader("Team Comparison")
    st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
