import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.ids import canonical_id
from app.core.data_loader import DataLoader
from app.core.data_session import (
    clear_uploaded_csv,
    ensure_session_state,
    flash,
    get_active_source,
    get_data_loader,
    get_loader_warnings,
    is_new_upload,
    mark_upload_processed,
    render_flashes,
    sample_download_button,
    set_uploaded_csv,
)
from app.core.engine_session import distinct_value_counts, roster_frame
from app.engine.attributes import (
    MAX_PAIRWISE_ATTRIBUTES,
    MAX_PAIRWISE_CARDINALITY,
    analysis_columns,
    association_matrix,
    attribute_overview,
    count_rule_targets,
    redundant_pairs,
)
from app.engine.roster import infer_attributes
from app.utils.exporters import safe_csv
from app.utils.validators import DataValidator
from app import extensions  # noqa: E402
from app.ui_kit import privacy_notice, refuse_large_table

st.set_page_config(
    page_title="Data Overview | Group Diversifier",
    page_icon="📊",
    layout="wide",
)

ensure_session_state()

_NON_CATEGORICAL = {
    "student_id", "prefer_teammates", "avoid_teammates", "group_number",
} | set(extensions.hook("ui.data_overview.non_categorical", ()))
_MAX_FILTER_VALUES = 60
_HIDDEN_COLUMNS = list(extensions.hook("ui.data_overview.hidden_columns", ()))


def _extension_ui(name: str):
    return extensions.hook(f"ui.data_overview.{name}") if extensions.active() else None
_LOW_COVERAGE = 0.8
_RELATION_ICONS = {
    "same": "🟰", "binning": "📦", "determines": "➡️", "correlated": "〰️", "associated": "〰️",
}
_KIND_LABELS = {
    "categorical": "categorical", "numeric": "numeric", "multi": "multi-value",
    "text": "free text", "constant": "constant", "empty": "empty",
}


def _split_ids(value) -> List[str]:
    text = "" if value is None else str(value)
    return [
        part.strip() for part in text.split("|")
        if part.strip() and part.strip().lower() not in ("nan", "none")
    ]


def _bulk_refs(value, roster_lookup: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    ids: List[str] = []
    dropped: List[str] = []
    split: List[str] = []
    for ref in _split_ids(value):
        sid = roster_lookup.get(_canonical(ref))
        if sid is not None:
            ids.append(sid)
            continue
        parts = [p.strip() for p in ref.replace(",", ";").split(";") if p.strip()]
        found = [roster_lookup.get(_canonical(p)) for p in parts]
        if ("," in ref or ";" in ref) and found and all(found):
            ids.extend(found)
            split.append(ref)
        else:
            dropped.append(ref)
    return ids, dropped, split


def _canonical(sid: str) -> str:
    return canonical_id(sid)


def _non_numeric_note(name: str, info) -> str:
    bad = int(info.stats.get("non_numeric", 0) or 0)
    if not bad or not info.non_numeric:
        return ""
    shown = ", ".join(str(v) for v in info.non_numeric[:5])
    more = " …" if len(info.non_numeric) > 5 else ""
    return (f"{bad} non-numeric value{'s' if bad != 1 else ''} in **{name}**: {shown}{more} — "
            "treated as missing")


def _flag_note(name: str, info) -> str:
    if info.flag == "identifier":
        return (f"**{name}** looks like an identifier (a different value for every student, "
                f"{info.n_unique} distinct) — no rule is suggested for it")
    if info.flag == "date":
        return (f"**{name}** holds dates — no rule is suggested for it (give it a goal on "
                "Allocation Rules to use the dates as categories)")
    return ""


def _excel_to_csv_text(data: bytes, sheet: str) -> str:
    return DataLoader.read_excel_frame(data, sheet).to_csv(index=False)


def _categorical_columns(frame: pd.DataFrame) -> List[str]:
    columns = []
    for col in frame.columns:
        if col in _NON_CATEGORICAL or col in _HIDDEN_COLUMNS:
            continue
        values = frame[col].dropna().astype(str).str.strip()
        n_unique = values[values != ""].nunique()
        if 2 <= n_unique <= _MAX_FILTER_VALUES:
            columns.append(col)
    return columns


def _quality_findings(frame: pd.DataFrame, infos, loader_warnings: List[str]) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    if loader_warnings:
        findings.append(("warning", f"{len(loader_warnings)} loader warning(s) while reading the "
                                    "file (listed at the top of this page)."))
    ids = frame["student_id"].astype(str)
    duplicated = sorted(set(ids[ids.duplicated()]))
    if duplicated:
        findings.append(("warning", f"Duplicate student ids: {', '.join(duplicated[:10])}"
                                    f"{' …' if len(duplicated) > 10 else ''}."))
    known = set(ids)
    unknown = set()
    for column in ("prefer_teammates", "avoid_teammates"):
        if column in frame.columns:
            for cell in frame[column].dropna().astype(str):
                unknown.update(sid for sid in _split_ids(cell)
                               if sid not in known and _canonical(sid) not in known)
    if unknown:
        findings.append(("warning", f"Preferences name {len(unknown)} id(s) that are not in the "
                                    f"roster (ignored): {', '.join(sorted(unknown)[:8])}."))
    for name, info in infos.items():
        non_numeric = _non_numeric_note(name, info)
        if non_numeric:
            findings.append(("warning", non_numeric + "."))
        flagged = _flag_note(name, info)
        if info.kind == "empty":
            findings.append(("info", f"**{name}** has no values — ignored."))
        elif info.kind == "constant":
            value = info.values[0] if info.values else ""
            findings.append(("info", f"**{name}** has a single value ({value}) — it cannot "
                                     "differentiate teams."))
        elif flagged:
            findings.append(("info", flagged + "."))
        elif info.kind == "text":
            findings.append(("info", f"**{name}** looks like free text ({info.n_unique} distinct "
                                     f"values) — not balanced unless grouped into categories."))
        if info.kind in ("categorical", "numeric", "multi") and info.coverage < _LOW_COVERAGE:
            findings.append(("warning", f"**{name}** is filled for only {info.coverage:.0%} of "
                                        "students — rules ignore missing values unless you treat "
                                        "missing as its own category."))
        if info.kind == "categorical" and 2 < info.n_unique <= 60:
            singles = [v for v in info.values if info.counts.get(v) == 1]
            if singles and len(singles) < info.n_unique:
                findings.append(("info", f"**{name}**: {len(singles)} value(s) held by a single "
                                         "student — a 'no lone …' rule can never hold for them."))
    return findings


def render_attribute_insights(raw_df: pd.DataFrame, students, loader) -> None:
    st.subheader("Attribute Insights")
    st.caption("How the allocation engine reads every column: kind, coverage, suggested rule, "
               "overlapping columns and realistic per-team head counts.")
    frame = roster_frame(students, raw_df)
    infos = infer_attributes(frame)
    config = st.session_state.config
    team_size = min(20, max(2, int(config.team_size_target or 5)))
    overview = attribute_overview(frame, infos, team_size=team_size)

    usable = [n for n, i in infos.items() if i.usable]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", len(frame))
    col2.metric("Usable attributes", len(usable),
                help="Categorical, numeric or multi-value columns rules can use")
    col3.metric("Suggested rules", sum(1 for r in overview if r["suggested_goal"] != "ignore"))
    col4.metric("Low coverage", sum(1 for n in usable if infos[n].coverage < _LOW_COVERAGE),
                help=f"Usable attributes filled for fewer than {_LOW_COVERAGE:.0%} of students")

    st.markdown("#### Data quality")
    findings = _quality_findings(frame, infos, get_loader_warnings(loader))
    if not findings:
        st.success("No data-quality issues found.")
    for level, message in findings[:15]:
        (st.warning if level == "warning" else st.info)(message)
    if len(findings) > 15:
        with st.expander(f"{len(findings) - 15} more finding(s)"):
            for _, message in findings[15:]:
                st.markdown(f"- {message}")

    st.markdown("#### Attribute overview")
    table = pd.DataFrame([{
        "Attribute": r["name"],
        "Kind": _KIND_LABELS.get(r["kind"], r["kind"]),
        "Coverage": r["coverage"],
        "Distinct": r["n_unique"],
        "Values": r["stats"] or r["top_values"],
        "Recognised as": r["role"],
        "Suggested goal": (f"{r['suggested_goal']} ({infos[r['name']].flag})"
                           if r["suggested_goal"] == "ignore" and r["name"] in infos
                           and infos[r["name"]].flag else r["suggested_goal"]),
        "Weight": r["suggested_weight"],
        "Why": r["why"],
    } for r in overview])
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Coverage": st.column_config.ProgressColumn(
                "Coverage", min_value=0.0, max_value=1.0, format="percent"),
            "Weight": st.column_config.NumberColumn("Weight", format="%.1f"),
            "Why": st.column_config.TextColumn("Why", width="large"),
        },
    )
    st.caption(f"Suggestions are what the Allocation Rules page proposes for teams of {team_size} "
               "(balanced preset); you can change every rule there.")

    st.markdown("#### How strongly attributes are related")
    matrix = association_matrix(frame, infos=infos)
    eligible = [n for n, i in infos.items()
                if i.kind in ("categorical", "numeric") and n in frame.columns]
    if len(analysis_columns(frame, infos)) < len(eligible):
        st.caption(f"Overlap statistics cover {len(matrix)} of {len(eligible)} categorical and "
                   f"numeric columns: at most {MAX_PAIRWISE_ATTRIBUTES} (in column order), and "
                   f"categorical columns with more than {MAX_PAIRWISE_CARDINALITY} values are left "
                   "out.")
    if len(matrix) >= 2:
        labels = list(matrix.columns)
        fig = go.Figure(go.Heatmap(
            z=matrix.values, x=labels, y=labels, zmin=0, zmax=1,
            colorscale=[[0, "#F5F7FA"], [0.5, "#7FA7CF"], [1, "#171C54"]],
            text=matrix.round(2).values, texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="strength"),
        ))
        fig.update_layout(height=max(380, 34 * len(labels) + 120), margin=dict(t=30),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", key="insights_association")
        st.caption("Cramér's V (categorical × categorical), correlation ratio η "
                   "(categorical × numeric) and |r| (numeric × numeric), all bias-corrected; "
                   "1 = one column predicts the other.")
    else:
        st.info("Need at least two categorical or numeric attributes for an association map.")

    pairs = redundant_pairs(frame, infos=infos)
    if pairs:
        st.markdown("**Overlapping columns** — weighting both columns of a pair counts the same "
                    "information twice:")
        for pair in pairs[:12]:
            icon = _RELATION_ICONS.get(pair["relation"], "•")
            st.markdown(f"- {icon} {pair['explanation']}")
        if len(pairs) > 12:
            st.caption(f"… and {len(pairs) - 12} more related pairs.")
    else:
        st.caption("No redundant attribute pairs found.")

    st.markdown("#### Expected head counts per team")
    st.caption("How many students with a value a team of the chosen size gets on average, and "
               "which 'at most / at least' limits are achievable at all.")
    target_options = [n for n, i in infos.items() if i.kind in ("categorical", "multi", "numeric", "text")]
    if not target_options:
        st.info("No attributes to analyse.")
        return
    default_attr = next((n for n in (_extension_ui("default_targets") or ()) if n in target_options),
                        target_options[0])
    c1, c2, c3 = st.columns([2, 1, 1])
    attr = c1.selectbox("Attribute", target_options, index=target_options.index(default_attr),
                        key="insights_target_attr")
    size = c2.number_input("Team size", min_value=2, max_value=20, value=team_size, step=1,
                           key="insights_team_size")
    n_teams = c3.number_input("Teams (0 = from size)", min_value=0, max_value=200,
                              value=min(200, int(config.num_teams or 0)), step=1,
                              key="insights_num_teams")
    info = infos[attr]
    if info.kind == "numeric":
        lo_default = float(info.stats.get("mean", 0.0) + info.stats.get("std", 0.0))
        r1, r2 = st.columns(2)
        lo = r1.number_input("From (inclusive)", value=round(lo_default, 1), key="insights_range_lo")
        hi = r2.number_input("To (inclusive)", value=float(info.stats.get("max", lo_default)),
                             key="insights_range_hi")
        targets = count_rule_targets(frame, attr, int(size), num_teams=int(n_teams) or None,
                                     numeric_range=(lo, hi))
    else:
        targets = count_rule_targets(frame, attr, int(size), num_teams=int(n_teams) or None, top=25)
    if targets:
        st.caption(f"{targets[0]['teams']} teams of typically {targets[0]['team_size']}.")
    st.dataframe(pd.DataFrame([{
        "Value": str(t["value"]) if info.kind != "numeric" else f"{attr} {lo:g}–{hi:g}",
        "Students": t["count"],
        "Share": t["share"],
        "Per team (expected)": round(t["expected_per_team"], 2),
        "Tightest feasible 'at most'": t["feasible_max"],
        "Loosest feasible 'at least'": t["feasible_min"],
        "Violations one tighter": t["violations_if_max_lower"],
        "Advice": t["hint"].split(" → ", 1)[-1],
    } for t in targets]), width="stretch", hide_index=True, column_config={
        "Share": st.column_config.NumberColumn("Share", format="percent"),
        "Advice": st.column_config.TextColumn("Advice", width="large"),
    })


def _apply_candidate(filename: str, text: str) -> Tuple[bool, str]:
    try:
        new_df = DataLoader.parse_csv_text(text)
    except ValueError as exc:
        return False, f"Cannot use this file: {exc}"
    if new_df.empty:
        return False, "Uploaded file is empty or contains no data rows."
    try:
        DataLoader().load_dataframe(new_df.copy())
    except Exception as exc:
        return False, f"Cannot use this file: {exc}"
    set_uploaded_csv(filename, text)
    return True, f"Loaded {len(new_df)} students from {filename} — active on every page."


st.title("📊 Data Overview")
st.markdown("View and manage student data, explore distributions, and edit preferences.")
render_flashes()

with st.expander("ℹ️ What is this page?", expanded=False):
    st.markdown(extensions.text("data_overview.help", """
    This page lets you **explore and manage your data** before forming teams.

    - **Student Table**: View everyone, filter by any categorical attribute
    - **Attribute Insights**: what the allocation engine sees — kinds, coverage, suggested
      rules, related columns and realistic per-team head counts
    - **Distributions**: Charts of every attribute of the data
    - **Edit Preferences**: Set who prefers/avoids working together
    """))

with st.expander("📤 Upload / Replace Data", expanded=False):
    st.caption(extensions.text(
        "data_overview.upload_caption",
        "Upload a CSV or Excel roster: an `identifier` column plus any attributes "
        "you like (colour, city, hobbies ...); every column becomes an attribute."))
    sample_download_button("cohort", label="⬇️ Download sample cohort")
    privacy_notice()
    uploaded_file = st.file_uploader("Student data (CSV or Excel)", type=["csv", "xlsx", "xlsm"],
                                     help="CSV (UTF-8 or Windows-1252, e.g. Excel's CSV export) "
                                          "or an .xlsx/.xlsm workbook. Save legacy .xls files as "
                                          ".xlsx first.")
    if uploaded_file is not None and not refuse_large_table(uploaded_file):
        suffix = Path(uploaded_file.name).suffix.lower()
        try:
            if suffix in (".xlsx", ".xlsm"):
                data = uploaded_file.getvalue()
                sheets = DataLoader.excel_sheet_names(data)
                sheet = st.selectbox("Worksheet", sheets, key="upload_sheet")
                if is_new_upload("student_data", uploaded_file, sheet):
                    ok, message = _apply_candidate(
                        f"{uploaded_file.name} [{sheet}]", _excel_to_csv_text(data, sheet)
                    )
                    if ok:
                        mark_upload_processed("student_data", uploaded_file, sheet)
                        st.success(message)
                    else:
                        st.error(message)
            elif is_new_upload("student_data", uploaded_file):
                text, decode_warnings = DataLoader.decode_bytes(uploaded_file.getvalue())
                ok, message = _apply_candidate(uploaded_file.name, text)
                if ok:
                    mark_upload_processed("student_data", uploaded_file)
                    st.success(message)
                    for decode_warning in decode_warnings:
                        st.warning(decode_warning)
                else:
                    st.error(message)
        except ImportError as exc:
            st.error(f"Cannot read this workbook format ({exc}). Save it as .xlsx or .csv.")
        except Exception as exc:
            st.error(f"Error loading file: {exc}")
    st.caption(f"Active dataset: {get_active_source()}")
    if st.session_state.get("uploaded_csv"):
        if st.button("↩️ Restore default dataset"):
            clear_uploaded_csv()
            flash("Default dataset restored.")
            st.rerun()

loader = get_data_loader()
render_flashes()
df = loader.get_dataframe()
students = loader.get_students()

for load_warning in get_loader_warnings(loader)[:20]:
    st.warning(load_warning)

if df.empty:
    st.warning("No data loaded. Please upload a CSV file above.")
    st.stop()

DISTRIBUTIONS_TAB = "📈 Distributions"
_extra_tabs = _extension_ui("extra_tabs")
extra_tabs = list(_extra_tabs()) if _extra_tabs is not None else []
_tab_labels = ["📋 Student Table", "🔬 Attribute Insights", DISTRIBUTIONS_TAB,
               *[label for label, _ in extra_tabs], "✏️ Edit Preferences"]
_tabs = st.tabs(_tab_labels)
tab1, tab_insights, tab2 = _tabs[:3]
extra_tab_slots = _tabs[3:-1]
tab5 = _tabs[-1]

with tab1:
    st.subheader("Student Data")

    filterable = _categorical_columns(df)
    _filters_hook = _extension_ui("default_filters")
    default_filters = (_filters_hook(filterable) if _filters_hook is not None
                       else [c for c in filterable[:2]])
    filter_columns = st.multiselect(
        "Filter by attributes",
        options=filterable,
        default=default_filters,
        format_func=lambda c: c.replace("_", " ").title(),
        key="filter_columns",
        help="Pick any categorical attribute of the loaded data to filter on",
    )

    filtered_df = df.copy()
    if filter_columns:
        filter_cols = st.columns(min(4, len(filter_columns)))
        for i, col in enumerate(filter_columns):
            values = sorted(v for v in df[col].dropna().astype(str).unique() if v.strip())
            with filter_cols[i % len(filter_cols)]:
                chosen = st.multiselect(
                    col.replace("_", " ").title(), options=values, default=[], key=f"filter_{col}"
                )
            if chosen:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(chosen)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(filtered_df))
    _metrics_hook = _extension_ui("table_metrics")
    if _metrics_hook is not None:
        _metrics_hook(filtered_df, col2, col3, col4)
    else:
        for column, (label, count) in zip((col2, col3, col4), distinct_value_counts(
                filtered_df.drop(columns=[c for c in _NON_CATEGORICAL if c in filtered_df.columns],
                                 errors="ignore"), limit=3)):
            column.metric(label, count)

    _columns_hook = _extension_ui("display_columns")
    if _columns_hook is not None:
        display_cols = _columns_hook(filtered_df)
    else:
        display_cols = ["student_id"] + [
            c for c in filtered_df.columns
            if c != "student_id" and c not in _HIDDEN_COLUMNS
            and filtered_df[c].astype(str).str.strip().replace("nan", "").ne("").any()]

    st.dataframe(filtered_df[display_cols], width="stretch", height=400)

    st.download_button(
        "📥 Download Filtered Data",
        safe_csv(filtered_df),
        "filtered_students.csv",
        "text/csv",
        on_click="ignore",
    )

with tab_insights:
    try:
        render_attribute_insights(df, students, loader)
    except Exception as exc:
        st.warning(f"Attribute insights are unavailable for this dataset: {exc}")

with tab2:
    st.subheader("Data Distributions")

    _distributions_hook = _extension_ui("distributions")
    if _distributions_hook is not None:
        _distributions_hook(df)
    else:
        _charts = st.columns(2)
        _shown = 0
        for _name, _info in infer_attributes(df).items():
            if not _info.usable or _info.kind not in ("categorical", "multi", "numeric"):
                continue
            _label = _name.replace("_", " ").title()
            with _charts[_shown % 2]:
                if _info.kind == "numeric":
                    _values = pd.to_numeric(df[_name], errors="coerce").dropna()
                    _counts = _values.astype(int).value_counts().sort_index() \
                        if (_values == _values.round()).all() else _values.round(1).value_counts().sort_index()
                else:
                    _counts = pd.Series([p for cell in df[_name].dropna() for p in str(cell).split("|") if p]
                                        ).value_counts()
                _fig = px.bar(x=_counts.index.astype(str), y=_counts.values,
                              title=f"{_label} Distribution")
                _fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
                st.plotly_chart(_fig, width="stretch")
            _shown += 1

    if len(filterable) >= 2:
        st.subheader("Cross-tab Explorer")
        xt1, xt2 = st.columns(2)
        row_attr = xt1.selectbox("Rows", filterable, index=0, key="xtab_rows")
        col_options = [c for c in filterable if c != row_attr]
        col_attr = xt2.selectbox("Columns", col_options, index=0, key="xtab_cols")
        crosstab = pd.crosstab(df[row_attr].astype(str), df[col_attr].astype(str))
        st.dataframe(crosstab, width="stretch")

for _slot, (_label, _render) in zip(extra_tab_slots, extra_tabs):
    with _slot:
        _render(df)

with tab5:
    st.subheader("Edit Student Preferences")
    st.info("""
    **Preferences** let students indicate who they want (or don't want) to work with:
    - **Preferred teammates**: The algorithm will try to place these students together (soft constraint)
    - **Avoided teammates**: The algorithm will not place these students together
      (hard or soft — see Configuration → Constraints)
    """)
    st.caption("Edits apply to this browser session only and are kept until the dataset changes.")

    student_ids = [str(s) for s in df["student_id"].tolist()]
    roster_lookup: Dict[str, str] = {}
    for sid in student_ids:
        roster_lookup.setdefault(_canonical(sid), sid)
    selected_student = st.selectbox("Select Student", student_ids)

    if selected_student:
        student_row = df[df["student_id"].astype(str) == selected_student].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Student Info**")
            info_columns = (_extension_ui("info_columns")
                            or [(c.replace("_", " ").title(), c) for c in df.columns
                                if c in infer_attributes(df) and infer_attributes(df)[c].usable])
            for label, column in info_columns:
                if column in df.columns:
                    st.write(f"- {label}: {student_row.get(column) or '—'}")

        prefer_list = [roster_lookup.get(_canonical(sid), sid)
                       for sid in _split_ids(student_row.get("prefer_teammates", ""))]
        avoid_list = [roster_lookup.get(_canonical(sid), sid)
                      for sid in _split_ids(student_row.get("avoid_teammates", ""))]
        with col2:
            st.markdown("**Current Preferences**")
            st.write(f"- Preferred: {', '.join(prefer_list) if prefer_list else 'None'}")
            st.write(f"- Avoided: {', '.join(avoid_list) if avoid_list else 'None'}")
            conflicts = sorted(set(prefer_list) & set(avoid_list))
            if conflicts:
                st.warning(f"Listed as both preferred and avoided: {', '.join(conflicts)}. "
                           "Saving keeps them as preferred only.")
            rostered = set(student_ids)
            unknown = [sid for sid in dict.fromkeys(prefer_list + avoid_list)
                       if sid not in rostered]
            if unknown:
                st.caption(f"Not in the roster (ignored, and dropped when you save): "
                           f"{', '.join(unknown)}.")

        st.markdown("---")
        other_students = [s for s in student_ids if s != selected_student]
        col1, col2 = st.columns(2)
        with col1:
            new_prefer = st.multiselect(
                "Preferred Teammates",
                options=other_students,
                default=[s for s in prefer_list if s in other_students],
                help="Students this person prefers to work with"
            )
        with col2:
            avoid_options = [s for s in other_students if s not in new_prefer]
            new_avoid = st.multiselect(
                "Avoided Teammates",
                options=avoid_options,
                default=[s for s in avoid_list if s in avoid_options],
                help="Students this person wants to avoid"
            )

        if st.button("💾 Save Preferences", type="primary"):
            try:
                loader.update_student_preferences(selected_student, new_prefer, new_avoid)
                flash(f"Preferences updated for {selected_student}")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating preferences: {e}")

        st.markdown("---")
        st.subheader("Bulk Edit")
        with st.expander("Upload Preferences CSV"):
            st.markdown("""
            Upload a CSV with columns: `student_id`, `prefer_teammates`, `avoid_teammates`

            Use pipe `|` to separate multiple IDs. IDs are matched ignoring leading zeros.
            """)
            sample_download_button("preferences", label="⬇️ Download sample preferences")
            pref_file = st.file_uploader("Upload Preferences", type=["csv"], key="pref_upload")
            if pref_file and not refuse_large_table(pref_file):
                try:
                    pref_df = pd.read_csv(pref_file, dtype=str, keep_default_na=False)
                    pref_df.columns = [str(c).strip() for c in pref_df.columns]
                    if "student_id" not in pref_df.columns:
                        st.error("The file needs a student_id column.")
                    else:
                        st.dataframe(pref_df.head(), width="stretch")
                        if st.button("Apply Preferences"):
                            applied, skipped = 0, []
                            dropped_refs: List[str] = []
                            conflicts: List[str] = []
                            split_refs: List[str] = []
                            for _, row in pref_df.iterrows():
                                sid = roster_lookup.get(_canonical(row["student_id"]))
                                if sid is None:
                                    skipped.append(str(row["student_id"]))
                                    continue
                                prefer_all, prefer_dropped, prefer_split = _bulk_refs(
                                    row.get("prefer_teammates", ""), roster_lookup)
                                avoid_all, avoid_dropped, avoid_split = _bulk_refs(
                                    row.get("avoid_teammates", ""), roster_lookup)
                                split_refs.extend(f"{sid}->{r}" for r in prefer_split + avoid_split)
                                prefer = list(dict.fromkeys(prefer_all))
                                avoid: List[str] = []
                                for a in dict.fromkeys(avoid_all):
                                    if a in prefer:
                                        conflicts.append(f"{sid}: {a}")
                                    else:
                                        avoid.append(a)
                                dropped_refs.extend(f"{sid}->{r}" for r in prefer_dropped + avoid_dropped)
                                loader.update_student_preferences(sid, prefer, avoid)
                                applied += 1
                            message = f"Bulk preferences applied to {applied} student(s)."
                            if skipped:
                                message += f" Skipped unknown id(s): {', '.join(skipped[:10])}"
                            if dropped_refs:
                                message += (f" {len(dropped_refs)} reference(s) match no student and were not"
                                            f" applied (e.g. {', '.join(dropped_refs[:10])}).")
                            if conflicts:
                                message += (f" {len(conflicts)} avoid request(s) also listed as a preference"
                                            f" were kept only as preferences (e.g. {', '.join(conflicts[:10])}).")
                            if split_refs:
                                message += (f" {len(split_refs)} reference(s) separate ids with ';' or ','"
                                            f" (e.g. {', '.join(split_refs[:10])}); they were read as separate"
                                            " ids. Separate ids with '|'.")
                            flash(message, "success" if not (skipped or dropped_refs or conflicts or split_refs)
                                  else "warning")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.subheader("🔍 Data Validation")

validator = DataValidator()
is_valid, errors, warnings = validator.validate_dataframe(df, schema_advice=extensions.active())

if is_valid:
    st.success("✅ Data validation passed!")
else:
    st.error("❌ Data validation failed")
    for error in errors:
        st.error(f"• {error}")

if warnings:
    with st.expander(f"⚠️ Warnings ({len(warnings)})"):
        for warning in warnings[:20]:
            st.warning(f"• {warning}")
        if len(warnings) > 20:
            st.info(f"... and {len(warnings) - 20} more warnings")
