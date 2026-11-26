from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .. import extensions
from .models import Student
from .student_fields import (
    has_value,
    normalize_student_id,
    parse_categorical_field,
    parse_enum_value,
    parse_int,
    parse_multi_select,
    parse_number,
    parse_ordinal_field,
    parse_ordinal_weight,
    split_ids,
)
from .config_loader import ConfigurationLoader
from .dimension_config import Profile
from .csv_check import unterminated_quote_problem
from .ids import canonical_id, id_collisions

ExcelSource = Union[str, Path, bytes, BinaryIO]

_MAX_EXAMPLES = 5
_OTHER_SEPARATORS = re.compile(r"[;,]")

def _id_match_key(student_id: str) -> str:
    return canonical_id(student_id)


def _normalize_header(name: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


_EXCEL_MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2}) 00:00:00$")


def _excel_text(value: Any) -> str:
    text = str(value).strip()
    match = _EXCEL_MIDNIGHT.match(text)
    return match.group(1) if match else text


class _NoExtension:
    def optional_columns(self) -> List[str]:
        return []

    def header_aliases(self) -> Dict[str, str]:
        return {}

    def issue_templates(self) -> Dict[str, str]:
        return {}

    def accepted_values(self, column: str) -> Optional[List[str]]:
        return None

    def free_form_names(self) -> Dict[str, str]:
        return {}

    def canonicalize(self, df: pd.DataFrame, warnings: Optional[List[str]] = None) -> pd.DataFrame:
        return df

    def add_missing_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame

    def row_context(self, columns: Iterable[Any]) -> Any:
        return None

    def parse_head(self, row: pd.Series, sid: str, issues: Any, context: Any) -> Dict[str, Any]:
        return {}

    def parse_typed(self, row: pd.Series, sid: str, issues: Any) -> Dict[str, Any]:
        return {}

    def parse_tail(self, row: pd.Series, sid: str, issues: Any,
                   dynamic_numeric: Dict[str, float]) -> None:
        return None

    def statistics(self, distribution, unique_count) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {}, {}


class DataLoader:
    REQUIRED_COLUMNS = [
        "student_id"
        ]

    OPTIONAL_COLUMNS = [
        "group_number",
        "prefer_teammates", "avoid_teammates",
        ]

    HEADER_ALIASES = {
        "identifier": "student_id",
        "id": "student_id",
        "studentid": "student_id",
        "student_number": "student_id",
        "student_no": "student_id",
        "prefer": "prefer_teammates",
        "avoid": "avoid_teammates",
    }

    WEAK_HEADER_ALIASES = {
        "student": "student_id",
    }

    def __init__(self,
                 config_loader: Optional[ConfigurationLoader] = None,
                 profile_name: Optional[str] = "default",
                 **ext_options: Any):
        factory = extensions.hook("loader_extension", None)
        self._ext = factory(self, **ext_options) if factory is not None else _NoExtension()
        self._df: Optional[pd.DataFrame] = None
        self._students: List[Student] = []
        self.warnings: List[str] = []
        self._profile_warnings: List[str] = []

        self.config_loader = config_loader
        self.profile_name = profile_name or "default"
        self._profile: Optional[Profile] = None
        self._dimension_config: Dict[str, Any] = {}

        if self.config_loader:
            self._load_profile()

    def _load_profile(self) -> None:
        if not self.config_loader:
            return
        self._profile_warnings = []
        try:
            self._profile = self.config_loader.load_profile(self.profile_name)
            self._dimension_config = {
                dim.name: dim.to_dict() for dim in self._profile.dimensions
            }
        except FileNotFoundError:
            self._dimension_config = {}
            self._profile_warnings.append(
                f"Profile '{self.profile_name}' not found; profile dimensions"
                + extensions.text("loader.dimension_examples", "")
                + " are not parsed."
            )
        except Exception as exc:
            self._dimension_config = {}
            self._profile_warnings.append(
                f"Profile '{self.profile_name}' could not be loaded ({exc}); "
                "profile dimensions are not parsed."
            )
        self.warnings = list(self._profile_warnings)

    @property
    def dimension_config(self) -> Dict[str, Any]:
        return dict(self._dimension_config)

    @staticmethod
    def decode_bytes(raw: bytes) -> Tuple[str, List[str]]:
        try:
            return raw.decode("utf-8-sig"), []
        except UnicodeDecodeError:
            return raw.decode("cp1252", errors="replace"), [
                "File is not UTF-8; decoded as Windows-1252 (cp1252). Check "
                "accented names."
            ]

    @staticmethod
    def sniff_delimiter(text: str) -> str:
        header = ""
        for line in text.splitlines():
            if line.strip():
                header = line
                break
        unquoted = re.sub(r'"[^"]*"', "", header)
        counts = {d: unquoted.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else ","

    @staticmethod
    def parse_csv_text(text: str, delimiter: Optional[str] = None) -> pd.DataFrame:
        text = text.lstrip("﻿")
        delimiter = delimiter or DataLoader.sniff_delimiter(text)
        problem = unterminated_quote_problem(text, delimiter)
        if problem:
            raise ValueError(f"Malformed CSV, {problem}. Fix or remove the stray quote.")
        rows = [
            [str(c).strip() for c in r]
            for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(str(c).strip() for c in r)
        ]
        if not rows:
            return pd.DataFrame()

        header_len = len(rows[0])
        while header_len > 1 and not rows[0][header_len - 1]:
            header_len -= 1
        short = sum(1 for r in rows[1:] if len(r) < header_len)
        long_rows = sum(1 for r in rows[1:] if any(r[header_len:]))

        ncols = max(len(r) for r in rows)
        padded = [r + [""] * (ncols - len(r)) for r in rows]
        frame = DataLoader._frame_from_rows(padded[0], padded[1:])
        warnings: List[str] = []
        if short:
            warnings.append(
                f"{short} row(s) have fewer values than the header; missing "
                "trailing values were left blank — check that columns line up."
            )
        if long_rows:
            warnings.append(
                f"{long_rows} row(s) have more values than the header; the "
                "extra values were kept in generated columns — check that "
                "columns line up."
            )
        frame.attrs["parse_warnings"] = warnings
        return frame

    @staticmethod
    def _frame_from_rows(header: List[str], data: List[List[str]]) -> pd.DataFrame:
        columns: List[str] = []
        keep_idx: List[int] = []
        seen: Dict[str, int] = {}
        for i, raw_name in enumerate(header):
            name = str(raw_name).strip()
            has_data = any(str(r[i]).strip() for r in data)
            if (not name or name.lower().startswith("unnamed")) and not has_data:
                continue
            if not name or name.lower().startswith("unnamed"):
                name = f"column_{i}"
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                name = f"{name}_{seen[name]}"
            columns.append(name)
            keep_idx.append(i)
        return pd.DataFrame([[r[i] for i in keep_idx] for r in data], columns=columns)

    @staticmethod
    def _excel_handle(source: ExcelSource):
        if isinstance(source, (bytes, bytearray)):
            return io.BytesIO(source)
        if hasattr(source, "read") and hasattr(source, "seek"):
            source.seek(0)
        return source

    @staticmethod
    def excel_sheet_names(source: ExcelSource) -> List[str]:
        with pd.ExcelFile(DataLoader._excel_handle(source), engine="openpyxl") as book:
            return list(book.sheet_names)

    @staticmethod
    def read_excel_frame(source: ExcelSource, sheet_name: Optional[Union[str, int]] = None) -> pd.DataFrame:
        raw = pd.read_excel(
            DataLoader._excel_handle(source),
            sheet_name=0 if sheet_name is None else sheet_name,
            dtype=str, keep_default_na=False, header=None, engine="openpyxl",
        )
        rows = [[_excel_text(v) for v in row] for row in raw.itertuples(index=False)]
        rows = [r for r in rows if any(r)]
        if not rows:
            return pd.DataFrame()
        return DataLoader._frame_from_rows(rows[0], rows[1:])

    def load_csv(self, file_path: str | Path) -> List[Student]:
        raw = Path(file_path).read_bytes()
        text, decode_warnings = self.decode_bytes(raw)
        frame = self.parse_csv_text(text)
        frame.attrs["parse_warnings"] = decode_warnings + list(frame.attrs.get("parse_warnings", []))
        return self._load_frame(frame)

    def load_excel(self, source: ExcelSource, sheet_name: Optional[Union[str, int]] = None) -> List[Student]:
        return self._load_frame(self.read_excel_frame(source, sheet_name))

    def load_dataframe(self, df: pd.DataFrame) -> List[Student]:
        frame = df.copy()
        frame.attrs["parse_warnings"] = list(df.attrs.get("parse_warnings", []))
        return self._load_frame(frame)

    def get_dataframe(self) -> pd.DataFrame:
        return self._df.copy() if self._df is not None else pd.DataFrame()

    def get_students(self) -> List[Student]:
        return self._students.copy()

    def _load_frame(self, df: pd.DataFrame) -> List[Student]:
        warnings: List[str] = list(self._profile_warnings)
        warnings.extend(df.attrs.get("parse_warnings", []))
        if len(df.columns) == 0:
            raise ValueError("The file is empty: it has no header row and no students.")
        df = self._normalize_headers(df)
        df = self._clean_cells(df)
        df = self._ext.canonicalize(df, warnings)
        self._validate_columns(df)
        df = self._normalize_ids(df, warnings)
        df = self._add_missing_columns(df)
        students = self._parse_students(df, warnings)
        self._resolve_preference_ids(students, warnings)
        self._df = df
        self._students = students
        self.warnings = warnings
        return students

    def _normalize_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        known = (set(self.REQUIRED_COLUMNS) | set(self.OPTIONAL_COLUMNS)
                 | set(self._ext.optional_columns()) | set(self._dimension_config))
        known_by_norm = {_normalize_header(k): k for k in known}
        aliases = {**self.HEADER_ALIASES, **self._ext.header_aliases()}
        existing = set(str(c) for c in df.columns)
        renames: Dict[Any, str] = {}
        for weak in (False, True):
            for column in df.columns:
                if column in renames:
                    continue
                norm = _normalize_header(column)
                if weak:
                    target = self.WEAK_HEADER_ALIASES.get(norm)
                else:
                    target = aliases.get(norm) or known_by_norm.get(norm)
                    if target is None and norm.startswith("team_project_"):
                        target = norm
                if target is None or target == column:
                    continue
                # INVARIANT: never merge two columns. If the canonical name already
                # exists (or another column was mapped to it), keep this one unchanged.
                if target in existing or target in renames.values():
                    continue
                renames[column] = target
        return df.rename(columns=renames) if renames else df

    @staticmethod
    def _clean_cells(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].map(lambda v: v.strip() if isinstance(v, str) else v)
        return df

    def _validate_columns(self, df: Optional[pd.DataFrame] = None) -> None:
        frame = self._df if df is None else df
        missing = set(self.REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Name the id column "
                "'student_id' (or 'identifier' / 'Student ID')."
            )

    def _normalize_ids(self, df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        df = df.copy()
        ids = [normalize_student_id(v) for v in df["student_id"]]
        blank_rows = [i + 2 for i, sid in enumerate(ids) if sid is None]
        df["student_id"] = ids
        if blank_rows:
            df = df[df["student_id"].notna()].reset_index(drop=True)
            shown = ", ".join(str(r) for r in blank_rows[:_MAX_EXAMPLES])
            warnings.append(
                f"{len(blank_rows)} row(s) without a student id were skipped "
                f"(data row(s) {shown}{'…' if len(blank_rows) > _MAX_EXAMPLES else ''})."
            )
        counts = Counter(df["student_id"])
        duplicates = sorted(sid for sid, c in counts.items() if c > 1)
        if duplicates:
            shown = ", ".join(duplicates[:10])
            raise ValueError(
                f"Duplicate student ids: {shown}"
                + (f" (+{len(duplicates) - 10} more)" if len(duplicates) > 10 else "")
                + ". Every student needs a unique id."
            )
        collisions = id_collisions(df["student_id"])
        if collisions:
            shown = "; ".join(" and ".join(group) for group in collisions[:10])
            raise ValueError(
                f"Student ids that differ only by leading zeros: {shown}"
                + (f" (+{len(collisions) - 10} more)" if len(collisions) > 10 else "")
                + ". Ids are matched across files ignoring leading zeros, so each "
                "student needs an id that is unique without them."
            )
        return df

    def _add_missing_columns(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        frame = (self._df if df is None else df).copy()
        for column in ("prefer_teammates", "avoid_teammates"):
            if column not in frame.columns:
                frame[column] = ""
        frame = self._ext.add_missing_columns(frame)
        for column in ("prefer_teammates", "avoid_teammates"):
            if column in frame.columns:
                frame[column] = frame[column].map(lambda v: "" if not self._has_value(v) else v)

        if "group_number" not in frame.columns:
            frame["group_number"] = None
        if df is None:
            self._df = frame
        return frame

    def _parse_students(self, df: Optional[pd.DataFrame] = None,
                        warnings: Optional[List[str]] = None) -> List[Student]:
        frame = self._df if df is None else df
        sink = self.warnings if warnings is None else warnings
        issues = self._issue_collector()
        context = self._ext.row_context(frame.columns)

        students: List[Student] = []
        for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
            try:
                students.append(self._parse_row(row, issues, context=context))
            except Exception as exc:
                issues.add("row_errors", f"{row.get('student_id', f'row {row_number}')}: {exc}")
        sink.extend(issues.messages())
        return students

    def _issue_collector(self) -> "_IssueCollector":
        return _IssueCollector(templates=self._ext.issue_templates(),
                               accepted_values=self._ext.accepted_values,
                               free_form_names=self._ext.free_form_names())

    def _parse_row(self, row: pd.Series, issues: Optional["_IssueCollector"] = None,
                   context: Any = None) -> Student:
        issues = issues or self._issue_collector()
        sid = normalize_student_id(row.get("student_id"))
        if sid is None:
            raise ValueError("missing student id")
        if context is None:
            context = self._ext.row_context(row.index)

        typed: Dict[str, Any] = dict(self._ext.parse_head(row, sid, issues, context))
        prefer = self._split_ids(row.get("prefer_teammates"))
        avoid = self._split_ids(row.get("avoid_teammates"))

        group_num = None
        if self._has_value(row.get("group_number")):
            group_num = self._parse_int(row.get("group_number"))
            if group_num is None:
                issues.add("group_number", f"{sid}: {row.get('group_number')!r}")

        typed.update(self._ext.parse_typed(row, sid, issues))

        dynamic_categorical: Dict[str, str] = {}
        dynamic_ordinal: Dict[str, Any] = {}
        dynamic_numeric: Dict[str, float] = {}
        dynamic_multi_select: Dict[str, List[str]] = {}
        for dim_name, dim_config in self._dimension_config.items():
            if dim_name not in row.index or not self._has_value(row[dim_name]):
                continue
            self._parse_dimension(dim_name, dim_config, row[dim_name], issues,
                                  dynamic_categorical, dynamic_ordinal,
                                  dynamic_numeric, dynamic_multi_select)

        self._ext.parse_tail(row, sid, issues, dynamic_numeric)

        return Student(
            student_id=sid,
            prefer_teammates=prefer,
            avoid_teammates=avoid,
            group_number=group_num,
            previous_teams=self._parse_previous_teams(row),
            dynamic_categorical=dynamic_categorical,
            dynamic_ordinal=dynamic_ordinal,
            dynamic_numeric=dynamic_numeric,
            dynamic_multi_select=dynamic_multi_select,
            **typed,
        )

    def _parse_dimension(self, dim_name, dim_config, value, issues,
                         categorical, ordinal, numeric, multi) -> None:
        dim_type = dim_config.get("type")
        if dim_type == "categorical":
            parsed = self._parse_categorical(value, dim_config.get("categories") or [])
            if parsed is None:
                return
            categories = dim_config.get("categories") or []
            if categories and parsed not in categories:
                issues.add(f"category:{dim_name}", parsed)
            categorical[dim_name] = parsed
        elif dim_type == "ordinal":
            weight = parse_ordinal_weight(value, dim_config.get("levels") or [])
            if weight is None:
                issues.add(f"level:{dim_name}", str(value).strip())
            else:
                ordinal[dim_name] = weight
        elif dim_type == "numeric":
            number = self._parse_numeric(value)
            if number is None:
                issues.add(f"numeric:{dim_name}", str(value).strip())
                return
            lo, hi = dim_config.get("min_value"), dim_config.get("max_value")
            if (lo is not None and number < lo) or (hi is not None and number > hi):
                issues.add(f"range:{dim_name}", str(value).strip())
                return
            numeric[dim_name] = number
        elif dim_type == "multi_select":
            allowed = dim_config.get("values") or []
            limit = dim_config.get("max_selections")
            items = []
            for item in self._parse_multi_select(value, None):
                match = next((a for a in allowed if a.lower() == item.lower()), None)
                if allowed and match is None:
                    issues.add(f"category:{dim_name}", item)
                items.append(match or item)
            if limit is not None and len(items) > limit:
                issues.add(f"truncated:{dim_name}", str(len(items)))
                items = items[:limit]
            if items:
                multi[dim_name] = items

    def _resolve_preference_ids(self, students: List[Student], warnings: List[str]) -> None:
        by_key = {_id_match_key(s.student_id): s.student_id for s in students}
        unknown: List[str] = []
        split: List[str] = []
        for student in students:
            for attr in ("prefer_teammates", "avoid_teammates"):
                resolved = []
                for ref in getattr(student, attr):
                    match = by_key.get(_id_match_key(ref))
                    parts = ([normalize_student_id(p) for p in _OTHER_SEPARATORS.split(ref)]
                             if match is None and _OTHER_SEPARATORS.search(ref) else [])
                    matches = [by_key.get(_id_match_key(p)) for p in parts if p]
                    if matches and all(matches):
                        split.append(f"{student.student_id}->{ref}")
                        resolved.extend(m for m in matches if m != student.student_id)
                    elif match is None:
                        unknown.append(f"{student.student_id}->{ref}")
                        resolved.append(ref)
                    elif match != student.student_id:
                        resolved.append(match)
                setattr(student, attr, list(dict.fromkeys(resolved)))
        if split:
            warnings.append(
                f"{len(split)} teammate preference cell(s) separate ids with ';' or ',' "
                f"(e.g. {', '.join(split[:_MAX_EXAMPLES])}); they were read as separate ids. "
                "Separate ids with '|'."
            )
        if unknown:
            hint = (" Separate several ids with '|'."
                    if any(_OTHER_SEPARATORS.search(u.split("->", 1)[1]) for u in unknown) else "")
            warnings.append(
                f"{len(unknown)} teammate preference(s) refer to unknown student ids "
                f"(e.g. {', '.join(unknown[:_MAX_EXAMPLES])}); they have no effect.{hint}"
            )

    _has_value = staticmethod(has_value)
    _parse_int = staticmethod(parse_int)
    _split_ids = staticmethod(split_ids)
    _parse_categorical = staticmethod(parse_categorical_field)
    _parse_ordinal = staticmethod(parse_ordinal_field)
    _parse_numeric = staticmethod(parse_number)
    _parse_multi_select = staticmethod(parse_multi_select)

    def _parse_previous_teams(self, row: pd.Series) -> Dict[str, str]:
        previous_teams = {}
        for col in row.index:
            if str(col).startswith("team_project_"):
                value = row[col]
                if self._has_value(value):
                    previous_teams[str(col)[len("team_project_"):]] = str(value).strip()
        return previous_teams

    def _parse_enum(self, enum_class, value, default):
        parsed = parse_enum_value(enum_class, value)
        return default if parsed is None else parsed

    def export_to_csv(self, file_path: str | Path) -> None:
        if self._df is None:
            raise ValueError("No data loaded")
        self._df.to_csv(file_path, index=False)

    def export_students_to_dataframe(self, students: List[Student]) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in students])

    def _row_mask(self, student_id: Any) -> pd.Series:
        key = _id_match_key(normalize_student_id(student_id) or "")
        return self._df["student_id"].map(lambda v: _id_match_key(str(v)) == key)

    def update_student_preferences(
        self,
        student_id: Any,
        prefer: Iterable[Any],
        avoid: Iterable[Any]
    ) -> None:
        if self._df is None:
            raise ValueError("No data loaded")
        mask = self._row_mask(student_id)
        if not mask.any():
            raise ValueError(f"Student {student_id} not found")

        by_key = {_id_match_key(s.student_id): s.student_id for s in self._students}
        own = _id_match_key(normalize_student_id(student_id) or "")

        def clean(values: Iterable[Any]) -> List[str]:
            result: List[str] = []
            for value in values or []:
                sid = normalize_student_id(value)
                if sid:
                    canonical = by_key.get(_id_match_key(sid), sid)
                    if _id_match_key(canonical) != own and canonical not in result:
                        result.append(canonical)
            return result

        prefer_ids, avoid_ids = clean(prefer), clean(avoid)
        self._df.loc[mask, "prefer_teammates"] = "|".join(prefer_ids)
        self._df.loc[mask, "avoid_teammates"] = "|".join(avoid_ids)

        key = _id_match_key(normalize_student_id(student_id) or "")
        for student in self._students:
            if _id_match_key(student.student_id) == key:
                student.prefer_teammates = prefer_ids
                student.avoid_teammates = avoid_ids
                break

    def get_unique_values(self, column: str) -> List[Any]:
        if self._df is None or column not in self._df.columns:
            return []
        return self._df[column].dropna().unique().tolist()

    def update_team_assignments(self, project_id: str, teams: List) -> None:
        if self._df is None:
            raise ValueError("No data loaded")
        col_name = f"team_project_{project_id}"
        if col_name not in self._df.columns:
            self._df[col_name] = ""
        for team in teams:
            for member in team.members:
                mask = self._df["student_id"] == member.student_id
                self._df.loc[mask, col_name] = team.team_id
                for student in self._students:
                    if student.student_id == member.student_id:
                        student.previous_teams[project_id] = team.team_id
                        break

    def get_previous_teammates(self, student_id: str) -> Dict[str, List[str]]:
        teammates: Dict[str, List[str]] = {}
        if self._df is None:
            return teammates
        student_mask = self._df["student_id"] == student_id
        if not student_mask.any():
            return teammates
        for col in [c for c in self._df.columns if str(c).startswith("team_project_")]:
            student_team = self._df.loc[student_mask, col].iloc[0]
            if not self._has_value(student_team):
                continue
            team_mask = self._df[col] == student_team
            teammates[str(col)[len("team_project_"):]] = (
                self._df.loc[team_mask & ~student_mask, "student_id"].tolist()
            )
        return teammates

    def get_statistics(self) -> Dict[str, Any]:
        if self._df is None:
            return {}

        team_cols = [c for c in self._df.columns if str(c).startswith("team_project_")]
        df = self._df

        def distribution(col: str) -> Dict[str, Any]:
            if col not in df.columns:
                return {}
            s = df[col].dropna().astype(str)
            return s[s.str.strip() != ""].value_counts().to_dict()

        def unique_count(col: str) -> int:
            if col not in df.columns:
                return 0
            s = df[col].dropna().astype(str)
            return int(s[s.str.strip() != ""].nunique())

        head, tail = self._ext.statistics(distribution, unique_count)
        return {
            "total_students": len(self._students),
            **head,
            "previous_projects": len(team_cols),
            **tail,
            "load_warnings": list(self.warnings),
        }


def _typed_field_hint(column: str, accepted: Optional[List[str]] = None,
                      free_form_names: Optional[Dict[str, str]] = None) -> str:
    accepted_text = (f" only accepts {', '.join(accepted)}"
                     if accepted is not None else " only accepts the levels of its profile dimension")
    suggestion = (free_form_names or {}).get(column, column + "_text")
    return (f"'{column}'{accepted_text}; for free-form data (or several values separated by "
            f"'|') use another column name, e.g. '{suggestion}'.")


class _IssueCollector:
    _TEMPLATES = {
        "group_number": "{n} group_number value(s) are not whole numbers and were "
                        "ignored (e.g. {ex}).",
        "row_errors": "{n} row(s) could not be parsed and were skipped (e.g. {ex}).",
    }

    def __init__(self, templates: Optional[Dict[str, str]] = None,
                 accepted_values=None,
                 free_form_names: Optional[Dict[str, str]] = None) -> None:
        self._items: Dict[str, List[str]] = defaultdict(list)
        self._templates = {**self._TEMPLATES, **(templates or {})}
        self._accepted_values = accepted_values or (lambda _column: None)
        self._free_form_names = dict(free_form_names or {})

    def _hint(self, column: str) -> str:
        return _typed_field_hint(column, self._accepted_values(column), self._free_form_names)

    def add(self, kind: str, example: str) -> None:
        self._items[kind].append(example)

    def messages(self) -> List[str]:
        lines = []
        for kind, examples in self._items.items():
            uniq = list(dict.fromkeys(examples))
            ex = ", ".join(repr(e) if kind.split(":")[0] in (
                "enum", "category", "level", "numeric", "range") else str(e)
                for e in uniq[:_MAX_EXAMPLES])
            if len(uniq) > _MAX_EXAMPLES:
                ex += ", …"
            n = len(examples)
            prefix, _, column = kind.partition(":")
            if kind in self._templates:
                lines.append(self._templates[kind].format(n=n, ex=ex))
            elif prefix == "enum":
                lines.append(f"{column}: {n} value(s) not recognised ({ex}); left empty. "
                             + self._hint(column))
            elif prefix == "category":
                lines.append(f"{column}: {n} value(s) are not in the profile's category "
                             f"list ({ex}); kept as-is.")
            elif prefix == "level":
                lines.append(f"{column}: {n} value(s) are not levels of this ordinal "
                             f"dimension ({ex}); left empty. " + self._hint(column))
            elif prefix == "numeric":
                lines.append(f"{column}: {n} value(s) are not numbers ({ex}); left empty.")
            elif prefix == "range":
                lines.append(f"{column}: {n} value(s) are outside the allowed range "
                             f"({ex}); left empty.")
            elif prefix == "truncated":
                lines.append(f"{column}: {n} student(s) selected more values than "
                             "allowed; extra selections were dropped.")
            else:
                lines.append(f"{kind}: {n} issue(s) ({ex}).")
        return lines
