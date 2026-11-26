from __future__ import annotations
from typing import List, Optional, Tuple
import pandas as pd

from .. import extensions


class DataValidator:
    REQUIRED_COLUMNS = ["student_id"]

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    @staticmethod
    def _has_value(value) -> bool:
        return pd.notna(value) and str(value).strip() != ""
    
    def validate_dataframe(self, df: pd.DataFrame, *,
                           schema_advice: Optional[bool] = None) -> Tuple[bool, List[str], List[str]]:
        self.errors = []
        self.warnings = []
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
        
        checks = extensions.hook("validator_checks", None)
        if schema_advice is None:
            schema_advice = extensions.active()
        if checks is not None and schema_advice:
            self.warnings.extend(checks.absent_advice(df))


        if df.empty:
            self.errors.append("DataFrame is empty")
            return False, self.errors, self.warnings

        for idx, row in df.iterrows():
            self._validate_row(row, idx)

        if "student_id" in df.columns:
            duplicates = df[df.duplicated(subset=["student_id"], keep=False)]
            if not duplicates.empty:
                dup_ids = duplicates["student_id"].unique().tolist()
                self.errors.append(f"Duplicate student IDs: {dup_ids}")

        self._validate_preferences(df)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_row(self, row: pd.Series, idx: int) -> None:
        row_id = row.get("student_id", f"row {idx}")
        
        checks = extensions.hook("validator_checks", None)
        if checks is not None:
            checks.check_row(row, row_id, self._has_value, self.warnings)

    def _validate_preferences(self, df: pd.DataFrame) -> None:
        if "student_id" not in df.columns:
            return
        
        valid_ids = set(df["student_id"].astype(str))
        
        for pref_col in ["prefer_teammates", "avoid_teammates"]:
            if pref_col not in df.columns:
                continue
            
            for idx, row in df.iterrows():
                pref_str = str(row.get(pref_col, ""))
                if not pref_str or pref_str == "nan":
                    continue
                
                for ref_id in pref_str.split("|"):
                    ref_id = ref_id.strip()
                    if ref_id and ref_id not in valid_ids:
                        self.warnings.append(
                            f"{row['student_id']}: {pref_col} references unknown ID '{ref_id}'"
                        )

