from __future__ import annotations

import csv
import io
from typing import Optional


def unterminated_quote_problem(text: str, delimiter: str = ",") -> Optional[str]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
    last_line = 0
    while True:
        try:
            next(reader)
        except StopIteration:
            return None
        except csv.Error as exc:
            if "end of data" in str(exc):
                line = last_line + 1
                return (f"line {line}: a quoted value starting here is never closed (unbalanced "
                        'quote "), so the rows after it would be merged into it')
        last_line = reader.line_num
