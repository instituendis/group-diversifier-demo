from __future__ import annotations

import logging
import math
import os
import threading
from typing import Optional
from urllib.parse import urlparse

__all__ = [
    "BUILTIN_PRIVACY_NOTICE",
    "CLASSIC_UNAVAILABLE",
    "DEFAULT_SOURCE_URL",
    "ENV_HOSTED",
    "ENV_MAX_CLASSIC_TIME",
    "ENV_MAX_CONCURRENT_RUNS",
    "ENV_MAX_TABLE_UPLOAD_MB",
    "ENV_MAX_TIME_LIMIT",
    "ENV_MAX_WORKERS",
    "ENV_PRIVACY_NOTICE",
    "ENV_PRIVACY_URL",
    "ENV_SOURCE_URL",
    "HOSTED_DEFAULT_MAX_CLASSIC_TIME",
    "HOSTED_DEFAULT_MAX_CONCURRENT_RUNS",
    "HOSTED_DEFAULT_MAX_TABLE_UPLOAD_MB",
    "HOSTED_DEFAULT_MAX_TIME_LIMIT",
    "HOSTED_DEFAULT_MAX_WORKERS",
    "PROFILES_READ_ONLY",
    "clamp_time_limit",
    "clamp_workers",
    "hosted",
    "max_classic_time",
    "max_concurrent_runs",
    "max_table_upload_mb",
    "max_time_limit",
    "max_workers",
    "operator_privacy_notice",
    "privacy_notice_markdown",
    "privacy_url",
    "source_url",
    "table_upload_problem",
]

ENV_HOSTED = "GROUP_DIVERSIFIER_HOSTED"
ENV_SOURCE_URL = "GROUP_DIVERSIFIER_SOURCE_URL"
ENV_PRIVACY_NOTICE = "GROUP_DIVERSIFIER_PRIVACY_NOTICE"
ENV_PRIVACY_URL = "GROUP_DIVERSIFIER_PRIVACY_URL"
ENV_MAX_WORKERS = "GROUP_DIVERSIFIER_MAX_WORKERS"
ENV_MAX_CONCURRENT_RUNS = "GROUP_DIVERSIFIER_MAX_CONCURRENT_RUNS"
ENV_MAX_TIME_LIMIT = "GROUP_DIVERSIFIER_MAX_TIME_LIMIT"
ENV_MAX_TABLE_UPLOAD_MB = "GROUP_DIVERSIFIER_MAX_TABLE_UPLOAD_MB"
ENV_MAX_CLASSIC_TIME = "GROUP_DIVERSIFIER_MAX_CLASSIC_TIME"

DEFAULT_SOURCE_URL = "https://github.com/instituendis/group-diversifier-demo"

HOSTED_DEFAULT_MAX_WORKERS = 1
HOSTED_DEFAULT_MAX_CONCURRENT_RUNS = 1
HOSTED_DEFAULT_MAX_TIME_LIMIT = 120
HOSTED_DEFAULT_MAX_TABLE_UPLOAD_MB = 5
HOSTED_DEFAULT_MAX_CLASSIC_TIME = 120

_TRUE = frozenset({"1", "true", "yes", "on"})

CLASSIC_UNAVAILABLE = ("the classic algorithms (greedy, simulated annealing, genetic) are not "
                       "available on this hosted server, so the rules engine is used")

PROFILES_READ_ONLY = ("Profiles are read-only on this hosted server: they only matter to the "
                      "classic algorithms, and a saved profile would be shared with every "
                      "user of the server.")

BUILTIN_PRIVACY_NOTICE = (
    "**How this app handles your data.** "
    "Files you upload (student rosters, earlier groups, participants, preferences, session "
    "files) and the results you create are held in the server's working memory; this "
    "program keeps no database and saves nothing to files of its own (the web server may "
    "buffer an upload larger than 1 MB in a temporary file while it receives it). After "
    "you close the tab, your session can be reopened for about two minutes and then no "
    "longer by anyone, but copies of files you uploaded or downloaded can remain in the "
    "server's memory until the server is restarted. Nothing is kept for you: use "
    "**⬇️ Download session** to keep your work. Other users of this "
    "server cannot see your session. The program makes no outbound network calls with "
    "your data. Rosters may hold personal data: upload "
    "only what the allocation needs. Downloads (CSV, Excel, JSON, LMS group files, the "
    "session file) contain student ids and roster columns; the printable HTML report can "
    "replace ids with pseudonyms (**Anonymize student ids**). Access to this server, its "
    "logs and its backups are the operator's responsibility and are covered by the "
    "operator's privacy statement, not by this notice."
)


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def hosted() -> bool:
    return _env(ENV_HOSTED).lower() in _TRUE


_LINK_BREAKERS = frozenset("()[]<>\"'`")


def _http_url(value: str) -> Optional[str]:
    if not value or any(ch.isspace() or ch in _LINK_BREAKERS for ch in value):
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None
    return value


_LOGGER = logging.getLogger(__name__)
_SOURCE_WARNING_LOCK = threading.Lock()
_source_warning_logged = False


def _warn_default_source_once() -> None:
    global _source_warning_logged
    with _SOURCE_WARNING_LOCK:
        if _source_warning_logged:
            return
        _source_warning_logged = True
    _LOGGER.warning(
        "Hosted mode: %s is not set (or not a usable http(s) URL), so the 'Source code' link "
        "points at the default %s. Under AGPL-3.0 section 13 the users of this server must be "
        "able to get the source of the version that is running: set %s to a URL they can "
        "open, and to your modified source if you changed the program.",
        ENV_SOURCE_URL, DEFAULT_SOURCE_URL, ENV_SOURCE_URL)


def source_url() -> str:
    url = _http_url(_env(ENV_SOURCE_URL))
    if url is not None:
        return url
    if hosted():
        _warn_default_source_once()
    return DEFAULT_SOURCE_URL


def privacy_url() -> Optional[str]:
    return _http_url(_env(ENV_PRIVACY_URL))


def operator_privacy_notice() -> str:
    return _env(ENV_PRIVACY_NOTICE)


def privacy_notice_markdown() -> str:
    parts = [BUILTIN_PRIVACY_NOTICE]
    extra = operator_privacy_notice()
    if extra:
        parts.append(extra)
    link = privacy_url()
    if link:
        parts.append(f"[Privacy statement of this server]({link})")
    return "\n\n".join(parts)


def _positive_int(name: str) -> Optional[int]:
    raw = _env(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def max_workers() -> Optional[int]:
    value = _positive_int(ENV_MAX_WORKERS)
    if value is not None:
        return value
    return HOSTED_DEFAULT_MAX_WORKERS if hosted() else None


def max_concurrent_runs() -> Optional[int]:
    value = _positive_int(ENV_MAX_CONCURRENT_RUNS)
    if value is not None:
        return value
    return HOSTED_DEFAULT_MAX_CONCURRENT_RUNS if hosted() else None


def max_time_limit() -> Optional[int]:
    value = _positive_int(ENV_MAX_TIME_LIMIT)
    if value is not None:
        return value
    return HOSTED_DEFAULT_MAX_TIME_LIMIT if hosted() else None


def clamp_time_limit(requested: object) -> float:
    cap = max_time_limit()
    try:
        value = float(requested)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = math.nan
    if cap is None:
        return value
    if not math.isfinite(value) or value <= 0:
        return float(cap)
    return min(value, float(cap))


def max_classic_time() -> Optional[int]:
    value = _positive_int(ENV_MAX_CLASSIC_TIME)
    if value is not None:
        return value
    return HOSTED_DEFAULT_MAX_CLASSIC_TIME if hosted() else None


def max_table_upload_mb() -> Optional[int]:
    value = _positive_int(ENV_MAX_TABLE_UPLOAD_MB)
    if value is not None:
        return value
    return HOSTED_DEFAULT_MAX_TABLE_UPLOAD_MB if hosted() else None


def table_upload_problem(size_bytes: object) -> Optional[str]:
    cap = max_table_upload_mb()
    if cap is None:
        return None
    try:
        size = int(size_bytes)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    if size <= cap * 1024 * 1024:
        return None
    return (f"This file is {size / (1024 * 1024):.1f} MB; this server reads tables of at most "
            f"{cap} MB (a cohort file is usually well under 1 MB). Upload a smaller file, for "
            f"example only the columns the allocation needs.")


def clamp_workers(requested: object) -> int:
    try:
        value = int(requested)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        value = 1
    value = max(1, value)
    cap = max_workers()
    return min(value, cap) if cap is not None else value
