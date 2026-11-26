from __future__ import annotations

from typing import Any

__all__ = ["resolve_state"]


def resolve_state() -> Any:
    try:
        import streamlit as st

        return st.session_state
    except Exception:  # pragma: no cover - streamlit always installed here
        return {}
