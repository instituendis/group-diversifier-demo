import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import streamlit as st

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import __version__, demo, extensions  # noqa: E402
from app.core import engine_session as es  # noqa: E402
from app.core import run_history as rh  # noqa: E402
from app.core import session_store  # noqa: E402
from app.core.config_loader import ConfigurationLoader  # noqa: E402
from app.core.data_session import (  # noqa: E402
    ensure_session_state,
    get_active_source,
    get_config_dir,
    get_data_loader,
    is_new_upload,
    mark_upload_processed,
    sync_profile_dimensions,
)
from app.ui_kit import (  # noqa: E402
    NAV_SECTIONS,
    available_pages,
    demo_notice,
    footer_markdown,
    privacy_notice,
    switch_to,
)

_SIDEBAR_NOTICE_KEY = "_sidebar_notice"
_RESTORE_NONCE_KEY = f"{session_store.PRESERVED_PREFIX}_nonce"
_RESTORE_UPLOADER_KEY = f"{session_store.PRESERVED_PREFIX}_upload"
_RESTORE_SLOT = "session_restore"


def build_navigation() -> Dict[str, List[Any]]:
    sections: Dict[str, List[Any]] = {}
    for info in available_pages():
        page = st.Page(info.file, title=info.title, icon=info.icon,
                       default=info.key == "home")
        sections.setdefault(info.section, []).append(page)
    return {name: sections[name] for name in NAV_SECTIONS if name in sections}


def notify(message: str, level: str = "success") -> None:
    st.session_state[_SIDEBAR_NOTICE_KEY] = (level, message)


def _on_profile_change() -> None:
    selected = st.session_state.get("profile_selector")
    if not selected:
        return
    error = sync_profile_dimensions(st.session_state.config, selected)
    if error:
        notify(f"Could not load profile '{selected}': {error}", "error")


def render_profile_selector() -> None:
    config = st.session_state.config
    profile_loader = ConfigurationLoader(get_config_dir())
    profiles = profile_loader.load_all_profiles()
    if not profiles:
        st.info("⚠️ No profiles found in config/")
        return
    if config.profile_name in profiles:
        st.session_state["profile_selector"] = config.profile_name
    elif st.session_state.get("profile_selector") not in profiles:
        st.session_state["profile_selector"] = profiles[0]
    current_profile = st.selectbox(
        "Configuration profile",
        options=profiles,
        key="profile_selector",
        on_change=_on_profile_change,
        help="Dimension profile used by the classic algorithms and legacy metrics",
    )
    try:
        profile = profile_loader.load_profile(current_profile)
    except Exception as exc:
        st.error(f"❌ Error loading profile: {exc}")
        return
    st.caption(f"✅ {profile.name} · v{profile.version} · "
               f"{len(profile.get_enabled_dimensions())} dimensions")
    if st.button("📁 Open Profile Manager", width="stretch", key="sidebar_profile_manager"):
        switch_to("profiles")


def render_sidebar() -> Any:
    with st.sidebar:
        if not demo.classic_hidden():
            with st.expander("🗂️ Profile", expanded=False):
                render_profile_selector()

        demo_notice()
        privacy_notice()
        render_session_controls()
        st.markdown("---")
        stats_slot = st.empty()
        st.caption(footer_markdown(__version__))
    render_stats(stats_slot)
    return stats_slot


def render_stats(slot: Any) -> None:
    with slot.container():
        loader = get_data_loader()
        stats = loader.get_statistics()
        st.caption(f"Dataset: {get_active_source()}")
        col1, col2 = st.columns(2)
        col1.metric("Students", stats.get("total_students", 0))
        second_metric = (extensions.hook("ui.sidebar.second_metric")
                         if extensions.active() else None)
        if second_metric is not None:
            second_metric(col2, stats)
        else:
            for column, (label, count) in zip((col2, col1), es.distinct_value_counts(
                    es.roster_frame(loader.get_students(), loader.get_dataframe()), limit=2)):
                column.metric(label, count)
        alloc = es.active_allocation()
        projects = alloc.projects if alloc is not None else []
        if projects:
            col1.metric("Teams", sum(len(p.teams) for p in projects if not p.imported))
        history = rh.runs()
        if history:
            col2.metric("Runs", len(history))


def _deferred_snapshot() -> Callable[[], bytes]:
    state: Any = st.session_state
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None:
            state = ctx.session_state
    except Exception:
        pass
    return lambda: session_store.snapshot_json(state)


def apply_session_file(raw: bytes) -> List[str]:
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The file is not valid JSON ({exc}).") from exc
    return session_store.restore_session(st.session_state, payload)


def render_session_controls() -> None:
    st.markdown("**💾 Session**")
    notice = st.session_state.pop(_SIDEBAR_NOTICE_KEY, None)
    if notice:
        level, text = notice
        getattr(st, level, st.info)(text)
    st.download_button(
        "⬇️ Download session",
        data=_deferred_snapshot(),
        file_name=f"group_diversifier_session_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json",
        key="sidebar_download_session",
        on_click="ignore",
        width="stretch",
        help="Save data, settings, rules, preference edits, run history and the active "
             "allocation (unsaved Team Editor edits included) as a file on your computer. "
             "The server keeps no session files: this download is the only saved copy.",
    )
    nonce = int(st.session_state.get(_RESTORE_NONCE_KEY, 0))
    uploaded = st.file_uploader(
        "⬆️ Restore session", type=["json"], key=f"{_RESTORE_UPLOADER_KEY}_{nonce}",
        help="Replace this session with a file from '⬇️ Download session'.")
    if uploaded is None or not is_new_upload(_RESTORE_SLOT, uploaded):
        return
    mark_upload_processed(_RESTORE_SLOT, uploaded)
    try:
        warnings = apply_session_file(uploaded.getvalue())
    except ValueError as exc:
        st.error(f"Could not restore the session: {exc}")
        return
    st.session_state[_RESTORE_NONCE_KEY] = nonce + 1
    runs = len(rh.runs())
    message = f"Session restored from {uploaded.name}" + (f" with {runs} run(s)." if runs else ".")
    if warnings:
        notify(message + " " + " ".join(f"⚠️ {w}" for w in warnings), "warning")
    else:
        notify(message)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Group Diversifier",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_session_state()
    page = st.navigation(build_navigation())
    stats_slot = render_sidebar()
    page.run()
    render_stats(stats_slot)


if __name__ == "__main__":
    main()
