from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from app import demo, extensions, hosting

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"

_CHIP_ESCAPES = str.maketrans({"[": "(", "]": ")", "*": "\\*", "_": "\\_", "`": "'",
                               "$": "\\$", "<": "‹", ">": "›", "#": "\\#", "|": "/"})

_FLASH_KEY = "_flash_messages"


def chip(label: Any, color: Optional[str] = None) -> str:
    return f":{color or 'gray'}-badge[{str(label).translate(_CHIP_ESCAPES)}]"


@dataclass(frozen=True)
class PageInfo:
    key: str
    file: str
    title: str
    icon: str
    section: str

    @property
    def path(self) -> Path:
        return APP_DIR / self.file

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def available(self) -> bool:
        return self.exists and not (self.key in HOSTED_HIDDEN_PAGES and demo.classic_hidden())


NAV_SECTIONS: Tuple[str, ...] = ("Start", "Setup", "Allocate", "Review", "Share", "Advanced")

PAGE_REGISTRY: Tuple[PageInfo, ...] = (
    PageInfo("home", "pages/0_🏠_Home.py", "Home", "🏠", "Start"),
    PageInfo("data", "pages/1_📊_Data_Overview.py", "Data Overview", "📊", "Setup"),
    PageInfo("configuration", "pages/2_⚙️_Configuration.py", "Configuration", "⚙️", "Setup"),
    PageInfo("rules", "pages/10_🧭_Allocation_Rules.py", "Allocation Rules", "🧭", "Setup"),
    PageInfo("formation", "pages/3_🎯_Team_Formation.py", "Team Formation", "🎯", "Allocate"),
    PageInfo("editor", "pages/11_✏️_Team_Editor.py", "Team Editor", "✏️", "Allocate"),
    PageInfo("analytics", "pages/4_📈_Analytics.py", "Analytics", "📈", "Review"),
    PageInfo("cross_project", "pages/5_🔄_Cross_Project.py", "Cross-Project", "🔄", "Review"),
    PageInfo("compare", "pages/12_🆚_Compare_Runs.py", "Compare Runs", "🆚", "Review"),
    PageInfo("export", "pages/6_📤_Export.py", "Export", "📤", "Share"),
    PageInfo("profiles", "pages/8_📁_Profile_Manager.py", "Profile Manager", "📁", "Advanced"),
)


def page_info(key: str) -> PageInfo:
    for info in PAGE_REGISTRY:
        if info.key == key:
            return info
    raise KeyError(f"Unknown page {key!r}")


HOSTED_HIDDEN_PAGES = frozenset({"profiles"})


def page_exists(key: str) -> bool:
    return page_info(key).available


def available_pages() -> List[PageInfo]:
    return [info for info in PAGE_REGISTRY if info.available]


def _page_registered(info: PageInfo) -> bool:
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is None:
            return False
        target = os.path.realpath(info.path)
        return any(
            os.path.realpath(str(page.get("script_path") or "")) == target
            for page in ctx.pages_manager.get_pages().values()
        )
    except Exception:
        return False


def page_link(key: str, label: Optional[str] = None, *, icon: Optional[str] = None,
              help: Optional[str] = None, width: str = "content",
              button_key: Optional[str] = None) -> bool:
    info = page_info(key)
    if not info.available:
        return False
    text = label or info.title
    shown_icon = icon if icon is not None else info.icon
    if _page_registered(info):
        st.page_link(info.file, label=text, icon=shown_icon, help=help, width=width)
    elif button_key is not None:
        if st.button(text, key=button_key, icon=shown_icon, help=help):
            st.switch_page(info.file)
    else:
        st.markdown(f"{shown_icon} **{text}** — open it from the navigation menu")
    return True


def switch_to(key: str) -> None:
    info = page_info(key)
    if info.available:
        st.switch_page(info.file)


def needs_router_handover(uses_pages_directory: Any, page_script: Optional[str],
                          main_script: Optional[str]) -> bool:
    if uses_pages_directory is not True or not page_script or not main_script:
        return False
    return os.path.realpath(page_script) != os.path.realpath(main_script)


def _real_server() -> bool:
    from streamlit import runtime
    from streamlit.runtime.runtime import Runtime

    return runtime.exists() and type(Runtime.instance()) is Runtime


def _router_handover_state() -> Optional[Tuple[Any, Optional[str], str]]:
    try:
        from streamlit.runtime.pages_manager import PagesManager
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        if not _real_server():
            return None
        ctx = get_script_run_ctx()
        if ctx is None or getattr(ctx, "script_requests", None) is None:
            return None
        manager = ctx.pages_manager
        page = manager.get_pages().get(ctx.page_script_hash) or {}
        return (getattr(PagesManager, "uses_pages_directory", None),
                page.get("script_path"), str(manager.main_script_path))
    except Exception:
        return None


def ensure_router() -> None:
    state = _router_handover_state()
    if state is None or not needs_router_handover(*state):
        return
    try:
        from streamlit.runtime.pages_manager import PagesManager

        PagesManager.uses_pages_directory = False
        if PagesManager.uses_pages_directory is not False:
            return
    except Exception:
        return
    st.rerun()


def require_results() -> None:
    from app.core.engine_session import has_active_allocation

    if has_active_allocation():
        return
    st.warning("No team formation results available. Please run team formation first.")
    if st.button("Go to Team Formation"):
        switch_to("formation")
    st.stop()


def departed_warning(projects: Any, roster_ids: Any, dataset_note: str = "") -> bool:
    from app.core import engine_session as es

    notice = es.roster_notice(projects or [], roster_ids)
    if not notice:
        return False
    kind, text = notice
    if kind == "dataset":
        st.warning(f"⚠️ {text}" + (f" {dataset_note}" if dataset_note else ""))
        page_link("data", "Open Data Overview")
        return True
    st.warning(f"⚠️ {text}")
    page_link("editor", "Open the Team Editor (🩹 Roster changes tab)")
    return True


SAMPLES_DIR = extensions.hook("samples_dir", demo.DEMO_DIR)

SAMPLE_FILES: Dict[str, Tuple[str, str]] = extensions.hook("sample_files", demo.DEMO_SAMPLE_FILES)


def sample_file(kind: str) -> Optional[Tuple[Path, str]]:
    if demo.demo_mode():
        files, folder = demo.DEMO_SAMPLE_FILES, demo.DEMO_DIR
    else:
        files, folder = SAMPLE_FILES, SAMPLES_DIR
    if kind not in files:
        return None
    file_name, description = files[kind]
    path = Path(folder) / file_name
    return (path, description) if path.is_file() else None


def sample_download_button(kind: str, *, label: str = "⬇️ Download sample") -> None:
    found = sample_file(kind)
    if found is None:
        return
    path, description = found
    file_name = path.name
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=file_name,
        mime="text/csv",
        key=f"sample_download_{kind}",
        on_click="ignore",
        help=f"{file_name}: {description}. The sample files belong together: they "
             "use the ids of the sample cohort.",
    )


def privacy_notice(*, expanded: bool = False) -> bool:
    if not hosting.hosted():
        return False
    with st.expander("🔒 Privacy notice", expanded=expanded):
        st.markdown(hosting.privacy_notice_markdown())
    return True


def demo_notice() -> bool:
    if not demo.demo_mode():
        return False
    st.caption(demo.BANNER_TEXT)
    return True


def refuse_large_table(uploaded: Any) -> bool:
    size = getattr(uploaded, "size", None)
    if size is None:
        size = len(uploaded.getvalue())
    problem = hosting.table_upload_problem(size)
    if problem is None:
        return False
    st.error(f"❌ {getattr(uploaded, 'name', 'The file')}: {problem}")
    return True


def footer_markdown(version: str) -> str:
    return (f"Group Diversifier v{version} · AGPL-3.0 · "
            f"[Source code]({demo.source_url()})")


def queue_flash(state: Any, message: str, level: str = "success") -> None:
    queue = list(state.get(_FLASH_KEY, []))
    queue.append((level, message))
    state[_FLASH_KEY] = queue


def flash(message: str, level: str = "success") -> None:
    queue_flash(st.session_state, message, level)


def render_flashes() -> None:
    for level, message in st.session_state.pop(_FLASH_KEY, []):
        renderer = getattr(st, level, st.info)
        renderer(message)
