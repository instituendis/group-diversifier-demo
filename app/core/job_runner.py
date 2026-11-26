from __future__ import annotations

import math
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterator, Optional

from app import hosting
from app.core.session_state import resolve_state

__all__ = [
    "CancelFn",
    "CollectedJob",
    "JOB_KEY",
    "ProgressFn",
    "ProgressView",
    "STOP_BUTTON_KEY",
    "STOP_HELP",
    "STOP_KEY",
    "STOP_LABEL",
    "SlotWaitCancelled",
    "WAITING_TEXT",
    "active_runs",
    "waiting_runs",
    "collect_stopped_run",
    "request_stop",
    "run_in_background",
    "run_slot",
    "streamlit_view",
]

JOB_KEY = "_background_job"
STOP_KEY = "_background_stop_requested"
STOP_BUTTON_KEY = "_background_stop_button"

STOP_LABEL = "⏹ Stop and keep best"
STOP_HELP = ("Stop the search now and keep the best allocation found so far "
             "(recorded as a run marked 'stopped early').")

SPIN_TEXT = "Stopping the search and keeping the best allocation found so far…"
ERROR_PREFIX = "❌ The rules engine could not form teams: "
EXPIRED_TEXT = ("The search did not stop in time, so no allocation was kept. Run Team "
                "Formation again.")

POLL_SECONDS = 0.25

ProgressFn = Callable[[float, float, str], None]
CancelFn = Callable[[], bool]

WAITING_TEXT = "Waiting for a free slot: another allocation is running on this server"
WAIT_CANCELLED_TEXT = "Stopped while waiting for a free slot; nothing was run."


class SlotWaitCancelled(RuntimeError):
    def __init__(self, message: str = WAIT_CANCELLED_TEXT):
        super().__init__(message)


_slots = threading.Condition()
_active = 0
_queue: Deque[object] = deque()


def active_runs() -> int:
    with _slots:
        return _active


def waiting_runs() -> int:
    with _slots:
        return len(_queue)


def _cap(limit: Optional[int]) -> Optional[int]:
    return limit if limit is not None else hosting.max_concurrent_runs()


def _release() -> None:
    global _active
    with _slots:
        _active = max(0, _active - 1)
        _slots.notify_all()


def _acquire(cancel: Optional[CancelFn], on_wait: Optional[Callable[[str], None]],
             limit: Optional[int], poll: float) -> None:
    global _active
    ticket = object()
    with _slots:
        cap = _cap(limit)
        if not _queue and (cap is None or _active < cap):
            _active += 1
            return
        _queue.append(ticket)
    try:
        while True:
            if cancel is not None and cancel():
                raise SlotWaitCancelled()
            with _slots:
                cap = _cap(limit)
                if _queue[0] is ticket and (cap is None or _active < cap):
                    _queue.popleft()
                    _active += 1
                    _slots.notify_all()
                    return
            if on_wait is not None:
                on_wait(WAITING_TEXT)
            with _slots:
                cap = _cap(limit)
                if not (_queue[0] is ticket and (cap is None or _active < cap)):
                    _slots.wait(timeout=poll)
    except BaseException:
        with _slots:
            try:
                _queue.remove(ticket)
            except ValueError:  # pragma: no cover - already granted and removed
                pass
            _slots.notify_all()
        raise


@contextmanager
def run_slot(*, cancel: Optional[CancelFn] = None,
             on_wait: Optional[Callable[[str], None]] = None,
             limit: Optional[int] = None, poll: float = POLL_SECONDS) -> Iterator[None]:
    _acquire(cancel, on_wait, limit, poll)
    try:
        yield
    finally:
        _release()


def _session() -> Any:
    return resolve_state()


def request_stop(state: Any = None, *, stop_key: str = STOP_KEY) -> None:
    state = _session() if state is None else state
    state[stop_key] = True


def _attach_script_run_ctx(thread: threading.Thread) -> None:
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

        add_script_run_ctx(thread, get_script_run_ctx())
    except Exception:  # pragma: no cover - bare mode (tests, scripts)
        pass


class ProgressView:
    def start(self, title: str) -> None: ...

    def progress(self, fraction: float, text: str) -> None: ...

    def detail(self, text: str) -> None: ...

    def stop_button(self, label: str, key: str, help: str) -> None: ...

    def done(self, text: str) -> None: ...


def streamlit_view(title: str, *, stop_key: str = STOP_KEY) -> ProgressView:
    import streamlit as st

    class _StreamlitView(ProgressView):
        def __init__(self) -> None:
            self._bar = st.progress(0.0, text=f"{title}: starting…")
            self._detail = st.empty()

        def progress(self, fraction: float, text: str) -> None:
            self._bar.progress(fraction, text=text)

        def detail(self, text: str) -> None:
            self._detail.markdown(text)

        def stop_button(self, label: str, key: str, help: str) -> None:
            st.button(label, key=key, on_click=request_stop,
                      kwargs={"stop_key": stop_key}, help=help)

        def done(self, text: str) -> None:
            self._bar.progress(1.0, text=text)

    return _StreamlitView()


def run_in_background(work: Callable[[ProgressFn, CancelFn], Any], *, title: str,
                      stop_job: Optional[Dict[str, Any]] = None,
                      state: Any = None,
                      job_key: str = JOB_KEY, stop_key: str = STOP_KEY,
                      stop_button_key: str = STOP_BUTTON_KEY,
                      stop_label: str = STOP_LABEL, stop_help: str = STOP_HELP,
                      view: Optional[ProgressView] = None,
                      poll: float = POLL_SECONDS, slot: bool = True) -> Any:
    state = _session() if state is None else state
    if view is None:
        view = streamlit_view(title, stop_key=stop_key)
    view.start(title)
    live: Dict[str, Any] = {"fraction": 0.0, "best": None, "message": "starting"}
    holder: Dict[str, Any] = {}
    cancelled = threading.Event()

    def progress(fraction: float, best: float, message: str) -> None:
        live.update(fraction=float(fraction), best=best, message=str(message))

    def waiting(text: str) -> None:
        live.update(message=text)

    def target() -> None:
        try:
            if slot:
                with run_slot(cancel=cancelled.is_set, on_wait=waiting, poll=poll):
                    live.update(message="starting")
                    holder["result"] = work(progress, cancelled.is_set)
            else:
                holder["result"] = work(progress, cancelled.is_set)
        except BaseException as exc:
            holder["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name="formation-worker")
    _attach_script_run_ctx(thread)
    if stop_job is not None:
        state.pop(stop_key, None)
        state[job_key] = {"thread": thread, "cancel": cancelled,
                          "holder": holder, "inputs": stop_job}
        view.stop_button(stop_label, stop_button_key, stop_help)
    started = time.time()
    thread.start()
    try:
        while thread.is_alive():
            fraction = min(max(float(live["fraction"]), 0.0), 0.99)
            view.progress(fraction, f"{title}: {live['message']}")
            best = live["best"]
            best_text = (f" · best objective **{float(best):.2f}**"
                         if isinstance(best, (int, float)) and math.isfinite(float(best)) else "")
            view.detail(f"⏱️ {time.time() - started:.1f}s{best_text}")
            thread.join(timeout=poll)
    finally:
        if thread.is_alive():
            cancelled.set()
    if stop_job is not None:
        state.pop(job_key, None)
    view.done(f"{title}: done in {time.time() - started:.1f}s")
    if "error" in holder:
        raise holder["error"]
    return holder.get("result")


@dataclass
class CollectedJob:
    outcome: str
    result: Any = None
    inputs: Any = None
    error: Optional[BaseException] = None


def collect_stopped_run(finish: Callable[[Any, Dict[str, Any]], None], *,
                        state: Any = None, job_key: str = JOB_KEY,
                        stop_key: str = STOP_KEY,
                        wait_seconds: Optional[Callable[[Dict[str, Any]], float]] = None,
                        poll: float = POLL_SECONDS) -> None:
    import streamlit as st

    state = _session() if state is None else state
    job = state.pop(job_key, None)
    requested = state.pop(stop_key, False)
    if job is None:
        return
    job["cancel"].set()
    if not requested:
        return
    budget = float(wait_seconds(job["inputs"])) if wait_seconds is not None else 0.0
    deadline = time.time() + budget
    with st.spinner(SPIN_TEXT):
        while job["thread"].is_alive() and time.time() < deadline:
            job["thread"].join(timeout=poll)
    holder = job["holder"]
    if isinstance(holder.get("error"), SlotWaitCancelled):
        st.info(str(holder["error"]))
    elif "error" in holder:
        st.error(f"{ERROR_PREFIX}{holder['error']}")
    elif "result" not in holder:
        st.error(EXPIRED_TEXT)
    else:
        finish(holder["result"][0], job["inputs"])
