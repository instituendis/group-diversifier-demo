from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import threading
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, Optional

from ..hosting import clamp_workers


def get_safe_context():
    return mp.get_context('spawn')


_executor_cache: Dict[int, concurrent.futures.ProcessPoolExecutor] = {}
_executor_lock = threading.Lock()


def max_workers(requested: int) -> int:
    return clamp_workers(max(1, min(int(requested or 1), os.cpu_count() or 1)))


def _is_broken(executor: concurrent.futures.ProcessPoolExecutor) -> bool:
    return bool(getattr(executor, "_broken", False)) or bool(
        getattr(executor, "_shutdown_thread", False))


def get_process_executor(num_workers: int) -> concurrent.futures.ProcessPoolExecutor:
    workers = max_workers(num_workers)
    with _executor_lock:
        executor = _executor_cache.get(workers)
        if executor is None or _is_broken(executor):
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_safe_context(),
            )
            _executor_cache[workers] = executor
        return executor


def reset_process_executor(num_workers: Optional[int] = None) -> None:
    with _executor_lock:
        keys = list(_executor_cache) if num_workers is None else [max_workers(num_workers)]
        for key in keys:
            executor = _executor_cache.pop(key, None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)


def handle_pool_failure(num_workers: int, error: BaseException) -> None:
    if isinstance(error, BrokenProcessPool):
        reset_process_executor(num_workers)


def estimate_time(n_students: int, n_projects: int, algorithm: str, num_cores: int) -> str:
    scale = max(0.2, n_students / 100)
    if algorithm == "greedy":
        per_project = 3.0 * scale
        spawn = 3.0
    elif algorithm == "simulated_annealing":
        per_project = 4.0 * scale
        spawn = 3.0
    else:
        per_project = 15.0 * scale
        spawn = 4.0
    if num_cores > 1:
        per_project += spawn
    total = per_project * max(1, n_projects)

    if total < 10:
        return f"~{total:.0f}s"
    elif total < 60:
        return f"~{total:.0f} seconds"
    elif total < 300:
        return f"~{total/60:.1f} minutes"
    else:
        return f"~{total/60:.0f} minutes"
