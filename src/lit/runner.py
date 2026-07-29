"""Parallel execution helper.

`find`, `ask` and `claim` all fan out: several discovery agents search different
angles at once, and several reader agents each read a different paper at once.
Each unit of work is an independent `claude -p` subprocess plus network I/O, so
threads are the right tool — the GIL is never the bottleneck.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult:
    index: int
    item: object
    value: object = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_parallel(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    workers: int = 4,
    on_start: Callable[[int, T], None] | None = None,
    on_done: Callable[[TaskResult], None] | None = None,
    stop_on_error: bool = False,
) -> list[TaskResult]:
    """Map `fn` over `items` concurrently, preserving input order in the result.

    A failing item yields a `TaskResult` carrying its exception rather than
    bringing down the batch: one unreachable paper must not abort a 20-paper run.
    """
    if not items:
        return []
    workers = max(1, min(workers, len(items)))
    results: list[TaskResult] = [TaskResult(i, it) for i, it in enumerate(items)]
    lock = threading.Lock()

    if workers == 1:
        for i, item in enumerate(items):
            if on_start:
                on_start(i, item)
            try:
                results[i].value = fn(item)
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                results[i].error = exc
                if stop_on_error:
                    break
            if on_done:
                on_done(results[i])
        return results

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lit") as pool:
        futures = {}
        for i, item in enumerate(items):
            if on_start:
                on_start(i, item)
            futures[pool.submit(fn, item)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i].value = fut.result()
            except BaseException as exc:  # noqa: BLE001
                results[i].error = exc
            if on_done:
                with lock:
                    on_done(results[i])
    return results


def values(results: Iterable[TaskResult]) -> list:
    """Successful, non-None values, in input order."""
    return [r.value for r in results if r.ok and r.value is not None]


def errors(results: Iterable[TaskResult]) -> list[tuple[object, BaseException]]:
    return [(r.item, r.error) for r in results if r.error is not None]
