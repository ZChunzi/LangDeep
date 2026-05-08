"""Worker pool for asynchronous task execution."""

import concurrent.futures
import threading
from typing import Any, Callable, Dict, Optional


class WorkerPool:
    """Pool of worker threads for executing scheduled tasks asynchronously.

    Prevents the scheduler tick loop from being blocked by long-running tasks.
    Supports graceful shutdown with timeout.
    """

    def __init__(self, max_workers: int = 4, name_prefix: str = "sched"):
        self._max_workers = max_workers
        self._name_prefix = name_prefix
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

    @property
    def _pool(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix=self._name_prefix,
            )
        return self._executor

    def submit(self, task_id: str, fn: Callable, *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Submit a task for async execution."""
        future = self._pool.submit(fn, *args, **kwargs)
        with self._lock:
            self._futures[task_id] = future

        def _done_cb(f: concurrent.futures.Future) -> None:
            with self._lock:
                self._futures.pop(task_id, None)

        future.add_done_callback(_done_cb)
        return future

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._lock:
            future = self._futures.get(task_id)
            if future is not None and future.running():
                cancelled = future.cancel()
                self._futures.pop(task_id, None)
                return cancelled
        return False

    def running_count(self) -> int:
        with self._lock:
            return len(self._futures)

    def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        """Shut down the pool, optionally waiting for running tasks."""
        self._shutdown_event.set()
        if self._executor is not None:
            if wait:
                with self._lock:
                    futures = list(self._futures.values())
                if futures:
                    concurrent.futures.wait(futures, timeout=timeout)
                self._executor.shutdown(wait=False)
            else:
                self._executor.shutdown(wait=False)
            self._executor = None
        with self._lock:
            self._futures.clear()
