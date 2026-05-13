"""Lightweight in-process metrics collector."""

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Thread-safe in-process metrics collector with counter, histogram, and gauge support.

    Usage::

        mc = MetricsCollector()
        mc.counter("requests.total")
        mc.histogram("latency_ms", 42.5)
        mc.gauge("connections.active", 10)
        mc.get_metrics()  # → {"counters": ..., "histograms": ..., "gauges": ...}

    Tags can be appended using the format ``name|key=value``::

        mc.counter("requests.total|endpoint=/chat")
    """

    def __init__(self) -> None:
        self._counters: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, value: float = 1, tags: Optional[Dict[str, Any]] = None) -> None:
        """Increment a counter by *value* (default 1)."""
        name = _metric_name(name, tags)
        with self._lock:
            self._counters[name] += value

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        """Record a value for percentile calculation."""
        name = _metric_name(name, tags)
        with self._lock:
            self._histograms[name].append(value)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        """Set a gauge to *value*."""
        name = _metric_name(name, tags)
        with self._lock:
            self._gauges[name] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of all metrics.

        Histograms include ``p50``, ``p90``, and ``p99`` percentiles
        in addition to ``count``, ``min``, ``max``, and ``avg``.
        """
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {}
            for name, values in self._histograms.items():
                histograms[name] = self._summarize_histogram(values)

        return {
            "counters": counters,
            "histograms": histograms,
            "gauges": gauges,
        }

    def clear(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _summarize_histogram(values: List[float]) -> Dict[str, float]:
        """Compute summary statistics for a list of values."""
        if not values:
            return {}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        return {
            "count": n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "avg": total / n,
            "p50": sorted_vals[int((n - 1) * 0.50)],
            "p90": sorted_vals[int((n - 1) * 0.90)],
            "p99": sorted_vals[int((n - 1) * 0.99)],
        }


def _metric_name(name: str, tags: Optional[Dict[str, Any]] = None) -> str:
    if not tags:
        return name
    normalized = {
        key: str(value).replace(",", "_").replace("|", "_")
        for key, value in tags.items()
        if value is not None
    }
    if not normalized:
        return name
    tag_text = ",".join(f"{key}={normalized[key]}" for key in sorted(normalized))
    return f"{name}|{tag_text}"
