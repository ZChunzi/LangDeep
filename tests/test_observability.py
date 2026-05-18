"""Unit tests for the observability module: HealthStatus, HealthChecker, MetricsCollector."""

from datetime import datetime
from contextlib import contextmanager

from langchain_core.tools import tool as lc_tool

from langdeep.core.observability import (
    HealthStatus,
    HealthChecker,
    MetricsCollector,
    NoOpTracingAdapter,
    OpenTelemetryTracingAdapter,
)
from langdeep.core.tools import wrap_tool


def setup_function():
    """Reset singleton registries before each test."""
    from tests.conftest import clean_registries
    clean_registries()


# ── HealthStatus ────────────────────────────────────────────────────────────


def test_health_status_defaults():
    """HealthStatus has sensible defaults."""
    hs = HealthStatus()
    assert hs.status == "healthy"
    assert hs.checks == {}
    assert hs.version == ""
    assert isinstance(hs.timestamp, datetime)


def test_health_status_custom_values():
    """HealthStatus accepts custom values."""
    hs = HealthStatus(
        status="degraded",
        checks={"db": {"status": "ok"}},
        version="1.0.0",
    )
    assert hs.status == "degraded"
    assert hs.checks == {"db": {"status": "ok"}}
    assert hs.version == "1.0.0"


# ── HealthChecker ───────────────────────────────────────────────────────────


def test_health_checker_all_returns_status():
    """check_all returns a HealthStatus object."""
    checker = HealthChecker(version="test")
    result = checker.check_all()
    assert isinstance(result, HealthStatus)
    assert result.status in ("healthy", "degraded", "unhealthy")
    assert result.version == "test"


def test_health_checker_checks_memory_and_cache_registries():
    """check_all probes registered memory/cache backends instead of skipping them."""
    from langdeep.core.memory.registry import memory_registry
    from langdeep.core.cache.registry import cache_registry

    memory_registry.register("mem", lambda: object())
    cache_registry.register("cache", lambda: object())

    result = HealthChecker().check_all()

    assert result.checks["memory"]["mem"]["status"] == "ok"
    assert result.checks["cache"]["cache"]["status"] == "ok"


def test_health_checker_reports_backend_and_registry_errors():
    """check_all exposes backend factory errors and registry diagnostic errors."""
    from langdeep.core.cache.registry import cache_registry
    from langdeep.core.memory.registry import memory_registry
    from langdeep.core.registry.agent_registry import AgentMetadata, agent_registry

    memory_registry.register("bad_mem", lambda: (_ for _ in ()).throw(RuntimeError("mem down")))
    cache_registry.register("bad_cache", lambda: (_ for _ in ()).throw(RuntimeError("cache down")))
    agent_registry.register(
        "bad_agent",
        lambda: object(),
        AgentMetadata(name="bad_agent", description="", model_name="missing_model"),
    )

    result = HealthChecker().check_all()

    assert result.status == "unhealthy"
    assert result.checks["memory"]["bad_mem"]["status"] == "error"
    assert result.checks["cache"]["bad_cache"]["status"] == "error"
    assert result.checks["agents"]["status"] == "error"


def test_health_checker_aggregate_healthy():
    """All-ok checks produce 'healthy' status."""
    checker = HealthChecker()
    checks = {"db": {"status": "ok"}, "cache": {"status": "ok"}}
    assert checker._aggregate(checks) == "healthy"


def test_health_checker_aggregate_unhealthy():
    """Any error produces 'unhealthy' status."""
    checker = HealthChecker()
    checks = {"db": {"status": "ok"}, "model": {"status": "error", "detail": "timeout"}}
    assert checker._aggregate(checks) == "unhealthy"


def test_health_checker_aggregate_degraded():
    """Degraded-only produces 'degraded' status."""
    checker = HealthChecker()
    checks = {"db": {"status": "degraded", "detail": "slow"}}
    assert checker._aggregate(checks) == "degraded"


def test_health_checker_empty_checks_healthy():
    """Empty checks dict produces 'healthy'."""
    checker = HealthChecker()
    assert checker._aggregate({}) == "healthy"


# ── MetricsCollector ────────────────────────────────────────────────────────


def test_metrics_counter():
    """Counter increments correctly."""
    mc = MetricsCollector()
    mc.counter("requests")
    mc.counter("requests")
    mc.counter("errors", 3)
    metrics = mc.get_metrics()
    assert metrics["counters"]["requests"] == 2
    assert metrics["counters"]["errors"] == 3


def test_metrics_tagged_names_are_deterministic():
    """Tagged metrics are normalized into deterministic metric keys."""
    mc = MetricsCollector()
    mc.counter("requests", tags={"status": "ok", "endpoint": "/chat"})
    mc.histogram("latency", 12, tags={"status": "ok"})
    mc.gauge("active", 1, tags={"worker": "a|b"})

    metrics = mc.get_metrics()
    assert metrics["counters"]["requests|endpoint=/chat,status=ok"] == 1
    assert metrics["histograms"]["latency|status=ok"]["count"] == 1
    assert metrics["gauges"]["active|worker=a_b"] == 1


def test_metrics_gauge():
    """Gauge sets and overwrites value."""
    mc = MetricsCollector()
    mc.gauge("connections", 10)
    mc.gauge("connections", 5)
    assert mc.get_metrics()["gauges"]["connections"] == 5


def test_metrics_histogram():
    """Histogram records values with summary statistics."""
    mc = MetricsCollector()
    for v in range(1, 101):
        mc.histogram("latency", float(v))
    hist = mc.get_metrics()["histograms"]["latency"]
    assert hist["count"] == 100
    assert hist["min"] == 1.0
    assert hist["max"] == 100.0
    assert hist["avg"] == 50.5
    assert hist["p50"] == 50.0
    assert hist["p90"] == 90.0


def test_metrics_get_metrics_structure():
    """get_metrics returns a dict with counters, histograms, gauges."""
    mc = MetricsCollector()
    mc.counter("c1")
    mc.gauge("g1", 1)
    mc.histogram("h1", 1)
    metrics = mc.get_metrics()
    assert "counters" in metrics
    assert "histograms" in metrics
    assert "gauges" in metrics


def test_metrics_clear():
    """Clear resets all metrics."""
    mc = MetricsCollector()
    mc.counter("c1", 10)
    mc.clear()
    assert mc.get_metrics()["counters"] == {}
    assert mc.get_metrics()["gauges"] == {}
    assert mc.get_metrics()["histograms"] == {}


# ── Tracing adapters ────────────────────────────────────────────────────────


def test_noop_tracing_adapter_context_manager():
    adapter = NoOpTracingAdapter()

    with adapter.start_span("test") as span:
        span.set_attribute("key", "value")
        span.record_exception(RuntimeError("ignored"))


def test_opentelemetry_adapter_falls_back_without_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "opentelemetry":
            raise ImportError("missing otel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    adapter = OpenTelemetryTracingAdapter()

    with adapter.start_span("test") as span:
        span.set_attribute("ok", True)


def test_policy_wrapped_tool_records_trace_span():
    tracing = RecordingTracingAdapter()

    @lc_tool
    def echo_tool(text: str) -> str:
        """Echo text."""
        return text

    wrapped = wrap_tool(echo_tool, tracing_adapter=tracing)

    assert wrapped.invoke({"text": "hello"}) == "hello"
    assert tracing.names() == ["langdeep.tool"]
    assert tracing.spans[0]["attributes"]["langdeep.tool"] == "echo_tool"
    assert tracing.spans[0]["attributes"]["langdeep.status"] == "success"


class RecordingTracingAdapter:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_span(self, name, attributes=None):
        span = RecordingSpan(name, dict(attributes or {}))
        self.spans.append(span.data)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            span.data["ended"] = True

    def names(self):
        return [span["name"] for span in self.spans]


class RecordingSpan:
    def __init__(self, name, attributes):
        self.data = {"name": name, "attributes": attributes, "exceptions": [], "ended": False}

    def set_attribute(self, key, value):
        self.data["attributes"][key] = value

    def record_exception(self, exc):
        self.data["exceptions"].append(type(exc).__name__)
