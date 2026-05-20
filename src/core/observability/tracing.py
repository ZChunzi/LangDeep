"""Optional tracing adapters for LangDeep runtime spans."""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional


class NoOpSpan:
    """Span object used when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass


class NoOpTracingAdapter:
    """Tracing adapter that records nothing."""

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[NoOpSpan]:
        yield NoOpSpan()


class OpenTelemetryTracingAdapter:
    """OpenTelemetry-backed tracing adapter with no-dependency fallback.

    If ``opentelemetry-api`` is not installed, spans become no-ops so LangDeep's
    default installation remains dependency-free.
    """

    def __init__(self, tracer: Optional[Any] = None, instrumentation_scope: str = "langdeep"):
        self._noop = NoOpTracingAdapter()
        if tracer is not None:
            self._tracer = tracer
            return
        try:
            from opentelemetry import trace
        except ImportError:
            self._tracer = None
            return
        self._tracer = trace.get_tracer(instrumentation_scope)

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Any]:
        if self._tracer is None:
            with self._noop.start_span(name, attributes) as span:
                yield span
            return
        with self._tracer.start_as_current_span(
            name,
            attributes=dict(attributes or {}),
        ) as span:
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                raise


def span_attributes(**values: Any) -> Dict[str, Any]:
    """Return tracing attributes without ``None`` values."""
    return {key: value for key, value in values.items() if value is not None}
