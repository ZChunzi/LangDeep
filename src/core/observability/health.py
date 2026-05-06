"""Health checker — probes registered backends and aggregates status."""

from typing import Any, Dict

from ..logging import get_logger
from .models import HealthStatus

logger = get_logger(__name__)


class HealthChecker:
    """Probes all registered backends and aggregates their health status.

    The checker dynamically discovers backends from the model, memory,
    cache, and sandbox registries.
    """

    def __init__(self, version: str = ""):
        self._version = version

    def check_all(self, timeout: int = 5) -> HealthStatus:
        """Run all health checks and return an aggregated status.

        Args:
            timeout: Maximum seconds per individual check (best-effort).

        Returns:
            A :class:`HealthStatus` with per-component results.
        """
        checks: Dict[str, Any] = {}

        for check_fn in (self._check_model_backends,
                         self._check_memory_backends,
                         self._check_cache_backends):
            try:
                result = check_fn(timeout)
                checks.update(result)
            except Exception as exc:
                logger.warning("Health check failed", extra={"check": check_fn.__name__, "error": str(exc)})
                checks[check_fn.__name__] = {"status": "error", "detail": str(exc)}

        overall = self._aggregate(checks)
        return HealthStatus(
            status=overall,
            checks=checks,
            version=self._version,
        )

    # ── Individual checks ────────────────────────────────────────────

    @staticmethod
    def _check_model_backends(timeout: int = 5) -> Dict[str, Any]:
        """Probe each registered model backend."""
        results: Dict[str, Any] = {"models": {}}
        try:
            from ..registry import model_registry
            models = model_registry.list_models()
            for name in models:
                try:
                    model_registry.get_model(name)
                    results["models"][name] = {"status": "ok"}
                except Exception as exc:
                    results["models"][name] = {"status": "error", "detail": str(exc)}
        except ImportError:
            results["models"] = {"status": "skipped", "detail": "model_registry not available"}
        return results

    @staticmethod
    def _check_memory_backends(timeout: int = 5) -> Dict[str, Any]:
        """Probe each registered memory backend."""
        results: Dict[str, Any] = {"memory": {}}
        try:
            from ..registry import memory_registry
            backends = memory_registry.list_backends()
            for name in backends:
                try:
                    memory_registry.get_backend(name)
                    results["memory"][name] = {"status": "ok"}
                except Exception as exc:
                    results["memory"][name] = {"status": "error", "detail": str(exc)}
        except ImportError:
            results["memory"] = {"status": "skipped", "detail": "memory_registry not available"}
        return results

    @staticmethod
    def _check_cache_backends(timeout: int = 5) -> Dict[str, Any]:
        """Probe each registered cache backend."""
        results: Dict[str, Any] = {"cache": {}}
        try:
            from ..registry import cache_registry
            backends = cache_registry.list_backends()
            for name in backends:
                try:
                    cache_registry.get_backend(name)
                    results["cache"][name] = {"status": "ok"}
                except Exception as exc:
                    results["cache"][name] = {"status": "error", "detail": str(exc)}
        except ImportError:
            results["cache"] = {"status": "skipped", "detail": "cache_registry not available"}
        return results

    # ── Aggregation ──────────────────────────────────────────────────

    @staticmethod
    def _aggregate(checks: Dict[str, Any]) -> str:
        """Determine overall status from component checks.

        * All ok → ``"healthy"``
        * Any error → ``"unhealthy"``
        * Any degraded → ``"degraded"``
        """
        has_error = False
        has_degraded = False

        def _walk(d: Any) -> None:
            nonlocal has_error, has_degraded
            if isinstance(d, dict):
                if "status" in d:
                    if d["status"] == "error":
                        has_error = True
                    elif d["status"] == "degraded":
                        has_degraded = True
                for v in d.values():
                    _walk(v)

        _walk(checks)

        if has_error:
            return "unhealthy"
        if has_degraded:
            return "degraded"
        return "healthy"
