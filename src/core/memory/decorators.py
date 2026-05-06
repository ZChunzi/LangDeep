"""@memory decorator — registers a memory backend factory."""

from typing import Any, Callable, Optional

from ..logging import get_logger
from .registry import memory_registry
from .builtin import InMemoryBackend

logger = get_logger(__name__)


def memory(
    name: Optional[str] = None,
    description: str = "",
    **kwargs: Any,
):
    """Decorator that registers a memory backend factory.

    Usage::

        @memory(name="session_store", description="Redis conversation store")
        def session_store():
            return RedisBackend(host="localhost", port=6379)

    If the decorated function returns None or is empty (pass),
    the built-in ``InMemoryBackend`` is used automatically.

    The registered backend can be retrieved at runtime::

        from langdeep.core.memory import memory_registry
        backend = memory_registry.get_backend("session_store")
    """

    def decorator(func: Callable) -> Callable:
        backend_name = name or func.__name__

        def factory() -> Any:
            result = func()
            if result is None:
                logger.info(
                    "Using InMemoryBackend for '%s' (factory returned None)",
                    backend_name,
                )
                return InMemoryBackend()
            return result

        metadata = {
            "description": description or func.__doc__ or "",
            "kwargs": kwargs,
        }
        memory_registry.register(backend_name, factory, metadata)
        return func

    return decorator
