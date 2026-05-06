"""@cache decorator — registers a cache backend factory."""

from typing import Any, Callable, Optional

from ..logging import get_logger
from .registry import cache_registry
from .builtin import MemoryCache

logger = get_logger(__name__)


def cache(
    name: Optional[str] = None,
    ttl: Optional[int] = 300,
    max_entries: int = 1024,
    description: str = "",
    **kwargs: Any,
):
    """Decorator that registers a cache backend factory.

    Usage::

        @cache(name="llm_cache", ttl=300, max_entries=1024)
        def llm_cache():
            pass  # uses built-in MemoryCache with LRU+TTL

    For custom backends, return a ``BaseCacheBackend`` instance::

        @cache(name="redis_cache", ttl=3600)
        def redis_cache():
            return RedisCache(host="localhost", port=6379)
    """

    def decorator(func: Callable) -> Callable:
        cache_name = name or func.__name__

        def factory() -> Any:
            result = func()
            if result is None:
                logger.info(
                    "Using MemoryCache for '%s' (factory returned None)",
                    cache_name,
                )
                return MemoryCache(max_size=max_entries, default_ttl=ttl)
            return result

        metadata = {
            "description": description or func.__doc__ or "",
            "ttl": ttl,
            "max_entries": max_entries,
            "kwargs": kwargs,
        }
        cache_registry.register(cache_name, factory, metadata)
        return func

    return decorator
