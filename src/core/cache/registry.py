"""Cache registry — singleton registry for cache backend factories."""

from typing import Any, Callable, Dict, List, Optional

from ..logging import get_logger
from ..errors import ConfigurationError

logger = get_logger(__name__)


class CacheRegistry:
    """Singleton registry for cache backends.

    Follows the same pattern as MemoryRegistry / AgentRegistry.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factories: Dict[str, Callable[[], Any]] = {}
            cls._instance._metadata: Dict[str, dict] = {}
            cls._instance._instances: Dict[str, Any] = {}
        return cls._instance

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        metadata: Optional[dict] = None,
    ) -> None:
        """Register a cache backend factory."""
        self._factories[name] = factory
        self._metadata[name] = metadata or {}
        if name in self._instances:
            del self._instances[name]
        logger.info(
            "Cache backend registered",
            extra={"cache_name": name, "metadata": metadata},
        )

    def get_backend(self, name: str):
        """Get or create the cache backend instance by name."""
        if name not in self._factories:
            raise ConfigurationError(
                f"Cache backend '{name}' is not registered",
                context={"available": self.list_backends()},
            )
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
            meta = self._metadata.get(name, {})
            logger.info(
                "Cache backend instance created",
                extra={"cache_name": name, "metadata": meta},
            )
        return self._instances[name]

    def list_backends(self) -> List[str]:
        return list(self._factories.keys())

    def get_metadata(self, name: str) -> Optional[dict]:
        return self._metadata.get(name)

    def remove(self, name: str) -> None:
        self._factories.pop(name, None)
        self._metadata.pop(name, None)
        self._instances.pop(name, None)

    def clear(self) -> None:
        self._factories.clear()
        self._metadata.clear()
        self._instances.clear()


# Global singleton
cache_registry = CacheRegistry()
