"""Memory registry — singleton registry for memory backend factories."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..logging import get_logger
from ..errors import ConfigurationError

logger = get_logger(__name__)


class MemoryRegistry:
    """Singleton registry for memory backends.

    Stores factory functions and metadata keyed by backend name.
    Follows the same pattern as AgentRegistry / ModelRegistry.
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
        """Register a memory backend factory.

        Args:
            name: Unique backend name.
            factory: Zero-arg callable returning a BaseMemoryBackend instance.
            metadata: Optional dict with description, config, etc.
        """
        self._factories[name] = factory
        self._metadata[name] = metadata or {}
        if name in self._instances:
            del self._instances[name]
        logger.info(
            "Memory backend registered",
            extra={"backend_name": name, "metadata": metadata},
        )

    def get_backend(self, name: str):
        """Get or create the backend instance by name."""
        if name not in self._factories:
            raise ConfigurationError(
                f"Memory backend '{name}' is not registered",
                context={"available": self.list_backends()},
            )
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
            meta = self._metadata.get(name, {})
            logger.info(
                "Memory backend instance created",
                extra={"backend_name": name, "metadata": meta},
            )
        return self._instances[name]

    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        return list(self._factories.keys())

    def get_metadata(self, name: str) -> Optional[dict]:
        """Get metadata for a registered backend."""
        return self._metadata.get(name)

    def remove(self, name: str) -> None:
        """Remove a registered backend."""
        self._factories.pop(name, None)
        self._metadata.pop(name, None)
        if name in self._instances:
            try:
                self._instances[name].close()
            except Exception:
                pass
            del self._instances[name]

    def clear(self) -> None:
        """Clear all registered backends."""
        for name in list(self._instances.keys()):
            try:
                self._instances[name].close()
            except Exception:
                pass
        self._factories.clear()
        self._metadata.clear()
        self._instances.clear()


# Global singleton
memory_registry = MemoryRegistry()
