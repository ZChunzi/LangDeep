"""SandboxRegistry singleton — holds named sandbox backend factories."""
import threading
from typing import Any, Callable, Dict, List, Optional

from ..errors import ConfigurationError
from ..logging import get_logger
from .base import BaseSandbox
from .builtin import SubprocessSandbox

logger = get_logger(__name__)

#: Factory signature: ``(**kwargs) -> BaseSandbox``
SandboxFactory = Callable[..., BaseSandbox]


class SandboxRegistry:
    """Singleton registry for sandbox backends.

    Backends are registered with a name and factory function.
    The built-in SubprocessSandbox is registered automatically as
    ``"subprocess"`` on first access.
    """

    _instance = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._backends: Dict[str, BaseSandbox] = {}
                    cls._instance._metadata: Dict[str, Dict[str, Any]] = {}
                    cls._instance._factory: Dict[str, SandboxFactory] = {}
                    cls._instance._registry_lock = threading.Lock()
                    cls._instance._register_builtin()
        return cls._instance

    def _register_builtin(self) -> None:
        """Pre-register the default SubprocessSandbox backend."""
        from .builtin import SubprocessSandbox
        self._backends["subprocess"] = SubprocessSandbox()
        self._metadata["subprocess"] = {
            "name": "subprocess",
            "description": "Isolated subprocess sandbox with resource limits",
            "builtin": True,
        }

    def register(
        self,
        name: str,
        factory: SandboxFactory,
        description: str = "",
        **metadata: Any,
    ) -> None:
        """Register a sandbox backend.

        Args:
            name: Unique backend name.
            factory: Callable returning a ``BaseSandbox`` instance.
            description: Human-readable description.
            **metadata: Additional metadata attached to the entry.

        Raises:
            TypeError: If *factory* return value is not a ``BaseSandbox``.
        """
        instance = factory()
        if not isinstance(instance, BaseSandbox):
            raise TypeError(
                f"Factory for '{name}' returned {type(instance).__name__}, "
                f"expected BaseSandbox instance"
            )
        with self._registry_lock:
            self._backends[name] = instance
            self._factory[name] = factory
            self._metadata[name] = {
                "name": name,
                "description": description,
                "builtin": False,
                **metadata,
            }
        logger.info("Sandbox backend registered", extra={"backend": name, "type": type(instance).__name__})

    def get_backend(self, name: str) -> BaseSandbox:
        """Get a sandbox backend by *name*.

        Raises:
            ConfigurationError: If *name* is not registered.
        """
        with self._registry_lock:
            backend = self._backends.get(name)
        if backend is None:
            raise ConfigurationError(
                detail=f"Sandbox backend '{name}' is not registered",
                context={"available": list(self._backends.keys())},
            )
        return backend

    def list_backends(self) -> List[str]:
        """Return the names of all registered backends."""
        with self._registry_lock:
            return list(self._backends.keys())

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a registered backend, or None."""
        with self._registry_lock:
            return self._metadata.get(name)

    def remove(self, name: str) -> None:
        """Remove a registered backend by *name*."""
        with self._registry_lock:
            self._backends.pop(name, None)
            self._factory.pop(name, None)
            self._metadata.pop(name, None)

    def clear(self) -> None:
        """Remove all backends and re-register the built-in."""
        with self._registry_lock:
            self._backends.clear()
            self._factory.clear()
            self._metadata.clear()
        self._register_builtin()


sandbox_registry = SandboxRegistry()
