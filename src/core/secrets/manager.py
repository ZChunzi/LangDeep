"""SecretsManager singleton — holds an ordered list of SecretsProviders."""
import threading
from typing import List, Optional

from ..logging import get_logger
from .base import SecretsProvider

logger = get_logger(__name__)


class SecretsManager:
    """Singleton that resolves secrets by iterating registered providers.

    Providers are checked in registration order; the first non-None result
    wins.  This lets you register a fallback provider (e.g. file-based)
    behind a primary one (e.g. Vault).
    """

    _instance = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._providers: List[SecretsProvider] = []
                    cls._instance._providers_lock = threading.Lock()
        return cls._instance

    def register_provider(self, provider: SecretsProvider) -> None:
        """Register a secrets provider.

        Args:
            provider: Must be a non-None ``SecretsProvider`` instance.
        """
        if provider is None:
            raise TypeError("provider must not be None")
        if not isinstance(provider, SecretsProvider):
            raise TypeError(f"Expected SecretsProvider, got {type(provider).__name__}")
        with self._providers_lock:
            self._providers.append(provider)
        logger.info("Secrets provider registered", extra={"type": type(provider).__name__})

    def add_provider(self, provider: SecretsProvider) -> None:
        """Alias for :meth:`register_provider`.

        Kept for compatibility with the public README and developer guide.
        """
        self.register_provider(provider)

    def get_secret(self, key: str) -> Optional[str]:
        """Resolve a secret by trying each registered provider in order.

        Returns the first non-None value, or None if no provider has the key.
        Provider exceptions are caught, logged, and do not propagate.
        """
        with self._providers_lock:
            providers = list(self._providers)
        for provider in providers:
            try:
                value = provider.get_secret(key)
                if value is not None:
                    return value
            except Exception:
                logger.warning(
                    "Secrets provider failed",
                    extra={"provider": type(provider).__name__, "key": key},
                )
        return None

    def resolve(self, key: str) -> Optional[str]:
        """Alias for :meth:`get_secret`.

        This name reads better in application code that resolves logical
        secret names such as ``"database.url"``.
        """
        return self.get_secret(key)

    def list_providers(self) -> List[str]:
        """Return the class names of all registered providers."""
        with self._providers_lock:
            return [type(p).__name__ for p in self._providers]

    def clear(self) -> None:
        """Remove all registered providers (used for test isolation)."""
        with self._providers_lock:
            self._providers.clear()


secrets_manager = SecretsManager()
