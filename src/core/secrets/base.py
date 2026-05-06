"""Abstract base class for secret providers."""
from abc import ABC, abstractmethod
from typing import Optional


class SecretsProvider(ABC):
    """Abstract interface for secret resolution.

    Implementations read secrets from environment variables, HashiCorp Vault,
    AWS Secrets Manager, or any other secure store.
    """

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key.

        Returns None if the key is not found (never raise for missing keys).
        """
        ...
