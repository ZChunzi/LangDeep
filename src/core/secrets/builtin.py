"""Built-in environment-variable secrets provider."""
import os
from typing import Optional

from .base import SecretsProvider


class EnvSecretsProvider(SecretsProvider):
    """Reads secrets from environment variables.

    By default looks for ``LANGDEEP_<KEY>``. Pass an empty prefix to
    read the key as-is from ``os.environ``.
    """

    def __init__(self, prefix: str = "LANGDEEP_"):
        self._prefix = prefix

    def get_secret(self, key: str) -> Optional[str]:
        return os.environ.get(f"{self._prefix}{key}")
