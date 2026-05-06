"""Secrets management — pluggable secret resolution for production deployments."""
from .base import SecretsProvider
from .builtin import EnvSecretsProvider
from .manager import secrets_manager, SecretsManager

__all__ = [
    "SecretsProvider",
    "EnvSecretsProvider",
    "SecretsManager",
    "secrets_manager",
]
