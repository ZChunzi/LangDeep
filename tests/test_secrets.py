"""Unit tests for the secrets module: EnvSecretsProvider, SecretsManager."""

import os

from langdeep.core.secrets import (
    EnvSecretsProvider,
    SecretsProvider,
    SecretsManager,
    secrets_manager,
)
from langdeep.core.errors import SecretsError


def setup_function():
    """Reset singleton registries before each test."""
    from tests.conftest import clean_registries
    clean_registries()
    # Clean up any test env vars
    for k in list(os.environ.keys()):
        if k.startswith("TEST_SECRETS_"):
            del os.environ[k]


# ── ABC ─────────────────────────────────────────────────────────────────────


def test_secrets_provider_is_abstract():
    """SecretsProvider cannot be instantiated directly."""
    try:
        SecretsProvider()  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# ── EnvSecretsProvider ──────────────────────────────────────────────────────


def test_env_provider_reads_existing_key():
    """get_secret returns the value for an existing key."""
    os.environ["LANGDEEP_TEST_KEY"] = "hello"
    provider = EnvSecretsProvider()
    assert provider.get_secret("TEST_KEY") == "hello"


def test_env_provider_returns_none_for_missing():
    """get_secret returns None for a non-existent key."""
    provider = EnvSecretsProvider()
    assert provider.get_secret("NONEXISTENT_999") is None


def test_env_provider_custom_prefix():
    """Custom prefix is used correctly."""
    os.environ["MY_APP_DB_PASS"] = "s3cret"
    provider = EnvSecretsProvider(prefix="MY_APP_")
    assert provider.get_secret("DB_PASS") == "s3cret"


def test_env_provider_empty_prefix():
    """Empty prefix means the key is used as-is."""
    os.environ["DIRECT_KEY"] = "val"
    provider = EnvSecretsProvider(prefix="")
    assert provider.get_secret("DIRECT_KEY") == "val"


def test_env_provider_empty_value():
    """Empty string values are returned as empty string (not None)."""
    os.environ["LANGDEEP_EMPTY_VAL"] = ""
    provider = EnvSecretsProvider()
    assert provider.get_secret("EMPTY_VAL") == ""


# ── SecretsManager ──────────────────────────────────────────────────────────


def test_manager_singleton():
    """SecretsManager is a singleton."""
    m1 = SecretsManager()
    m2 = SecretsManager()
    assert m1 is m2


def test_manager_register_and_get():
    """Register a provider and resolve a secret."""
    os.environ["LANGDEEP_MGR_KEY"] = "mgr_val"
    provider = EnvSecretsProvider()
    secrets_manager.register_provider(provider)
    assert secrets_manager.get_secret("MGR_KEY") == "mgr_val"


def test_manager_public_aliases():
    """README-facing aliases stay compatible with the core methods."""
    os.environ["LANGDEEP_ALIAS_KEY"] = "alias_val"
    secrets_manager.add_provider(EnvSecretsProvider())
    assert secrets_manager.resolve("ALIAS_KEY") == "alias_val"


def test_manager_fallback_multiple_providers():
    """Second provider is consulted when first returns None."""
    os.environ["LANGDEEP_FALLBACK"] = "from_env"
    p1 = EnvSecretsProvider()
    # A second provider with empty prefix
    p2 = EnvSecretsProvider(prefix="")
    secrets_manager.register_provider(p1)
    secrets_manager.register_provider(p2)
    assert secrets_manager.get_secret("FALLBACK") == "from_env"


def test_manager_returns_none_when_not_found():
    """Returns None when no provider has the key."""
    secrets_manager.register_provider(EnvSecretsProvider())
    assert secrets_manager.get_secret("DOES_NOT_EXIST_ANYWHERE") is None


def test_manager_provider_exception_isolation():
    """A failing provider does not crash the manager — next provider is tried."""

    class BrokenProvider(SecretsProvider):
        def get_secret(self, key):
            raise RuntimeError("broken")

    class GoodProvider(SecretsProvider):
        def get_secret(self, key):
            return "recovered"

    secrets_manager.register_provider(BrokenProvider())
    secrets_manager.register_provider(GoodProvider())
    assert secrets_manager.get_secret("anything") == "recovered"


def test_manager_list_providers():
    """list_providers returns registered provider class names."""
    secrets_manager.register_provider(EnvSecretsProvider())
    names = secrets_manager.list_providers()
    assert "EnvSecretsProvider" in names


def test_manager_clear():
    """clear removes all providers."""
    secrets_manager.register_provider(EnvSecretsProvider())
    secrets_manager.clear()
    assert secrets_manager.list_providers() == []


def test_manager_register_none_raises():
    """register_provider(None) raises TypeError."""
    try:
        secrets_manager.register_provider(None)  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_manager_register_wrong_type_raises():
    """register_provider with non-SecretsProvider raises TypeError."""

    class NotAProvider:
        def get_secret(self, key):
            return "nope"

    try:
        secrets_manager.register_provider(NotAProvider())  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
