"""Unit tests for ModelRegistry and ProviderRegistry."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langdeep.core.registry.model_registry import (
    model_registry, provider_registry, ModelRegistry,
    ProviderRegistry, ModelConfig,
)
from langdeep.core.errors import ConfigurationError, ModelNotFoundError, ProviderNotFoundError

from conftest import clean_registries


def setup_function():
    clean_registries()
    provider_registry.reset()


def test_model_registry_singleton():
    assert ModelRegistry() is model_registry


def test_register_and_get_model():
    model_registry.register("test_model", ModelConfig(provider="mock", model_name="mock-test"))
    model = model_registry.get_model("test_model")
    assert isinstance(model, BaseChatModel)
    assert model.model_name == "mock-test"
    # Cached
    assert model_registry.get_model("test_model") is model


def test_get_model_not_found():
    try:
        model_registry.get_model("nope")
        assert False, "Should raise"
    except ModelNotFoundError:
        pass


def test_list_models():
    clean_registries()
    provider_registry.reset()
    assert model_registry.list_models() == []
    model_registry.register("m1", ModelConfig(provider="mock", model_name="m1"))
    assert "m1" in model_registry.list_models()


def test_re_registration_clears_cache():
    model_registry.register("re_test", ModelConfig(provider="mock", model_name="v1"))
    m1 = model_registry.get_model("re_test")
    model_registry.register("re_test", ModelConfig(provider="mock", model_name="v2"))
    m2 = model_registry.get_model("re_test")
    assert m2 is not m1
    assert m2.model_name == "v2"


def test_provider_registry_singleton():
    assert ProviderRegistry() is provider_registry


def test_register_and_get_provider():
    # mock provider should be builtin
    factory = provider_registry.get_provider("mock")
    assert callable(factory)
    model = factory(ModelConfig(provider="mock", model_name="test"))
    assert isinstance(model, BaseChatModel)


def test_get_provider_not_found():
    try:
        provider_registry.get_provider("non_existent_provider")
        assert False, "Should raise"
    except ProviderNotFoundError:
        pass


def test_list_providers():
    providers = provider_registry.list_providers()
    assert "mock" in providers
    assert "openai" in providers


def test_mock_model_responds():
    model_registry.register("my_mock", ModelConfig(provider="mock", model_name="my-mock"))
    llm = model_registry.get_model("my_mock")
    from langchain_core.messages import HumanMessage
    result = llm.invoke([HumanMessage(content="hello")])
    assert result.content is not None
    assert len(str(result.content)) > 0


def test_custom_provider():
    def custom_factory(config):
        class CustomLLM(BaseChatModel):
            model_name: str = config.model_name
            temperature: float = config.temperature
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content=f"custom: {config.model_name}")
                )])
            @property
            def _llm_type(self):
                return "custom"
        return CustomLLM()

    from langchain_core.messages import AIMessage, HumanMessage
    provider_registry.register("custom_proto", custom_factory)
    model_registry.register("custom_model", ModelConfig(provider="custom_proto", model_name="cm"))
    llm = model_registry.get_model("custom_model")
    resp = llm.invoke([HumanMessage(content="x")])
    assert "custom: cm" in str(resp.content)


def test_model_registry_lifecycle_snapshot_reset_and_duplicate_policy():
    model_registry.register("life", ModelConfig(provider="mock", model_name="life"))
    model = model_registry.get_model("life")
    snapshot = model_registry.snapshot()

    assert snapshot["namespace"] == "default"
    assert "life" in snapshot["models"]
    assert snapshot["cached_model_count"] == 1
    assert model_registry.get_model("life") is model

    try:
        model_registry.register("life", ModelConfig(provider="mock", model_name="other"), replace=False)
        assert False, "Should reject duplicate registration when replace=False"
    except ConfigurationError:
        pass

    model_registry.reset()
    assert model_registry.list_models() == []


def test_model_registry_namespace_isolation_and_manual_instance():
    tenant = ModelRegistry.for_namespace("tenant-a")
    tenant.reset()
    tenant.register("tenant_model", ModelConfig(provider="mock", model_name="tenant"))
    tenant_model = tenant.get_model("tenant_model")

    assert tenant.namespace == "tenant-a"
    assert ModelRegistry("tenant-a") is tenant
    assert "tenant_model" in tenant.list_models()
    assert "tenant_model" not in model_registry.list_models()

    replacement = tenant_model
    tenant.set_model_instance("tenant_model", replacement)
    assert tenant.get_model("tenant_model") is replacement
    tenant.reset()


def test_provider_registry_reset_snapshot_and_duplicate_policy():
    providers = provider_registry.snapshot()
    assert "mock" in providers

    def custom_factory(config):
        return provider_registry.get_provider("mock")(config)

    provider_registry.register("dup_provider", custom_factory)
    try:
        provider_registry.register("dup_provider", custom_factory, replace=False)
        assert False, "Should reject duplicate provider when replace=False"
    except ConfigurationError:
        pass

    provider_registry.reset(include_builtins=False)
    assert provider_registry.list_providers() == []
    provider_registry.reset()
    assert "mock" in provider_registry.list_providers()
