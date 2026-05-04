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
from langdeep.core.errors import ModelNotFoundError, ProviderNotFoundError

from conftest import clean_registries


def setup_function():
    clean_registries()
    # Re-register builtin providers (cleared by clean_registries)
    provider_registry._register_builtin_providers()


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
    provider_registry._register_builtin_providers()
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
