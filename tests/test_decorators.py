"""Unit tests for @agent, @model, @tool, @provider decorators."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langdeep.core.decorators.agent import agent
from langdeep.core.decorators.model import model
from langdeep.core.decorators.tool import register_tool, regist_tool
from langdeep.core.adapters.deepseek import configure_deepseek_v4
from langdeep.core.decorators.provider import (
    anthropic_provider,
    azure_provider,
    deepseek_provider,
    google_genai_provider,
    ollama_provider,
    openai_provider,
    provider,
    vertexai_provider,
)
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata
from langdeep.core.registry.model_registry import model_registry, ModelConfig
from langdeep.core.registry.tool_registry import tool_registry, ToolMetadata

from conftest import clean_registries


def setup_function():
    clean_registries()


def test_agent_decorator():
    @agent(name="math_bot", description="Solves math", capabilities=["math"],
           routing_keywords=["calculate"], model="gpt4o", tools=["calc"], priority=2)
    def create_math_bot():
        class Agent:
            def invoke(self, state):
                return {"messages": [AIMessage(content="42")]}
        return Agent()

    assert "math_bot" in agent_registry.list_agents()
    meta = agent_registry.get_metadata("math_bot")
    assert meta.description == "Solves math"
    assert meta.routing_keywords == ["calculate"]
    assert meta.tools == ["calc"]
    assert meta.priority == 2

    cleaned = clean_registries()


def test_agent_decorator_auto_build_metadata():
    @agent(
        name="weather_expert",
        model="mock",
        tools=["get_weather"],
        prompt_path="prompts/weather.md",
        auto_build=True,
        agent_type="react",
    )
    def weather_expert():
        pass

    meta = agent_registry.get_metadata("weather_expert")
    assert meta.auto_build is True
    assert meta.agent_type == "react"
    assert meta.prompt_path == "prompts/weather.md"


def test_agent_decorator_default_name():
    @agent(description="Uses function name")
    def my_custom_agent():
        class A:
            def invoke(self, s):
                return {"messages": [AIMessage(content="ok")]}
        return A()

    assert "my_custom_agent" in agent_registry.list_agents()


def test_model_decorator():
    @model(name="test-gpt", provider="mock", model_name="test-model", temperature=0.5)
    def my_model():
        pass

    assert "test-gpt" in model_registry.list_models()

    cleaned = clean_registries()


def test_model_decorator_accepts_extra_params_mapping():
    @model(
        name="deepseek-v4-test",
        provider="deepseek",
        model_name="deepseek-v4-pro",
        extra_params={
            "extra_body": {"thinking": {"type": "enabled"}},
            "reasoning_effort": "high",
        },
    )
    def deepseek_v4():
        pass

    config = model_registry.get_config("deepseek-v4-test")
    assert "extra_params" not in config.extra_params
    assert config.extra_params["extra_body"]["thinking"]["type"] == "enabled"
    assert config.extra_params["reasoning_effort"] == "high"


def test_model_decorator_accepts_deepseek_config_helper():
    @model(
        name="deepseek-v4-helper-test",
        provider="deepseek",
        model_name="deepseek-v4-pro",
        extra_params=configure_deepseek_v4(
            thinking="enabled",
            reasoning_effort="high",
        ),
    )
    def deepseek_v4_helper():
        pass

    config = model_registry.get_config("deepseek-v4-helper-test")
    assert config.extra_params["extra_body"]["thinking"]["type"] == "enabled"
    assert config.extra_params["reasoning_effort"] == "high"
    assert config.extra_params["reasoning_content_policy"] == "auto"


def test_model_decorator_merges_extra_params_mapping_with_kwargs():
    @model(
        name="merged-extra-params",
        provider="mock",
        model_name="mock",
        extra_params={"timeout": 10, "metadata": {"source": "mapping"}},
        timeout=20,
    )
    def merged_model():
        pass

    config = model_registry.get_config("merged-extra-params")
    assert config.extra_params["timeout"] == 20
    assert config.extra_params["metadata"] == {"source": "mapping"}


def test_regist_tool_decorator():
    @regist_tool(name="weather_tool", category="api", tags=["weather", "external"])
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"{city}: 22C"

    assert "weather_tool" in tool_registry.list_tools()
    meta = tool_registry.get_metadata("weather_tool")
    assert meta.category == "api"
    assert "weather" in meta.tags
    assert get_weather("Paris") == "Paris: 22C"


def test_register_tool_alias():
    @register_tool(name="alias_weather_tool", category="api")
    def get_alias_weather(city: str) -> str:
        """Get weather for a city."""
        return f"{city}: 20C"

    assert "alias_weather_tool" in tool_registry.list_tools()
    assert get_alias_weather("Paris") == "Paris: 20C"

    from langdeep import register_tool as public_register_tool

    assert public_register_tool is register_tool


def test_provider_decorator():
    @provider(name="my_provider")
    def create_my_model(config):
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        class MyLLM(BaseChatModel):
            model_name: str = config.model_name
            temperature: float = config.temperature
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="my provider"))])
            @property
            def _llm_type(self):
                return "my_provider"
        return MyLLM()

    from langdeep.core.registry.model_registry import provider_registry
    factory = provider_registry.get_provider("my_provider")
    assert callable(factory)


def test_predefined_provider_decorators_register_expected_names():
    from langdeep.core.registry.model_registry import provider_registry

    decorators = [
        (openai_provider, "openai"),
        (anthropic_provider, "anthropic"),
        (azure_provider, "azure_openai"),
        (ollama_provider, "ollama"),
        (vertexai_provider, "vertexai"),
        (google_genai_provider, "google_genai"),
        (deepseek_provider, "deepseek"),
    ]

    for decorator, expected_name in decorators:
        @decorator
        def factory(config, provider_name=expected_name):
            return provider_name

        assert provider_registry.get_provider(expected_name)(None) == expected_name


# Need this import for the agent decorator test
from langchain_core.messages import AIMessage
