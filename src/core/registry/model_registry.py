"""Model registry for LLM model management with dynamic provider registration.

This version supports:
1. Dynamic provider registration without modifying core code
2. Backward compatibility with existing code
3. Built-in providers for common LLM services
"""

from typing import Dict, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
import importlib


@dataclass
class ModelConfig:
    """Model configuration"""
    provider: str                    # Provider name
    model_name: str                  # Model name
    base_url: Optional[str] = None   # Custom endpoint
    api_key: Optional[str] = None    # API key
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


class ProviderRegistry:
    """Registry for model provider factories"""

    _instance = None
    _providers: Dict[str, Callable[[ModelConfig], BaseChatModel]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Register built-in providers
            cls._instance._register_builtin_providers()
        return cls._instance

    def _register_builtin_providers(self):
        """Register built-in providers"""
        # OpenAI provider
        self.register("openai", self._create_openai_model)
        # Anthropic provider
        self.register("anthropic", self._create_anthropic_model)
        # Azure OpenAI provider
        self.register("azure_openai", self._create_azure_model)
        # Ollama provider
        self.register("ollama", self._create_ollama_model)
        # Google Vertex AI provider
        self.register("vertexai", self._create_vertexai_model)
        # Google Generative AI provider
        self.register("google_genai", self._create_google_genai_model)
        # DeepSeek provider (OpenAI-compatible)
        self.register("deepseek", self._create_deepseek_model)
        # Mock provider for testing
        self.register("mock", self._create_mock_model)

    def register(self, provider_name: str, factory: Callable[[ModelConfig], BaseChatModel]) -> None:
        """Register a provider factory function"""
        self._providers[provider_name] = factory

    def get_provider(self, provider_name: str) -> Callable[[ModelConfig], BaseChatModel]:
        """Get provider factory function"""
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not registered. "
                           f"Available providers: {list(self._providers.keys())}")
        return self._providers[provider_name]

    def list_providers(self) -> list:
        """List all registered providers"""
        return list(self._providers.keys())

    def _create_openai_model(self, config: ModelConfig) -> BaseChatModel:
        """Create OpenAI model instance"""
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                base_url=config.base_url,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        except ImportError:
            raise ImportError("OpenAI provider not available. Install langchain-openai.")

    def _create_anthropic_model(self, config: ModelConfig) -> BaseChatModel:
        """Create Anthropic model instance"""
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                api_key=config.api_key,
                temperature=config.temperature,
                **(config.extra_params or {})
            )
        except ImportError:
            raise ImportError("Anthropic provider not available. Install langchain-anthropic.")

    def _create_azure_model(self, config: ModelConfig) -> BaseChatModel:
        """Create Azure OpenAI model instance"""
        try:
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_deployment=config.model_name,
                azure_endpoint=config.base_url,
                api_key=config.api_key,
                api_version=config.extra_params.get("api_version", "2024-02-15-preview"),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        except ImportError:
            raise ImportError("Azure OpenAI provider not available. Install langchain-openai with azure support.")

    def _create_ollama_model(self, config: ModelConfig) -> BaseChatModel:
        """Create Ollama model instance"""
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                base_url=config.base_url,
                temperature=config.temperature,
                **(config.extra_params or {})
            )
        except ImportError:
            raise ImportError("Ollama provider not available. Install langchain-ollama.")

    def _create_vertexai_model(self, config: ModelConfig) -> BaseChatModel:
        """Create Google Vertex AI model instance"""
        try:
            from langchain_google_vertexai import ChatVertexAI
            return ChatVertexAI(
                model_name=config.model_name,
                project=config.extra_params.get("project"),
                location=config.extra_params.get("location", "us-central1"),
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        except ImportError:
            raise ImportError("Google Vertex AI provider not available. Install langchain-google-vertexai.")

    def _create_google_genai_model(self, config: ModelConfig) -> BaseChatModel:
        """Create Google Generative AI model instance"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                google_api_key=config.api_key,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        except ImportError:
            raise ImportError("Google Generative AI provider not available. Install langchain-google-genai.")

    def _create_deepseek_model(self, config: ModelConfig) -> BaseChatModel:
        """Create DeepSeek model instance (OpenAI-compatible)"""
        try:
            from langchain_openai import ChatOpenAI
            # DeepSeek API endpoint (default if not specified)
            base_url = config.base_url or "https://api.deepseek.com"
            return ChatOpenAI(
                model=config.model_name,
                base_url=base_url,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        except ImportError:
            raise ImportError("DeepSeek provider requires langchain-openai.")

    def _create_mock_model(self, config: ModelConfig) -> BaseChatModel:
        """Create mock model instance for testing"""
        from typing import Any, List, Optional
        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.callbacks import CallbackManagerForLLMRun

        class MockLLM(BaseChatModel):
            """Mock LLM for testing without API keys"""

            model_name: str = "mock"
            temperature: float = 0.7

            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> Any:
                # Return a mock response
                content = f"Mock response from {self.model_name}. Messages: {len(messages)}"
                if messages:
                    last_message = messages[-1].content
                    if "calculate" in last_message.lower():
                        content = "The calculation result is 42."
                    elif "search" in last_message.lower():
                        content = "Search results: AI is a fascinating field with many applications."

                from langchain_core.outputs import ChatGeneration, ChatResult
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            @property
            def _llm_type(self) -> str:
                return "mock"

            def bind_tools(self, tools, **kwargs):
                """Mock implementation of bind_tools."""
                # For mock model, just return self since we don't actually use tools
                return self

        return MockLLM(model_name=config.model_name, temperature=config.temperature)


class ModelRegistry:
    """Model registry - centralized management of all registered models"""

    _instance = None
    _models: Dict[str, ModelConfig] = {}
    _instances: Dict[str, BaseChatModel] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.provider_registry = ProviderRegistry()
        return cls._instance

    def register(self, name: str, config: ModelConfig) -> None:
        """Register model configuration"""
        self._models[name] = config
        # Clear existing instance cache to ensure latest config is used
        if name in self._instances:
            del self._instances[name]

    def get_model(self, name: str) -> BaseChatModel:
        """Get model instance (lazy loading)"""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered")

        if name not in self._instances:
            config = self._models[name]
            self._instances[name] = self._create_model_instance(config)

        return self._instances[name]

    def _create_model_instance(self, config: ModelConfig) -> BaseChatModel:
        """Create model instance using provider factory"""
        provider_factory = self.provider_registry.get_provider(config.provider)
        return provider_factory(config)

    def list_models(self) -> list:
        """List all registered models"""
        return list(self._models.keys())

    def get_provider_registry(self) -> ProviderRegistry:
        """Get provider registry for registering custom providers"""
        return self.provider_registry


# Global singleton instances
provider_registry = ProviderRegistry()
model_registry = ModelRegistry()