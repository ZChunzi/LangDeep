"""Model registry with dynamic provider registration."""

from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel

from ..logging import get_logger
from ..errors import (
    ModelNotFoundError,
    ProviderNotFoundError,
    ProviderImportError,
)

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    provider: str
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


class ProviderRegistry:
    """Registry for model provider factories."""

    _instance = None
    _providers: Dict[str, Callable[[ModelConfig], BaseChatModel]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._register_builtin_providers()
        return cls._instance

    def _register_builtin_providers(self):
        self.register("openai", self._create_openai_model)
        self.register("anthropic", self._create_anthropic_model)
        self.register("azure_openai", self._create_azure_model)
        self.register("ollama", self._create_ollama_model)
        self.register("vertexai", self._create_vertexai_model)
        self.register("google_genai", self._create_google_genai_model)
        self.register("deepseek", self._create_deepseek_model)
        self.register("mock", self._create_mock_model)
        logger.info("Built-in providers registered", extra={"providers": list(self._providers.keys())})

    def register(self, provider_name: str, factory: Callable[[ModelConfig], BaseChatModel]) -> None:
        self._providers[provider_name] = factory
        logger.debug("Provider registered", extra={"provider": provider_name})

    def get_provider(self, provider_name: str) -> Callable[[ModelConfig], BaseChatModel]:
        if provider_name not in self._providers:
            raise ProviderNotFoundError(
                f"Provider '{provider_name}' is not registered",
                context={"available": list(self._providers.keys())},
            )
        return self._providers[provider_name]

    def list_providers(self) -> list:
        return list(self._providers.keys())

    # ── Provider factories ───────────────────────────────────────────────────

    def _create_openai_model(self, config: ModelConfig) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ProviderImportError(
                "OpenAI provider requires langchain-openai package",
                context={"provider": "openai"},
            )
        return ChatOpenAI(
            model=config.model_name,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **(config.extra_params or {}),
        )

    def _create_anthropic_model(self, config: ModelConfig) -> BaseChatModel:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ProviderImportError(
                "Anthropic provider requires langchain-anthropic package",
                context={"provider": "anthropic"},
            )
        return ChatAnthropic(
            model=config.model_name,
            api_key=config.api_key,
            temperature=config.temperature,
            **(config.extra_params or {}),
        )

    def _create_azure_model(self, config: ModelConfig) -> BaseChatModel:
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ProviderImportError(
                "Azure OpenAI provider requires langchain-openai package",
                context={"provider": "azure_openai"},
            )
        return AzureChatOpenAI(
            azure_deployment=config.model_name,
            azure_endpoint=config.base_url,
            api_key=config.api_key,
            api_version=config.extra_params.get("api_version", "2024-02-15-preview"),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **(config.extra_params or {}),
        )

    def _create_ollama_model(self, config: ModelConfig) -> BaseChatModel:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ProviderImportError(
                "Ollama provider requires langchain-ollama package",
                context={"provider": "ollama"},
            )
        return ChatOllama(
            model=config.model_name,
            base_url=config.base_url,
            temperature=config.temperature,
            **(config.extra_params or {}),
        )

    def _create_vertexai_model(self, config: ModelConfig) -> BaseChatModel:
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError:
            raise ProviderImportError(
                "Google Vertex AI provider requires langchain-google-vertexai package",
                context={"provider": "vertexai"},
            )
        return ChatVertexAI(
            model_name=config.model_name,
            project=config.extra_params.get("project"),
            location=config.extra_params.get("location", "us-central1"),
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            **(config.extra_params or {}),
        )

    def _create_google_genai_model(self, config: ModelConfig) -> BaseChatModel:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ProviderImportError(
                "Google Generative AI provider requires langchain-google-genai package",
                context={"provider": "google_genai"},
            )
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            **(config.extra_params or {}),
        )

    def _create_deepseek_model(self, config: ModelConfig) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ProviderImportError(
                "DeepSeek provider requires langchain-openai package",
                context={"provider": "deepseek"},
            )
        base_url = config.base_url or "https://api.deepseek.com"
        return ChatOpenAI(
            model=config.model_name,
            base_url=base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **(config.extra_params or {}),
        )

    def _create_mock_model(self, config: ModelConfig) -> BaseChatModel:
        from typing import List as ListType, Optional as Opt
        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.outputs import ChatGeneration, ChatResult

        class MockLLM(BaseChatModel):
            model_name: str = "mock"
            temperature: float = 0.7

            def _generate(
                self,
                messages: ListType[BaseMessage],
                stop: Opt[ListType[str]] = None,
                run_manager: Opt[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> Any:
                content = f"Mock response from {self.model_name}. Messages: {len(messages)}"
                if messages:
                    last = messages[-1].content
                    if "calculate" in str(last).lower():
                        content = "The calculation result is 42."
                    elif "search" in str(last).lower():
                        content = "Search results: AI is a fascinating field."
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

            @property
            def _llm_type(self) -> str:
                return "mock"

            def bind_tools(self, tools, **kwargs):
                return self

        return MockLLM(model_name=config.model_name, temperature=config.temperature)


class ModelRegistry:
    """Central model registry — maps model names to provider+config."""

    _instance = None
    _models: Dict[str, ModelConfig] = {}
    _instances: Dict[str, BaseChatModel] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._provider_registry = ProviderRegistry()
        return cls._instance

    def register(self, name: str, config: ModelConfig) -> None:
        self._models[name] = config
        if name in self._instances:
            del self._instances[name]
        logger.info("Model registered", extra={"model_name": name, "provider": config.provider})

    def get_model(self, name: str) -> BaseChatModel:
        if name not in self._models:
            raise ModelNotFoundError(
                f"Model '{name}' is not registered",
                context={"available": list(self._models.keys())},
            )
        if name not in self._instances:
            config = self._models[name]
            self._instances[name] = self._create_instance(config)
            logger.debug("Model instance created", extra={"model_name": name})
        return self._instances[name]

    def _create_instance(self, config: ModelConfig) -> BaseChatModel:
        factory = self._provider_registry.get_provider(config.provider)
        return factory(config)

    def list_models(self) -> list:
        return list(self._models.keys())

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._provider_registry


# Global singletons
provider_registry = ProviderRegistry()
model_registry = ModelRegistry()
