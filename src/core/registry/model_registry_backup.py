"""Model registry for LLM model management."""
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_google_vertexai import ChatVertexAI
try:
    from langchain_openai import AzureChatOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False


@dataclass
class ModelConfig:
    """Model configuration"""
    provider: str                    # Provider: openai / anthropic / azure / ollama
    model_name: str                  # Model name
    base_url: Optional[str] = None   # Custom endpoint
    api_key: Optional[str] = None    # API key
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Model registry - centralized management of all registered models"""

    _instance = None
    _models: Dict[str, ModelConfig] = {}
    _instances: Dict[str, BaseChatModel] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
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
        """Create model instance based on configuration"""
        if config.provider == "openai":
            return ChatOpenAI(
                model=config.model_name,
                base_url=config.base_url,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        elif config.provider == "anthropic":
            return ChatAnthropic(
                model=config.model_name,
                api_key=config.api_key,
                temperature=config.temperature,
                **(config.extra_params or {})
            )
        elif config.provider == "azure_openai":
            if not AZURE_AVAILABLE:
                raise ImportError("Azure OpenAI not available. Install langchain-openai with azure support.")
            return AzureChatOpenAI(
                azure_deployment=config.model_name,
                azure_endpoint=config.base_url,
                api_key=config.api_key,
                api_version=config.extra_params.get("api_version", "2024-02-15-preview"),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        elif config.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not available. Install langchain-ollama.")
            return ChatOllama(
                model=config.model_name,
                base_url=config.base_url,
                temperature=config.temperature,
                **(config.extra_params or {})
            )
        elif config.provider == "vertexai":
            return ChatVertexAI(
                model_name=config.model_name,
                project=config.extra_params.get("project"),
                location=config.extra_params.get("location", "us-central1"),
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        elif config.provider == "google_genai":
            if not GOOGLE_GENAI_AVAILABLE:
                raise ImportError("Google Generative AI not available. Install langchain-google-genai.")
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                google_api_key=config.api_key,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        elif config.provider == "deepseek":
            # DeepSeek provides OpenAI-compatible API
            # Default endpoint for DeepSeek API
            base_url = config.base_url or "https://api.deepseek.com"
            return ChatOpenAI(
                model=config.model_name,
                base_url=base_url,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **(config.extra_params or {})
            )
        elif config.provider == "mock":
            from typing import Any, List
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

            return MockLLM(model_name=config.model_name, temperature=config.temperature)
        # Support more providers...
        else:
            raise ValueError(f"Unsupported model provider: {config.provider}")

    def list_models(self) -> list:
        """List all registered models"""
        return list(self._models.keys())


# Global singleton
model_registry = ModelRegistry()