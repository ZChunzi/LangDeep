"""Model registry with dynamic provider registration."""

import copy
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ..logging import get_logger
from ..errors import (
    ConfigurationError,
    ModelNotFoundError,
    ProviderNotFoundError,
    ProviderImportError,
)
from ..cache import BaseCacheBackend, FileCacheBackend, MemoryCache

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
    _class_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._providers: Dict[str, Callable[[ModelConfig], BaseChatModel]] = {}
                    cls._instance._registry_lock = threading.RLock()
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
        self.register("mock", _create_mock_model)
        logger.info("Built-in providers registered", extra={"providers": list(self._providers.keys())})

    def register(
        self,
        provider_name: str,
        factory: Callable[[ModelConfig], BaseChatModel],
        *,
        replace: bool = True,
    ) -> None:
        with self._registry_lock:
            if provider_name in self._providers and not replace:
                raise ConfigurationError(
                    f"Provider '{provider_name}' is already registered",
                    context={"provider": provider_name},
                )
            self._providers[provider_name] = factory
        logger.debug("Provider registered", extra={"provider": provider_name})

    def get_provider(self, provider_name: str) -> Callable[[ModelConfig], BaseChatModel]:
        with self._registry_lock:
            if provider_name not in self._providers:
                raise ProviderNotFoundError(
                    f"Provider '{provider_name}' is not registered",
                    context={"available": list(self._providers.keys())},
                )
            return self._providers[provider_name]

    def list_providers(self) -> list:
        with self._registry_lock:
            return list(self._providers.keys())

    def snapshot(self) -> Dict[str, Callable[[ModelConfig], BaseChatModel]]:
        """Return a shallow snapshot of registered provider factories."""
        with self._registry_lock:
            return dict(self._providers)

    def reset(self, *, include_builtins: bool = True) -> None:
        """Clear providers and optionally re-register built-ins."""
        with self._registry_lock:
            self._providers.clear()
            if include_builtins:
                self._register_builtin_providers()

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
            from ..adapters.deepseek import DeepSeekChatModel, DeepSeekCompatibilityProfile
        except ImportError:
            raise ProviderImportError(
                "DeepSeek provider requires langchain-openai package",
                context={"provider": "deepseek"},
            )
        base_url = config.base_url or "https://api.deepseek.com"
        extra = _normalize_deepseek_extra_params(config.extra_params or {})

        # Build kwargs from ModelConfig — let extra_params override if needed.
        kwargs: Dict[str, Any] = {
            "model": config.model_name,
            "base_url": base_url,
            "api_key": config.api_key,
        }
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        kwargs.update(extra)  # extra_params take precedence

        profile = DeepSeekCompatibilityProfile(
            model_name=config.model_name,
            reasoning_content_policy=extra.get("reasoning_content_policy", "auto"),
            thinking_enabled=_detect_deepseek_thinking(extra),
        )
        logger.info(
            "DeepSeek provider selected — using DeepSeekChatModel",
            extra={
                "reasoning_content_policy": profile.resolved_reasoning_content_policy(),
                "thinking_enabled": profile.thinking_enabled,
            },
        )
        return DeepSeekChatModel(**kwargs)


def _normalize_deepseek_extra_params(extra_params: Dict[str, Any]) -> Dict[str, Any]:
    """Accept low-friction DeepSeek thinking config and convert it to SDK shape."""
    params = dict(extra_params)
    thinking = params.pop("thinking", None)
    if thinking is not None:
        extra_body = dict(params.get("extra_body") or {})
        extra_body.setdefault("thinking", thinking)
        params["extra_body"] = extra_body
    return params


def _detect_deepseek_thinking(extra_params: Dict[str, Any]) -> Optional[bool]:
    """Check whether *extra_params* explicitly toggles DeepSeek thinking mode."""
    extra_body = extra_params.get("extra_body") or {}
    thinking = extra_body.get("thinking") or {}
    if thinking.get("type") == "enabled":
        return True
    if thinking.get("type") == "disabled":
        return False
    enabled = thinking.get("enabled")
    return enabled if isinstance(enabled, bool) else None


# ── Mock provider ────────────────────────────────────────────────────────────────────


def _create_mock_model(config: ModelConfig) -> BaseChatModel:
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
    """Namespace-aware model registry mapping model names to provider configs."""

    _instance = None
    _registries: Dict[str, "ModelRegistry"] = {}
    _class_lock = threading.Lock()

    def __new__(cls, namespace: str = "default"):
        namespace = namespace or "default"
        with cls._class_lock:
            if namespace not in cls._registries:
                instance = super().__new__(cls)
                instance._namespace = namespace
                instance._models: Dict[str, ModelConfig] = {}
                instance._provider_registry = ProviderRegistry()
                # Replace unbounded instance storage with an LRU cache.
                instance._instance_cache: MemoryCache = MemoryCache(max_size=16)
                instance._response_cache: BaseCacheBackend = MemoryCache(max_size=0)
                instance._registry_lock = threading.RLock()
                cls._registries[namespace] = instance
                if namespace == "default":
                    cls._instance = instance
            return cls._registries[namespace]

    @classmethod
    def for_namespace(cls, namespace: str) -> "ModelRegistry":
        """Return an isolated registry for a namespace."""
        return cls(namespace=namespace)

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(self, name: str, config: ModelConfig, *, replace: bool = True) -> None:
        with self._registry_lock:
            if name in self._models and not replace:
                raise ConfigurationError(
                    f"Model '{name}' is already registered",
                    context={"model": name, "namespace": self._namespace},
                )
            self._models[name] = copy.deepcopy(config)
            self._instance_cache.delete(name)
        logger.info(
            "Model registered",
            extra={"model_name": name, "namespace": self._namespace, "provider": config.provider},
        )

    def get_model(self, name: str) -> BaseChatModel:
        with self._registry_lock:
            if name not in self._models:
                raise ModelNotFoundError(
                    f"Model '{name}' is not registered",
                    context={"available": list(self._models.keys())},
                )
            instance = self._instance_cache.get(name)
            if instance is None:
                config = self._models[name]
                instance = self._create_instance(config)
                self._instance_cache.set(name, instance)
                logger.debug(
                    "Model instance created",
                    extra={"model_name": name, "namespace": self._namespace},
                )
            return instance

    def set_model_instance(self, name: str, instance: BaseChatModel) -> None:
        """Set a cached model instance for a registered model."""
        if not isinstance(instance, BaseChatModel):
            raise TypeError(f"Expected BaseChatModel, got {type(instance).__name__}")
        with self._registry_lock:
            if name not in self._models:
                raise ModelNotFoundError(
                    f"Model '{name}' is not registered",
                    context={"available": list(self._models.keys())},
                )
            self._instance_cache.set(name, instance)

    def _create_instance(self, config: ModelConfig) -> BaseChatModel:
        factory = self._provider_registry.get_provider(config.provider)
        return factory(config)

    def list_models(self) -> list:
        with self._registry_lock:
            return list(self._models.keys())

    def get_config(self, name: str) -> ModelConfig:
        """Return a copy of the registered model config."""
        with self._registry_lock:
            if name not in self._models:
                raise ModelNotFoundError(
                    f"Model '{name}' is not registered",
                    context={"available": list(self._models.keys())},
                )
            return copy.deepcopy(self._models[name])

    def list_model_configs(self) -> Dict[str, ModelConfig]:
        """Return a copy of all registered model configs keyed by model name."""
        with self._registry_lock:
            return copy.deepcopy(self._models)

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._provider_registry

    # ── Response caching (opt-in) ──────────────────────────────────────────────

    def enable_response_cache(
        self,
        ttl: int = 300,
        max_entries: int = 1024,
        disk_path: Optional[str] = None,
    ) -> None:
        """Enable LLM response caching (off by default — changes LLM semantics)."""
        with self._registry_lock:
            if disk_path:
                self._response_cache = FileCacheBackend(
                    disk_path,
                    max_entries=max_entries,
                    default_ttl=ttl,
                )
            else:
                self._response_cache = MemoryCache(max_size=max_entries, default_ttl=ttl)
        logger.info(
            "LLM response cache enabled",
            extra={"ttl": ttl, "max_entries": max_entries, "disk_path": disk_path},
        )

    def disable_response_cache(self) -> None:
        with self._registry_lock:
            self._response_cache.clear()
            self._response_cache = MemoryCache(max_size=0)
        logger.info("LLM response cache disabled")

    def set_response_cache(self, backend: BaseCacheBackend) -> None:
        """Use a custom cache backend for LLM responses."""
        with self._registry_lock:
            self._response_cache = backend
        logger.info("LLM response cache set", extra={"backend": type(backend).__name__})

    def invoke_with_cache(
        self,
        model_name: str,
        messages: Sequence[BaseMessage],
        *,
        cache_context: Optional[Dict[str, Any]] = None,
        invoker: Optional[Callable[..., Any]] = None,
        **kwargs,
    ) -> Any:
        """Invoke a model with response caching (cache is checked on hit, populated on miss).

        Only active when response cache is enabled (see ``enable_response_cache``).
        ``cache_context`` should include component-specific inputs not already
        represented in the message payload, such as bound tool names.
        """
        llm = self.get_model(model_name)
        key = _response_cache_key(
            model_name,
            messages,
            kwargs=kwargs,
            context=cache_context,
        )

        with self._registry_lock:
            response_cache = self._response_cache

        cached = response_cache.get(key)
        if cached is not None:
            logger.debug("LLM cache hit", extra={"model": model_name, "cache_key": key})
            return cached

        logger.debug("LLM cache miss", extra={"model": model_name, "cache_key": key})
        call = invoker or llm.invoke
        result = call(messages, **kwargs)
        response_cache.set(key, result)
        return result

    def snapshot(self) -> Dict[str, Any]:
        """Return a runtime snapshot with copied configs and cache sizes."""
        with self._registry_lock:
            return {
                "namespace": self._namespace,
                "models": copy.deepcopy(self._models),
                "cached_model_count": len(self._instance_cache),
                "response_cache_type": type(self._response_cache).__name__,
            }

    def reset(self) -> None:
        """Clear model configs and runtime caches."""
        with self._registry_lock:
            self._models.clear()
            self._instance_cache = MemoryCache(max_size=16)
            self._response_cache = MemoryCache(max_size=0)


# Global singletons
provider_registry = ProviderRegistry()
model_registry = ModelRegistry()


def _response_cache_key(
    model_name: str,
    messages: Sequence[BaseMessage],
    *,
    kwargs: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    import hashlib
    import json

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": getattr(message, "type", type(message).__name__),
                "content": _json_safe(getattr(message, "content", "")),
                "additional_kwargs": _json_safe(getattr(message, "additional_kwargs", {})),
            }
            for message in messages
        ],
        "kwargs": _json_safe(kwargs or {}),
        "context": _json_safe(context or {}),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{model_name}:{hashlib.sha256(serialized.encode()).hexdigest()[:32]}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
