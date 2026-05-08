"""Provider registration helpers.

This module exposes both decorator-style and function-style APIs for
registering model providers without touching the core registry.

Examples:
    @provider(name="my_provider")
    def create_model(config: ModelConfig) -> BaseChatModel:
        ...

    def create_model(config: ModelConfig) -> BaseChatModel:
        ...
    register_provider("my_provider", create_model)
"""

from functools import wraps
from typing import Callable, Optional, TypeVar, Union, overload

from langchain_core.language_models import BaseChatModel

from ..registry.model_registry import ModelConfig, provider_registry

ProviderFactory = Callable[[ModelConfig], BaseChatModel]
F = TypeVar("F", bound=ProviderFactory)


@overload
def register_provider(name: str, factory: F) -> F:
    ...


@overload
def register_provider(name: str) -> Callable[[F], F]:
    ...


def register_provider(
    name: str,
    factory: Optional[F] = None,
) -> Union[F, Callable[[F], F]]:
    """Register a model provider factory.

    This is the function-style companion to ``@provider`` and is exported at
    ``langdeep.register_provider`` for README/API compatibility.

    Args:
        name: Provider name used by ``@model(provider=...)``.
        factory: Callable that receives ``ModelConfig`` and returns a chat model.

    Returns:
        The original factory, so it can also be used as a decorator.
    """

    def decorator(func: F) -> F:
        provider_registry.register(name, func)
        return func

    if factory is None:
        return decorator
    return decorator(factory)


def provider(name: Optional[str] = None):
    """Decorator for registering a model provider factory.

    Args:
        name: Provider name. Defaults to the wrapped function name.
    """

    def decorator(func: F) -> F:
        provider_name = name or func.__name__
        register_provider(provider_name, func)

        @wraps(func)
        def wrapper(config: ModelConfig) -> BaseChatModel:
            return func(config)

        return wrapper  # type: ignore[return-value]

    return decorator


# Pre-defined provider decorators for common providers.
def openai_provider(func: F) -> F:
    """Decorator for overriding the OpenAI provider."""
    return provider("openai")(func)


def anthropic_provider(func: F) -> F:
    """Decorator for overriding the Anthropic provider."""
    return provider("anthropic")(func)


def azure_provider(func: F) -> F:
    """Decorator for overriding the Azure OpenAI provider."""
    return provider("azure_openai")(func)


def ollama_provider(func: F) -> F:
    """Decorator for overriding the Ollama provider."""
    return provider("ollama")(func)


def vertexai_provider(func: F) -> F:
    """Decorator for overriding the Google Vertex AI provider."""
    return provider("vertexai")(func)


def google_genai_provider(func: F) -> F:
    """Decorator for overriding the Google Generative AI provider."""
    return provider("google_genai")(func)


def deepseek_provider(func: F) -> F:
    """Decorator for overriding the DeepSeek provider."""
    return provider("deepseek")(func)
