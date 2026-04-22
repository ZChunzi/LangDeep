"""Provider registration decorator for dynamic model provider registration.

This decorator allows registering new model providers without modifying
the core model_registry.py file.

Usage:
    @provider(name="deepseek")
    def create_deepseek_model(config: ModelConfig) -> BaseChatModel:
        # Implementation using DeepSeek SDK
        from langchain_openai import ChatOpenAI
        base_url = config.base_url or "https://api.deepseek.com"
        return ChatOpenAI(
            model=config.model_name,
            base_url=base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **(config.extra_params or {})
        )

    # Then register models using this provider
    @model(name="my_deepseek", provider="deepseek", model_name="deepseek-chat")
    def my_deepseek_model(): pass
"""

from functools import wraps
from typing import Optional, Callable
from ..registry.model_registry import provider_registry, ModelConfig
from langchain_core.language_models import BaseChatModel


def provider(name: Optional[str] = None):
    """
    Provider registration decorator

    Args:
        name: Provider name (defaults to function name)

    Returns:
        Decorator function
    """
    def decorator(func: Callable[[ModelConfig], BaseChatModel]):
        provider_name = name or func.__name__

        # Register the factory function
        provider_registry.register(provider_name, func)

        @wraps(func)
        def wrapper(config: ModelConfig) -> BaseChatModel:
            return func(config)

        return wrapper

    return decorator


# Pre-defined provider decorators for common providers
def openai_provider(func: Callable[[ModelConfig], BaseChatModel]):
    """Decorator for OpenAI provider (pre-defined)"""
    return provider("openai")(func)


def anthropic_provider(func: Callable[[ModelConfig], BaseChatModel]):
    """Decorator for Anthropic provider (pre-defined)"""
    return provider("anthropic")(func)


def azure_provider(func: Callable[[ModelConfig], BaseChatModel]):
    """Decorator for Azure OpenAI provider"""
    return provider("azure_openai")(func)


def ollama_provider(func: Callable[[ModelConfig], BaseChatModel]):
    """Decorator for Ollama provider"""
    return provider("ollama")(func)


def vertexai_provider(func: Callable[[ModelConfig], BaseChatModel]):
    """Decorator for Google Vertex AI provider"""
    return provider("vertexai")(func)


def google_genai_provider(func: Callable[[ModelConfig], BaseChatModel]):
    """Decorator for Google Generative AI provider"""
    return provider("google_genai")(func)


def deepseek_provider(func: Callable[[ModelConfig], BaseChatModel]):
    """Decorator for DeepSeek provider"""
    return provider("deepseek")(func)