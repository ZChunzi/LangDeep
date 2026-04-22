"""Model registration decorator."""
from functools import wraps
from typing import Optional
from ..registry.model_registry import model_registry, ModelConfig


def model(
    name: Optional[str] = None,
    provider: str = "openai",
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **extra_params
):
    """
    Model registration decorator

    Usage:
        @model(name="gpt4", provider="openai", model_name="gpt-4")
        def my_agent():
            pass
    """
    def decorator(func_or_class):
        model_id = name or func_or_class.__name__
        actual_model_name = model_name or model_id

        config = ModelConfig(
            provider=provider,
            model_name=actual_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params
        )
        model_registry.register(model_id, config)
        return func_or_class

    return decorator