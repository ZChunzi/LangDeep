"""Model registration decorator."""
from typing import Any, Mapping, Optional

from ..errors import ConfigurationError
from ..registry.model_registry import model_registry, ModelConfig


def _normalize_extra_params(extra_params: Mapping[str, Any]) -> dict[str, Any]:
    """Support both ``@model(..., extra_params={...})`` and ``@model(..., **params)``."""
    params = dict(extra_params)
    nested = params.pop("extra_params", None)
    if nested is None:
        return params
    if not isinstance(nested, Mapping):
        raise ConfigurationError(
            "model decorator extra_params must be a mapping when provided",
            context={"type": type(nested).__name__},
        )

    merged = dict(nested)
    merged.update(params)
    return merged


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

        @model(
            name="deepseek_v4",
            provider="deepseek",
            model_name="deepseek-v4-pro",
            extra_params={"extra_body": {"thinking": {"type": "enabled"}}},
        )
        def deepseek_v4():
            pass
    """
    def decorator(func_or_class):
        model_id = name or func_or_class.__name__
        actual_model_name = model_name or model_id
        normalized_extra_params = _normalize_extra_params(extra_params)

        config = ModelConfig(
            provider=provider,
            model_name=actual_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=normalized_extra_params
        )
        model_registry.register(model_id, config)
        return func_or_class

    return decorator
