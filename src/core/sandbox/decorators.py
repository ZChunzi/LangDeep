"""Decorator-based sandbox registration."""
import functools
from typing import Any, Callable, Optional

from ..logging import get_logger
from .base import BaseSandbox
from .builtin import SubprocessSandbox
from .registry import SandboxFactory, sandbox_registry

logger = get_logger(__name__)


def sandbox(
    name: Optional[str] = None,
    description: str = "",
    **kwargs: Any,
) -> Callable[[SandboxFactory], SandboxFactory]:
    """Register a sandbox backend via decorator.

    If the decorated factory returns ``None``, a default
    ``SubprocessSandbox`` is created with the supplied **kwargs**.

    Usage::

        @sandbox(name="my_sandbox", description="Custom sandbox")
        def my_factory():
            return MySandbox()

    Args:
        name: Backend name (defaults to the factory function name).
        description: Human-readable description.
        **kwargs: Extra metadata or default ``SubprocessSandbox`` args.

    Returns:
        The original factory (now registered).
    """
    def decorator(factory: SandboxFactory) -> SandboxFactory:
        backend_name = name or factory.__name__

        @functools.wraps(factory)
        def wrapper(*args: Any, **wrapper_kwargs: Any) -> BaseSandbox:
            instance = factory(*args, **wrapper_kwargs)
            if instance is None:
                instance = SubprocessSandbox(**kwargs)
            return instance

        # Register eagerly
        instance = wrapper()
        if not isinstance(instance, BaseSandbox):
            raise TypeError(
                f"Factory '{backend_name}' returned {type(instance).__name__}, "
                f"expected BaseSandbox instance"
            )
        sandbox_registry.register(backend_name, wrapper, description=description, **kwargs)
        return factory

    return decorator
