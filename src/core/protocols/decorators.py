"""Decorator helpers for protocol adapter registration."""

from typing import Any, Callable, Optional, Sequence, TypeVar

from .models import (
    FunctionProtocolAdapter,
    ProtocolAdapter,
    ProtocolEndpoint,
    make_a2a_endpoint,
    make_mcp_endpoint,
)
from .registry import ProtocolRegistry, protocol_registry

T = TypeVar("T")


def protocol_adapter(
    *,
    name: str,
    endpoint: Optional[ProtocolEndpoint] = None,
    protocol: str = "custom",
    transport: str = "in_process",
    url: Optional[str] = None,
    capabilities: Sequence[str] = (),
    auth_required: bool = False,
    registry: ProtocolRegistry = protocol_registry,
) -> Callable[[T], T]:
    """Register a callable or adapter class as a protocol adapter."""

    def decorator(obj: T) -> T:
        declared_endpoint = endpoint or ProtocolEndpoint(
            name=name,
            protocol=protocol,
            transport=transport,
            url=url,
            capabilities=list(capabilities),
            auth_required=auth_required,
        )

        def factory() -> ProtocolAdapter:
            if isinstance(obj, type) and issubclass(obj, ProtocolAdapter):
                return obj(declared_endpoint)  # type: ignore[misc, call-arg]
            if isinstance(obj, ProtocolAdapter):
                return obj
            if callable(obj):
                return FunctionProtocolAdapter(declared_endpoint, obj)  # type: ignore[arg-type]
            raise TypeError("protocol_adapter can only decorate a ProtocolAdapter class, instance, or callable")

        registry.register(declared_endpoint, factory)
        return obj

    return decorator


def mcp_adapter(
    *,
    name: str,
    transport: Any = "stdio",
    url: Optional[str] = None,
    capabilities: Sequence[str] = (),
    auth_required: bool = False,
    registry: ProtocolRegistry = protocol_registry,
) -> Callable[[T], T]:
    """Register a callable or adapter class as an MCP adapter."""
    return protocol_adapter(
        name=name,
        endpoint=make_mcp_endpoint(
            name,
            transport=transport,
            url=url,
            capabilities=capabilities,
            auth_required=auth_required,
        ),
        registry=registry,
    )


def a2a_adapter(
    *,
    name: str,
    transport: Any = "http",
    url: Optional[str] = None,
    capabilities: Sequence[str] = (),
    auth_required: bool = False,
    registry: ProtocolRegistry = protocol_registry,
) -> Callable[[T], T]:
    """Register a callable or adapter class as an A2A adapter."""
    return protocol_adapter(
        name=name,
        endpoint=make_a2a_endpoint(
            name,
            transport=transport,
            url=url,
            capabilities=capabilities,
            auth_required=auth_required,
        ),
        registry=registry,
    )
