"""Low-coupling protocol integration contracts for MCP, A2A, and custom adapters."""

from .decorators import a2a_adapter, mcp_adapter, protocol_adapter
from .models import (
    AsyncProtocolHandler,
    FunctionProtocolAdapter,
    ProtocolAdapter,
    ProtocolAdapterFactory,
    ProtocolEndpoint,
    ProtocolHandler,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolTransport,
    ProtocolType,
    make_a2a_endpoint,
    make_mcp_endpoint,
)
from .registry import ProtocolRegistry, protocol_registry

__all__ = [
    "AsyncProtocolHandler",
    "FunctionProtocolAdapter",
    "ProtocolAdapter",
    "ProtocolAdapterFactory",
    "ProtocolEndpoint",
    "ProtocolHandler",
    "ProtocolRegistry",
    "ProtocolRequest",
    "ProtocolResponse",
    "ProtocolTransport",
    "ProtocolType",
    "a2a_adapter",
    "make_a2a_endpoint",
    "make_mcp_endpoint",
    "mcp_adapter",
    "protocol_adapter",
    "protocol_registry",
]
