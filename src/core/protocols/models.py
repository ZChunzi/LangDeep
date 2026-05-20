"""Protocol-neutral contracts for MCP, A2A, and custom integrations."""

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

from ..errors import ConfigurationError, ProtocolError

PROTOCOL_SCHEMA_VERSION = "langdeep.protocol.v1"


class ProtocolType(str, Enum):
    """Supported protocol families."""

    MCP = "mcp"
    A2A = "a2a"
    CUSTOM = "custom"


class ProtocolTransport(str, Enum):
    """Common transport hints used by protocol adapters."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    WEBSOCKET = "websocket"
    IN_PROCESS = "in_process"


@dataclass
class ProtocolEndpoint:
    """Serializable declaration for an external protocol endpoint."""

    name: str
    protocol: ProtocolType
    transport: ProtocolTransport = ProtocolTransport.IN_PROCESS
    url: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    auth_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _require_text(self.schema_version, "schema_version")
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported protocol endpoint schema '{self.schema_version}'",
                context={"schema_version": self.schema_version, "supported": PROTOCOL_SCHEMA_VERSION},
            )
        self.name = _require_text(self.name, "name")
        self.protocol = normalize_protocol_type(self.protocol)
        self.transport = normalize_transport(self.transport)
        self.capabilities = _normalize_text_list(self.capabilities, "capabilities")
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolEndpoint":
        """Build an endpoint declaration from a dictionary."""
        return cls(
            schema_version=data.get("schema_version", PROTOCOL_SCHEMA_VERSION),
            name=data.get("name", ""),
            protocol=data.get("protocol", ""),
            transport=data.get("transport", ProtocolTransport.IN_PROCESS.value),
            url=data.get("url"),
            capabilities=list(data.get("capabilities") or []),
            auth_required=bool(data.get("auth_required", False)),
            metadata=dict(data.get("metadata") or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable endpoint representation."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "protocol": self.protocol.value,
            "transport": self.transport.value,
            "url": self.url,
            "capabilities": list(self.capabilities),
            "auth_required": self.auth_required,
            "metadata": dict(self.metadata),
        }


@dataclass
class ProtocolRequest:
    """A normalized request routed through a protocol adapter."""

    operation: str
    payload: Dict[str, Any] = field(default_factory=dict)
    protocol: Optional[ProtocolType] = None
    endpoint: Optional[str] = None
    request_id: Optional[str] = None
    tenant_id: str = "default"
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.operation = _require_text(self.operation, "operation")
        if self.protocol is not None:
            self.protocol = normalize_protocol_type(self.protocol)
        if self.endpoint is not None:
            self.endpoint = _require_text(self.endpoint, "endpoint")
        self.payload = dict(self.payload or {})
        self.headers = {str(key): str(value) for key, value in dict(self.headers or {}).items()}
        self.metadata = dict(self.metadata or {})

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable request representation."""
        return {
            "operation": self.operation,
            "payload": dict(self.payload),
            "protocol": self.protocol.value if self.protocol else None,
            "endpoint": self.endpoint,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "headers": dict(self.headers),
            "metadata": dict(self.metadata),
        }


@dataclass
class ProtocolResponse:
    """A normalized response from a protocol adapter."""

    ok: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status_code: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        payload: Optional[Dict[str, Any]] = None,
        *,
        status_code: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ProtocolResponse":
        """Create a successful response."""
        return cls(ok=True, payload=dict(payload or {}), status_code=status_code, metadata=dict(metadata or {}))

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        status_code: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ProtocolResponse":
        """Create an error response."""
        return cls(ok=False, error=error, status_code=status_code, metadata=dict(metadata or {}))

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable response representation."""
        return {
            "ok": self.ok,
            "payload": dict(self.payload),
            "error": self.error,
            "status_code": self.status_code,
            "metadata": dict(self.metadata),
        }


class ProtocolAdapter:
    """Base class for low-coupling protocol adapters."""

    endpoint: ProtocolEndpoint

    def __init__(self, endpoint: ProtocolEndpoint):
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return self.endpoint.name

    @property
    def protocol(self) -> ProtocolType:
        return self.endpoint.protocol

    def can_handle(self, request: ProtocolRequest) -> bool:
        """Return whether this adapter can handle a normalized request."""
        if request.endpoint and request.endpoint != self.endpoint.name:
            return False
        if request.protocol and request.protocol != self.endpoint.protocol:
            return False
        return not self.endpoint.capabilities or request.operation in self.endpoint.capabilities

    def invoke(self, request: ProtocolRequest) -> ProtocolResponse:
        """Handle a request synchronously."""
        raise NotImplementedError

    async def ainvoke(self, request: ProtocolRequest) -> ProtocolResponse:
        """Handle a request asynchronously."""
        result = self.invoke(request)
        if inspect.isawaitable(result):
            return await result
        return result


ProtocolHandler = Callable[[ProtocolRequest], ProtocolResponse]
AsyncProtocolHandler = Callable[[ProtocolRequest], Awaitable[ProtocolResponse]]
ProtocolAdapterFactory = Callable[[], ProtocolAdapter]


class FunctionProtocolAdapter(ProtocolAdapter):
    """Adapter backed by a plain Python callable."""

    def __init__(
        self,
        endpoint: ProtocolEndpoint,
        handler: Callable[[ProtocolRequest], Any],
    ):
        super().__init__(endpoint)
        self._handler = handler

    def invoke(self, request: ProtocolRequest) -> ProtocolResponse:
        result = self._handler(request)
        return normalize_protocol_response(result)

    async def ainvoke(self, request: ProtocolRequest) -> ProtocolResponse:
        result = self._handler(request)
        if inspect.isawaitable(result):
            result = await result
        return normalize_protocol_response(result)


def normalize_protocol_response(value: Any) -> ProtocolResponse:
    """Normalize adapter return values to ``ProtocolResponse``."""
    if isinstance(value, ProtocolResponse):
        return value
    if isinstance(value, dict):
        return ProtocolResponse.success(value)
    raise ProtocolError(
        "Protocol adapter returned an invalid response",
        context={"expected": "ProtocolResponse or dict", "actual": type(value).__name__},
    )


def normalize_protocol_type(value: Any) -> ProtocolType:
    """Normalize protocol aliases and enum values."""
    if isinstance(value, ProtocolType):
        return value
    try:
        return ProtocolType(str(value).strip().lower())
    except ValueError as exc:
        raise ConfigurationError(
            f"Unsupported protocol type '{value}'",
            context={"supported": [item.value for item in ProtocolType]},
            cause=exc,
        )


def normalize_transport(value: Any) -> ProtocolTransport:
    """Normalize transport aliases and enum values."""
    if isinstance(value, ProtocolTransport):
        return value
    try:
        return ProtocolTransport(str(value).strip().lower())
    except ValueError as exc:
        raise ConfigurationError(
            f"Unsupported protocol transport '{value}'",
            context={"supported": [item.value for item in ProtocolTransport]},
            cause=exc,
        )


def make_mcp_endpoint(
    name: str,
    *,
    transport: Any = ProtocolTransport.STDIO,
    url: Optional[str] = None,
    capabilities: Sequence[str] = (),
    auth_required: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProtocolEndpoint:
    """Create an MCP endpoint declaration."""
    return ProtocolEndpoint(
        name=name,
        protocol=ProtocolType.MCP,
        transport=transport,
        url=url,
        capabilities=list(capabilities),
        auth_required=auth_required,
        metadata=dict(metadata or {}),
    )


def make_a2a_endpoint(
    name: str,
    *,
    transport: Any = ProtocolTransport.HTTP,
    url: Optional[str] = None,
    capabilities: Sequence[str] = (),
    auth_required: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProtocolEndpoint:
    """Create an A2A endpoint declaration."""
    return ProtocolEndpoint(
        name=name,
        protocol=ProtocolType.A2A,
        transport=transport,
        url=url,
        capabilities=list(capabilities),
        auth_required=auth_required,
        metadata=dict(metadata or {}),
    )


def ensure_protocol_adapter(value: Any, *, name: Optional[str] = None) -> ProtocolAdapter:
    """Validate and return a protocol adapter instance."""
    if not isinstance(value, ProtocolAdapter):
        raise ProtocolError(
            "Protocol adapter factory returned an invalid object",
            context={"expected": "ProtocolAdapter", "actual": type(value).__name__, "name": name},
        )
    if name and value.endpoint.name != name:
        raise ConfigurationError(
            "Protocol adapter endpoint name does not match registry key",
            context={"registry_name": name, "endpoint_name": value.endpoint.name},
        )
    return value


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Protocol field '{field_name}' must be a non-empty string")
    return value.strip()


def _normalize_text_list(values: Iterable[Any], field_name: str) -> List[str]:
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                f"Protocol field '{field_name}' must contain only non-empty strings",
                context={"value": value},
            )
        result.append(value.strip())
    return result
