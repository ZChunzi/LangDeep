"""Namespace-aware registry for protocol adapters."""

import copy
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..errors import ConfigurationError, ProtocolAdapterNotFoundError, ProtocolError
from ..logging import get_logger
from .models import (
    ProtocolAdapter,
    ProtocolAdapterFactory,
    ProtocolEndpoint,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolType,
    ensure_protocol_adapter,
    normalize_protocol_type,
)

logger = get_logger(__name__)


class ProtocolRegistry:
    """Registry for MCP, A2A, and custom protocol adapters."""

    _registries: Dict[str, "ProtocolRegistry"] = {}
    _class_lock = threading.Lock()

    def __new__(cls, namespace: str = "default"):
        namespace = namespace or "default"
        with cls._class_lock:
            if namespace not in cls._registries:
                instance = super().__new__(cls)
                instance._namespace = namespace
                instance._factories: Dict[Tuple[ProtocolType, str], ProtocolAdapterFactory] = {}
                instance._endpoints: Dict[Tuple[ProtocolType, str], ProtocolEndpoint] = {}
                instance._instances: Dict[Tuple[ProtocolType, str], ProtocolAdapter] = {}
                instance._registry_lock = threading.RLock()
                cls._registries[namespace] = instance
            return cls._registries[namespace]

    @classmethod
    def for_namespace(cls, namespace: str) -> "ProtocolRegistry":
        """Return an isolated registry for a namespace."""
        return cls(namespace=namespace)

    @property
    def namespace(self) -> str:
        return self._namespace

    def register(
        self,
        endpoint: ProtocolEndpoint,
        factory: ProtocolAdapterFactory,
        *,
        replace: bool = True,
    ) -> None:
        """Register a protocol adapter factory."""
        key = _key(endpoint.protocol, endpoint.name)
        with self._registry_lock:
            if key in self._factories and not replace:
                raise ConfigurationError(
                    f"Protocol adapter '{endpoint.name}' is already registered",
                    context={"protocol": endpoint.protocol.value, "namespace": self._namespace},
                )
            self._factories[key] = factory
            self._endpoints[key] = copy.deepcopy(endpoint)
            self._instances.pop(key, None)
        logger.info(
            "Protocol adapter registered",
            extra={"protocol": endpoint.protocol.value, "endpoint": endpoint.name, "namespace": self._namespace},
        )

    def register_adapter(self, adapter: ProtocolAdapter, *, replace: bool = True) -> None:
        """Register an existing adapter instance."""
        adapter = ensure_protocol_adapter(adapter)
        endpoint = copy.deepcopy(adapter.endpoint)
        key = _key(endpoint.protocol, endpoint.name)
        with self._registry_lock:
            if key in self._factories and not replace:
                raise ConfigurationError(
                    f"Protocol adapter '{endpoint.name}' is already registered",
                    context={"protocol": endpoint.protocol.value, "namespace": self._namespace},
                )
            self._factories[key] = lambda adapter=adapter: adapter
            self._endpoints[key] = endpoint
            self._instances[key] = adapter
        logger.info(
            "Protocol adapter instance registered",
            extra={"protocol": endpoint.protocol.value, "endpoint": endpoint.name, "namespace": self._namespace},
        )

    def get_adapter(self, protocol: Any, name: str) -> ProtocolAdapter:
        """Return a protocol adapter instance by protocol and endpoint name."""
        key = _key(protocol, name)
        with self._registry_lock:
            if key not in self._factories:
                raise ProtocolAdapterNotFoundError(
                    f"Protocol adapter '{key[1]}' is not registered",
                    context={
                        "protocol": key[0].value,
                        "available": [self._format_key(item) for item in self._factories.keys()],
                        "namespace": self._namespace,
                    },
                )
            if key not in self._instances:
                self._instances[key] = ensure_protocol_adapter(self._factories[key](), name=key[1])
            return self._instances[key]

    def get_endpoint(self, protocol: Any, name: str) -> ProtocolEndpoint:
        """Return a protocol endpoint declaration."""
        key = _key(protocol, name)
        with self._registry_lock:
            endpoint = self._endpoints.get(key)
        if endpoint is not None:
            return copy.deepcopy(endpoint)
        return copy.deepcopy(self.get_adapter(key[0], key[1]).endpoint)

    def list_endpoints(self, protocol: Optional[Any] = None) -> List[ProtocolEndpoint]:
        """List registered endpoint declarations."""
        protocol_type = normalize_protocol_type(protocol) if protocol is not None else None
        with self._registry_lock:
            endpoints = [
                endpoint
                for key, endpoint in self._endpoints.items()
                if protocol_type is None or key[0] == protocol_type
            ]
        return [copy.deepcopy(endpoint) for endpoint in endpoints]

    def route(self, request: ProtocolRequest) -> ProtocolAdapter:
        """Find an adapter that can handle a normalized request."""
        if request.endpoint and request.protocol:
            adapter = self.get_adapter(request.protocol, request.endpoint)
            if adapter.can_handle(request):
                return adapter
            raise ProtocolError(
                "Protocol adapter cannot handle requested operation",
                context={"endpoint": request.endpoint, "operation": request.operation},
            )

        with self._registry_lock:
            keys = list(self._factories.keys())
        for protocol, name in keys:
            if request.protocol and protocol != request.protocol:
                continue
            adapter = self.get_adapter(protocol, name)
            if adapter.can_handle(request):
                return adapter
        raise ProtocolAdapterNotFoundError(
            "No protocol adapter can handle the request",
            context={"operation": request.operation, "protocol": request.protocol.value if request.protocol else None},
        )

    def invoke(self, request: ProtocolRequest) -> ProtocolResponse:
        """Route and invoke a request synchronously."""
        return self.route(request).invoke(request)

    async def ainvoke(self, request: ProtocolRequest) -> ProtocolResponse:
        """Route and invoke a request asynchronously."""
        return await self.route(request).ainvoke(request)

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable registry snapshot."""
        with self._registry_lock:
            return {
                "namespace": self._namespace,
                "endpoints": [endpoint.as_dict() for endpoint in self._endpoints.values()],
            }

    def reset(self) -> None:
        """Clear registered protocol adapters."""
        with self._registry_lock:
            self._factories.clear()
            self._endpoints.clear()
            self._instances.clear()

    @staticmethod
    def _format_key(key: Tuple[ProtocolType, str]) -> str:
        return f"{key[0].value}:{key[1]}"


def _key(protocol: Any, name: str) -> Tuple[ProtocolType, str]:
    protocol_type = normalize_protocol_type(protocol)
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("Protocol adapter name must be a non-empty string")
    return protocol_type, name.strip()


protocol_registry = ProtocolRegistry()
