import pytest

from langdeep.core.errors import ProtocolAdapterNotFoundError
from langdeep.core.protocols import (
    FunctionProtocolAdapter,
    ProtocolAdapter,
    ProtocolEndpoint,
    ProtocolRegistry,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolTransport,
    ProtocolType,
    a2a_adapter,
    make_a2a_endpoint,
    make_mcp_endpoint,
    mcp_adapter,
)


def test_endpoint_round_trip_and_helpers():
    endpoint = make_mcp_endpoint(
        "filesystem",
        capabilities=["list_tools"],
        metadata={"package": "mcp-server-filesystem"},
    )

    data = endpoint.as_dict()
    restored = ProtocolEndpoint.from_dict(data)

    assert restored.protocol is ProtocolType.MCP
    assert restored.transport is ProtocolTransport.STDIO
    assert restored.capabilities == ["list_tools"]
    assert restored.metadata["package"] == "mcp-server-filesystem"

    a2a = make_a2a_endpoint("support-agent", url="https://agent.example/a2a")
    assert a2a.protocol is ProtocolType.A2A
    assert a2a.transport is ProtocolTransport.HTTP


def test_registry_routes_mcp_adapter_by_operation():
    registry = ProtocolRegistry.for_namespace("protocol-test-route")
    registry.reset()

    adapter = FunctionProtocolAdapter(
        make_mcp_endpoint("tools", capabilities=["list_tools"]),
        lambda request: {"operation": request.operation, "tools": ["search"]},
    )
    registry.register_adapter(adapter)

    response = registry.invoke(ProtocolRequest(protocol="mcp", operation="list_tools"))

    assert response.ok is True
    assert response.payload == {"operation": "list_tools", "tools": ["search"]}
    assert registry.snapshot()["endpoints"][0]["protocol"] == "mcp"


def test_registry_honors_namespace_isolation():
    left = ProtocolRegistry.for_namespace("protocol-left")
    right = ProtocolRegistry.for_namespace("protocol-right")
    left.reset()
    right.reset()

    left.register_adapter(FunctionProtocolAdapter(make_a2a_endpoint("agent"), lambda request: {"ok": True}))

    assert [endpoint.name for endpoint in left.list_endpoints()] == ["agent"]
    assert right.list_endpoints() == []
    with pytest.raises(ProtocolAdapterNotFoundError):
        right.invoke(ProtocolRequest(protocol="a2a", endpoint="agent", operation="message/send"))


@pytest.mark.asyncio
async def test_async_callable_adapter():
    registry = ProtocolRegistry.for_namespace("protocol-async")
    registry.reset()

    async def handler(request):
        return ProtocolResponse.success({"tenant": request.tenant_id})

    registry.register_adapter(FunctionProtocolAdapter(make_a2a_endpoint("remote"), handler))

    response = await registry.ainvoke(
        ProtocolRequest(protocol=ProtocolType.A2A, endpoint="remote", operation="message/send", tenant_id="acme")
    )

    assert response.ok is True
    assert response.payload == {"tenant": "acme"}


def test_decorators_register_mcp_and_a2a_callables():
    registry = ProtocolRegistry.for_namespace("protocol-decorators")
    registry.reset()

    @mcp_adapter(name="mcp-tools", capabilities=["tools/list"], registry=registry)
    def list_tools(request):
        return {"tools": [request.operation]}

    @a2a_adapter(name="assistant", capabilities=["message/send"], registry=registry)
    def send_message(request):
        return ProtocolResponse.success({"received": request.payload["text"]})

    assert list_tools.__name__ == "list_tools"
    assert send_message.__name__ == "send_message"
    assert registry.invoke(ProtocolRequest(protocol="mcp", operation="tools/list")).payload == {
        "tools": ["tools/list"]
    }
    assert registry.invoke(
        ProtocolRequest(protocol="a2a", endpoint="assistant", operation="message/send", payload={"text": "hi"})
    ).payload == {"received": "hi"}


def test_decorator_can_register_adapter_class():
    registry = ProtocolRegistry.for_namespace("protocol-class")
    registry.reset()

    @mcp_adapter(name="class-adapter", capabilities=["ping"], registry=registry)
    class PingAdapter(ProtocolAdapter):
        def invoke(self, request):
            return ProtocolResponse.success({"pong": request.operation})

    assert PingAdapter.__name__ == "PingAdapter"
    assert registry.invoke(ProtocolRequest(protocol="mcp", endpoint="class-adapter", operation="ping")).payload == {
        "pong": "ping"
    }
