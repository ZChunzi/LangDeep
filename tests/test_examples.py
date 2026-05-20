"""Smoke tests for runnable examples."""

import importlib.util
import sys
import types
from pathlib import Path

from conftest import clean_registries


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_basic_mock_agent_example_runs(capsys):
    clean_registries()
    example_path = Path(__file__).resolve().parents[1] / "examples" / "basic_mock_agent.py"
    module = load_module(example_path)

    try:
        module.main()
        output = capsys.readouterr().out
    finally:
        clean_registries()

    assert "Question: How is the weather in Beijing?" in output
    assert "Beijing: sunny, 25C" in output


def test_customer_support_agent_example_runs(capsys):
    clean_registries()
    example_path = Path(__file__).resolve().parents[1] / "examples" / "customer_support_agent.py"
    module = load_module(example_path)

    try:
        module.main()
        output = capsys.readouterr().out
    finally:
        clean_registries()

    assert "Customer question: Can I get a refund for an unused order?" in output
    assert "Matched topic: refund" in output
    assert "Policy: Refund requests are accepted within 30 days" in output


def test_fastapi_server_example_routes_without_fastapi_dependency(monkeypatch):
    clean_registries()

    class FakeHTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class FakeFastAPI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.routes = {}
            self.events = {}

        def get(self, path, **kwargs):
            return self._route("GET", path, kwargs)

        def post(self, path, **kwargs):
            return self._route("POST", path, kwargs)

        def on_event(self, event):
            def decorator(func):
                self.events[event] = func
                return func

            return decorator

        def _route(self, method, path, kwargs):
            def decorator(func):
                self.routes[(method, path)] = {"func": func, "kwargs": kwargs}
                return func

            return decorator

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.HTTPException = FakeHTTPException
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)

    example_path = Path(__file__).resolve().parents[1] / "examples" / "fastapi_server.py"
    module = load_module(example_path)

    try:
        assert ("GET", "/health") in module.app.routes
        assert ("POST", "/chat") in module.app.routes
        assert "startup" in module.app.events

        module.startup()
        health = module.health()
        assert health.status == "ok"

        response = module.chat(module.ChatRequest(message="hello from HTTP", session_id="demo"))
        assert response.session_id == "demo"
        assert "Mock LangDeep response over HTTP." in response.reply
        assert "User message: hello from HTTP" in response.reply
    finally:
        clean_registries()


def test_customer_service_demo_runs(capsys):
    clean_registries()
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "customer_service_demo"
        / "demo.py"
    )
    module = load_module(example_path)

    try:
        module.main()
        output = capsys.readouterr().out
    finally:
        clean_registries()

    assert "Conversation" in output
    assert "Order LD-1001" in output
    assert "Return case CASE-0001 has been opened for LD-1001." in output
    assert "Memory entries: " in output
    assert "create_return_case: success=True, confirmed=True, blocked=False" in output
    assert "customer_service.tool.calls" in output
