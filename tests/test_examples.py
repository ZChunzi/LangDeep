"""Smoke tests for runnable examples."""

import importlib.util
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
