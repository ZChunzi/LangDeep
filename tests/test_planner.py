"""Unit tests for Planner, LLMPlanGenerator, FallbackPlanGenerator, TemplateLoader."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile

from langchain_core.messages import AIMessage, HumanMessage

from langdeep.core.observability import MetricsCollector
from langdeep.core.orchestrator.planner import (
    Planner, LLMPlanGenerator, FallbackPlanGenerator,
    TemplateLoader, parse_plan_content, update_plan_status,
)
from langdeep.core.registry.model_registry import model_registry, ModelConfig
from langdeep.core.registry.agent_registry import agent_registry, AgentMetadata
from langdeep.core.errors import TemplateNotFoundError

from conftest import clean_registries, SmartMockLLM


def setup_function():
    clean_registries()
    model_registry.register("planner_model", ModelConfig(provider="mock", model_name="planner"))
    model_registry.set_model_instance("planner_model", SmartMockLLM(model_name="planner"))
    agent_registry.register("test_agent", lambda: object(), AgentMetadata(
        name="test_agent", description="test", capabilities=[], routing_keywords=[], model_name="gpt4o",
    ))


def test_fallback_plan_generator():
    gen = FallbackPlanGenerator()
    plan = gen.generate("do something", ["test_agent", "other"])
    assert len(plan) == 1
    assert plan[0]["agent"] == "test_agent"
    assert plan[0]["status"] == "pending"


def test_fallback_plan_generator_empty_agents():
    gen = FallbackPlanGenerator()
    plan = gen.generate("do something", [])
    assert plan[0]["agent"] == "default_agent"


def test_llm_plan_generator():
    gen = LLMPlanGenerator(model_name="planner_model")
    plan = gen.generate("do research", ["test_agent"])
    assert isinstance(plan, list)


def test_llm_plan_generator_records_model_metrics():
    metrics = MetricsCollector()
    gen = LLMPlanGenerator(model_name="planner_model", metrics_collector=metrics)

    plan = gen.generate("do research", ["test_agent"])

    assert isinstance(plan, list)
    collected = metrics.get_metrics()
    assert collected["counters"]["model.calls|component=planner,model=planner_model"] == 1
    assert (
        "model.duration_ms|component=planner,model=planner_model,status=success"
        in collected["histograms"]
    )


def test_planner_creates_plan():
    planner = Planner(model_name="planner_model")
    state = {
        "messages": [HumanMessage(content="research topic")],
    }
    result = planner.plan(state)
    assert "workflow_plan" in result
    assert isinstance(result["workflow_plan"], list)


def test_planner_reuses_existing_plan():
    planner = Planner(model_name="planner_model")
    existing = [{"id": "t1", "agent": "test_agent", "status": "pending"}]
    state = {
        "messages": [HumanMessage(content="research")],
        "workflow_plan": existing,
    }
    result = planner.plan(state)
    assert result["workflow_plan"] is existing


def test_parse_plan_content_json():
    plan = parse_plan_content('[{"id": "t1", "agent": "a", "status": "pending"}]')
    assert len(plan) == 1
    assert plan[0]["id"] == "t1"


def test_parse_plan_content_json_block():
    plan = parse_plan_content('```json\n[{"id": "t1", "agent": "a"}]\n```')
    assert len(plan) == 1


def test_parse_plan_content_with_tasks_key():
    plan = parse_plan_content('{"tasks": [{"id": "t1", "agent": "a", "status": "pending"}]}')
    assert len(plan) == 1


def test_parse_plan_content_invalid():
    plan = parse_plan_content("not json")
    assert isinstance(plan, list)


def test_update_plan_status():
    plan = [
        {"id": "t1", "status": "pending"},
        {"id": "t2", "status": "pending"},
    ]
    results = {"t1": {"success": True, "data": "ok"}}
    updated = update_plan_status(plan, results)
    assert updated[0]["status"] == "completed"
    assert updated[1]["status"] == "pending"


def test_template_loader_loads_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
        json.dump({"id": "test_tmpl", "steps": [
            {"id": "s1", "agent": "a", "depends_on": []},
        ]}, f)
        tmpl_path = os.path.dirname(f.name)

    loader = TemplateLoader(templates_dir=tmpl_path)
    plan = loader.apply("test_tmpl", "user input")
    assert len(plan) == 1
    assert plan[0]["id"] == "s1"
    os.unlink(f.name)


def test_template_loader_template_not_found():
    loader = TemplateLoader()
    try:
        loader.apply("ghost", "")
        assert False, "Should raise"
    except TemplateNotFoundError:
        pass


def test_template_loader_template_names():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False, encoding="utf-8") as f:
        f.write("id: my_wf\nsteps:\n  - id: step1\n")
        tmpl_path = os.path.dirname(f.name)

    loader = TemplateLoader(templates_dir=tmpl_path)
    names = loader.template_names
    assert "my_wf" in names
    os.unlink(f.name)


def test_template_loader_user_input_substitution():
    import tempfile, json
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
        json.dump({"id": "tmpl_var", "steps": [
            {"id": "s1", "config": {"query": "{{ user_input }}"}},
        ]}, f)
        tmpl_path = os.path.dirname(f.name)

    loader = TemplateLoader(templates_dir=tmpl_path)
    plan = loader.apply("tmpl_var", "hello world")
    assert plan[0]["config"]["query"] == "hello world"
    os.unlink(f.name)


# ── Edge case tests ─────────────────────────────────────────


def test_parse_plan_content_empty_string():
    """Empty string triggers fallback plan."""
    plan = parse_plan_content("")
    assert isinstance(plan, list)
    # fallback returns one pending task
    assert len(plan) >= 1
    assert plan[0].get("status") == "pending"


def test_parse_plan_content_wrong_structure():
    """Valid JSON but not a task list → fallback plan."""
    plan = parse_plan_content('{"not_tasks": []}')
    assert isinstance(plan, list)
    assert len(plan) >= 1
    assert plan[0].get("status") == "pending"


def test_parse_plan_content_nested_code_block():
    """Multiple code-block formatting still parses correctly."""
    plan = parse_plan_content('''text here
```json
[{"id": "t1", "agent": "a", "status": "pending"}]
```
more text''')
    assert len(plan) == 1
    assert plan[0]["id"] == "t1"


def test_template_loader_yaml_user_input():
    """YAML template with {{ user_input }} substitution."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False, encoding="utf-8") as f:
        f.write("id: yaml_var\nsteps:\n  - id: s1\n    config:\n      query: '{{ user_input }}'\n")
        tmpl_path = os.path.dirname(f.name)

    loader = TemplateLoader(templates_dir=tmpl_path)
    plan = loader.apply("yaml_var", "hello from yaml")
    assert plan[0]["config"]["query"] == "hello from yaml"
    os.unlink(f.name)


def test_planner_fallback_on_llm_exception():
    """When LLM model not found, Planner falls back to FallbackPlanGenerator."""
    clean_registries()  # no model registered
    planner = Planner(model_name="ghost_model")
    state = {
        "messages": [HumanMessage(content="do something")],
    }
    result = planner.plan(state)
    assert "workflow_plan" in result
    assert len(result["workflow_plan"]) >= 1
