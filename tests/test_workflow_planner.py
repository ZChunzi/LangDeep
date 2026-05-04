"""Unit tests for WorkflowPlanner, WorkflowNode, topological sort."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import json

from langdeep.core.planner.workflow_planner import (
    WorkflowPlanner, WorkflowNode, NodeType,
)
from langdeep.core.errors import TemplateNotFoundError, CircularDependencyError


def _write_workflow(dirpath: str, name: str, data: dict, ext: str = ".json"):
    path = os.path.join(dirpath, f"{name}{ext}")
    with open(path, "w", encoding="utf-8") as f:
        if ext == ".json":
            json.dump(data, f)
        else:
            import yaml
            yaml.dump(data, f)
    return path


def test_load_workflow_json():
    with tempfile.TemporaryDirectory() as tmp:
        _write_workflow(tmp, "my_wf", {
            "id": "my_wf",
            "steps": [
                {"id": "s1", "type": "agent", "name": "search", "depends_on": []},
            ],
        })
        planner = WorkflowPlanner(workflow_dir=tmp)
        nodes = planner.load_workflow("my_wf")
        assert len(nodes) == 1
        assert nodes[0].id == "s1"
        assert nodes[0].type == NodeType.AGENT


def test_load_workflow_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        _write_workflow(tmp, "yaml_wf", {
            "id": "yaml_wf",
            "steps": [{"id": "y1", "type": "agent", "name": "processor"}],
        }, ext=".yaml")
        planner = WorkflowPlanner(workflow_dir=tmp)
        nodes = planner.load_workflow("yaml_wf")
        assert len(nodes) == 1
        assert nodes[0].id == "y1"


def test_load_workflow_not_found():
    planner = WorkflowPlanner(workflow_dir="/tmp/nonexistent")
    try:
        planner.load_workflow("ghost")
        assert False, "Should raise"
    except TemplateNotFoundError:
        pass


def test_caching():
    with tempfile.TemporaryDirectory() as tmp:
        _write_workflow(tmp, "cached", {"id": "cached", "steps": [{"id": "c1", "type": "agent"}]})
        planner = WorkflowPlanner(workflow_dir=tmp)
        n1 = planner.load_workflow("cached")
        n2 = planner.load_workflow("cached")
        assert n1 is n2


def test_list_templates():
    with tempfile.TemporaryDirectory() as tmp:
        _write_workflow(tmp, "wf_a", {"id": "wf_a", "steps": []})
        _write_workflow(tmp, "wf_b", {"id": "wf_b", "steps": []}, ext=".yaml")
        planner = WorkflowPlanner(workflow_dir=tmp)
        templates = planner.list_templates()
        assert "wf_a" in templates
        assert "wf_b" in templates


def test_list_templates_no_dir():
    planner = WorkflowPlanner(workflow_dir="/tmp/__nonexistent_dir__")
    assert planner.list_templates() == []


def test_to_plan_dicts():
    nodes = [
        WorkflowNode(id="n1", type=NodeType.AGENT, name="search"),
        WorkflowNode(id="n2", type=NodeType.CUSTOM, name="transform",
                     config={"tools": ["x"]}, depends_on=["n1"]),
    ]
    planner = WorkflowPlanner()
    plan = planner.to_plan_dicts(nodes)
    assert len(plan) == 2
    assert plan[0]["id"] == "n1"
    assert plan[0]["agent"] == "search"
    assert plan[1]["id"] == "n2"
    assert plan[1]["node"] == "transform"


def test_to_plan_dicts_user_input():
    nodes = [
        WorkflowNode(id="n1", type=NodeType.AGENT, name="echo",
                     config={"tools": ["search"]}),
        WorkflowNode(id="n2", type=NodeType.CUSTOM, name="transform",
                     config={"query": "{{ user_input }}"}),
    ]
    planner = WorkflowPlanner()
    plan = planner.to_plan_dicts(nodes, user_input="hello world")
    assert plan[0]["agent"] == "echo"
    assert plan[0]["tools"] == ["search"]
    # CUSTOM nodes preserve config with user_input substitution
    assert "hello world" in plan[1]["config"]["query"]


def test_topological_sort():
    nodes = [
        WorkflowNode(id="a", type=NodeType.AGENT, name="a"),
        WorkflowNode(id="b", type=NodeType.AGENT, name="b", depends_on=["a"]),
        WorkflowNode(id="c", type=NodeType.AGENT, name="c", depends_on=["a"]),
        WorkflowNode(id="d", type=NodeType.AGENT, name="d", depends_on=["b", "c"]),
    ]
    planner = WorkflowPlanner()
    groups = planner.topological_sort(nodes)
    # a must be first, b/c second, d last
    assert groups[0][0].id == "a"
    assert groups[-1][0].id == "d"


def test_topological_sort_circular():
    nodes = [
        WorkflowNode(id="x", type=NodeType.AGENT, name="x", depends_on=["y"]),
        WorkflowNode(id="y", type=NodeType.AGENT, name="y", depends_on=["x"]),
    ]
    planner = WorkflowPlanner()
    try:
        planner.topological_sort(nodes)
        assert False, "Should raise"
    except CircularDependencyError:
        pass


def test_estimate_duration():
    nodes = [
        WorkflowNode(id="a", type=NodeType.AGENT, name="a", timeout=10),
        WorkflowNode(id="b", type=NodeType.AGENT, name="b", depends_on=["a"], timeout=20),
        WorkflowNode(id="c", type=NodeType.AGENT, name="c", depends_on=["a"], timeout=30),
    ]
    planner = WorkflowPlanner()
    duration = planner.estimate_duration(nodes)
    # group 1: a (10s), group 2: max(b=20, c=30) = 30s → total 40s
    assert duration == 40


def test_all_node_types():
    for nt in NodeType:
        node = WorkflowNode(id="t1", type=nt, name=nt.value)
        assert node.type == nt


def test_parse_nodes_from_data():
    planner = WorkflowPlanner()
    data = {
        "nodes": [
            {"id": "n1", "type": "agent", "name": "agent1", "depends_on": [], "priority": 5},
            {"id": "n2", "type": "condition", "name": "check", "condition": "x > 0"},
        ],
    }
    nodes = planner._parse_nodes(data)
    assert len(nodes) == 2
    assert nodes[0].priority == 5
    assert nodes[1].condition == "x > 0"
