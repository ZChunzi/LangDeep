"""Fuzz/property-based tests for core algorithmic functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from langdeep.core.orchestrator.executor import _clean_messages
from langdeep.core.orchestrator.aggregator import _split_results
from langdeep.core.planner.workflow_planner import WorkflowPlanner, WorkflowNode, NodeType
from langdeep.core.errors import CircularDependencyError

_SEED = 42


def _random_message(rng: random.Random, has_tool_call: bool = False) -> BaseMessage:
    """Generate a random message for fuzz testing."""
    kind = rng.randint(0, 5)
    content = rng.choice(["hello", "world", "test", "data", "", "a" * rng.randint(0, 20)])
    if kind == 0:
        return HumanMessage(content=content)
    elif kind == 1:
        return SystemMessage(content=content)
    elif kind == 2:
        return ToolMessage(content=content, tool_call_id=f"call_{rng.randint(1, 100)}")
    elif kind == 3:
        # AIMessage with content but no tool_calls
        return AIMessage(content=content)
    elif kind == 4:
        # AIMessage with tool_calls (and optionally content)
        tc = [{"name": "test_tool", "args": {"x": "1"}, "id": f"call_{rng.randint(1, 100)}"}]
        return AIMessage(content=content, tool_calls=tc)
    else:
        # AIMessage with empty content and empty/no tool_calls (noise)
        return AIMessage(content="")


# ── _clean_messages fuzz ────────────────────────────────────


def test_clean_messages_random_sequences():
    """Invariants of _clean_messages under random message sequences."""
    rng = random.Random(_SEED)

    for _ in range(200):
        n = rng.randint(0, 50)
        msgs = [_random_message(rng) for _ in range(n)]

        # Ensure at least one tool-call chain exists in some sequences
        if rng.random() < 0.3 and n > 2:
            idx = rng.randint(0, n - 2)
            msgs[idx] = AIMessage(content="", tool_calls=[
                {"name": "tool", "args": {}, "id": "fuzz_call"}])
            msgs[idx + 1] = ToolMessage(content="fuzz result", tool_call_id="fuzz_call")

        cleaned = _clean_messages(msgs)

        # Invariant 1: Output is capped at 80
        assert len(cleaned) <= 80

        # Invariant 2: All ToolMessages survive
        original_tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        cleaned_tool_msgs = [m for m in cleaned if isinstance(m, ToolMessage)]
        assert len(original_tool_msgs) == len(cleaned_tool_msgs)

        # Invariant 3: No empty AIMessage without tool_calls survives
        for m in cleaned:
            if isinstance(m, AIMessage):
                has_tc = bool(getattr(m, "tool_calls", None))
                has_content = bool(m.content)
                assert has_tc or has_content, "Empty AIMessage without tool_calls survived"

        # Invariant 4: All HumanMessages survive
        original_human = [m for m in msgs if isinstance(m, HumanMessage)]
        cleaned_human = [m for m in cleaned if isinstance(m, HumanMessage)]
        assert len(original_human) == len(cleaned_human)


# ── topological_sort fuzz ────────────────────────────────────


def _random_dag(rng: random.Random, num_nodes: int) -> list:
    """Generate a random DAG of WorkflowNodes."""
    if num_nodes == 0:
        return []
    nodes = []
    for i in range(num_nodes):
        nid = chr(ord("a") + i) if i < 26 else f"n{i}"
        # Depend only on earlier nodes (guarantees acyclicity)
        possible_deps = [chr(ord("a") + j) if j < 26 else f"n{j}" for j in range(i)]
        deps = []
        for d in possible_deps:
            if rng.random() < 0.3:
                deps.append(d)
        nodes.append(WorkflowNode(id=nid, type=NodeType.AGENT, name=nid, depends_on=deps))
    return nodes


def test_topological_sort_random_dags():
    """Verify topological_sort invariants on random DAGs."""
    rng = random.Random(_SEED)
    planner = WorkflowPlanner()

    for num_nodes in [0, 1, 2, 5, 10, 20]:
        for _ in range(50):
            nodes = _random_dag(rng, num_nodes)
            if not nodes:
                # Empty list
                groups = planner.topological_sort([])
                assert groups == []
                continue

            groups = planner.topological_sort(nodes)
            flat = [n.id for group in groups for n in group]

            # Invariant 1: All nodes appear in result
            assert len(flat) == len(nodes)
            assert set(flat) == {n.id for n in nodes}

            # Invariant 2: No node appears before its dependencies
            node_map = {n.id: n for n in nodes}
            for nid in flat:
                node = node_map[nid]
                my_idx = flat.index(nid)
                for dep in node.depends_on:
                    if dep in flat:
                        dep_idx = flat.index(dep)
                        assert dep_idx < my_idx, f"{nid} appears before dep {dep}"


def test_topological_sort_random_cycles():
    """Random graphs with cycles always raise CircularDependencyError."""
    rng = random.Random(_SEED + 1)
    planner = WorkflowPlanner()

    for _ in range(50):
        # Create a graph guaranteed to have a cycle
        if rng.random() < 0.5:
            # Simple 2-node cycle
            nodes = [
                WorkflowNode(id="x", type=NodeType.AGENT, name="x", depends_on=["y"]),
                WorkflowNode(id="y", type=NodeType.AGENT, name="y", depends_on=["x"]),
            ]
        else:
            # 3-node cycle: a→b→c→a
            nodes = [
                WorkflowNode(id="a", type=NodeType.AGENT, name="a", depends_on=["c"]),
                WorkflowNode(id="b", type=NodeType.AGENT, name="b", depends_on=["a"]),
                WorkflowNode(id="c", type=NodeType.AGENT, name="c", depends_on=["b"]),
            ]

        try:
            planner.topological_sort(nodes)
            assert False, "Should have raised CircularDependencyError"
        except CircularDependencyError:
            pass  # expected


# ── _split_results fuzz ─────────────────────────────────────


def test_split_results_random_strings():
    """_split_results partition invariants under random string inputs."""
    rng = random.Random(_SEED)

    for _ in range(200):
        n = rng.randint(0, 30)
        results = {}
        for i in range(n):
            key = f"k{i}"
            # Generate random string payload
            choice = rng.randint(0, 6)
            if choice == 0:
                value = ""  # falsy → failed
            elif choice == 1:
                value = "Agent " + "xyz"  # starts with "Agent" → failed
            elif choice == 2:
                value = "something error happened"  # contains "error" → failed
            elif choice == 3:
                value = "ERROR: timeout"  # contains "ERROR" → failed
            elif choice == 4:
                value = "AgentSmith"  # starts with Agent → failed
            elif choice == 5:
                value = "normal result"
            else:
                value = "x" * rng.randint(0, 50)  # random length string
            results[key] = value

        success, failed = _split_results(results)

        # Invariant 1: All keys appear in exactly one of success or failed
        all_keys = set(success.keys()) | set(failed.keys())
        assert all_keys == set(results.keys())

        # Invariant 2: No key appears in both
        overlap = set(success.keys()) & set(failed.keys())
        assert len(overlap) == 0

        # Invariant 3: Heuristic rules
        for key, value in results.items():
            if not value or "error" in value.lower() or value.startswith("Agent"):
                assert key in failed, f"Key {key}={value!r} should be in failed"
