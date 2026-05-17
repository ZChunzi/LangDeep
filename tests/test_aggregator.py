"""Unit tests for Aggregator, LLMMerger, ConcatMerger."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage

from langdeep.core.orchestrator.aggregator import (
    Aggregator, LLMMerger, ConcatMerger, ResultMerger, _split_results, _no_results_fallback,
)
from langdeep.core.observability import MetricsCollector
from langdeep.core.registry.model_registry import model_registry, ModelConfig

from conftest import clean_registries, SmartMockLLM


def setup_function():
    clean_registries()
    model_registry.register("test_model", ModelConfig(provider="mock", model_name="test"))
    model_registry.set_model_instance("test_model", SmartMockLLM(model_name="test"))


def test_concat_merger():
    merger = ConcatMerger()
    result = merger.merge("user request", {"a": "result a", "b": "result b"})
    assert "[a]" in result
    assert "[b]" in result
    assert "result a" in result


def test_concat_merger_single():
    merger = ConcatMerger()
    result = merger.merge("req", {"x": "single"})
    assert "single" in result


def test_llm_merger():
    merger = LLMMerger(model_name="test_model")
    result = merger.merge("user request", {"a": "data from a"})
    assert result is not None
    assert len(result) > 0


def test_llm_merger_records_model_metrics():
    metrics = MetricsCollector()
    merger = LLMMerger(model_name="test_model", metrics_collector=metrics)

    result = merger.merge("user request", {"a": "data from a"})

    assert result is not None
    collected = metrics.get_metrics()
    assert collected["counters"]["model.calls|component=aggregator,model=test_model"] == 1
    assert (
        "model.duration_ms|component=aggregator,model=test_model,status=success"
        in collected["histograms"]
    )


def test_aggregate_single_result_returns_directly():
    agg = Aggregator(model_name="test_model")
    state = {
        "messages": [HumanMessage(content="hi")],
        "agent_results": {"only_one": "direct result"},
    }
    result = agg.aggregate(state)
    assert "direct result" in result["messages"][0].content


def test_aggregate_no_results():
    agg = Aggregator(model_name="test_model")
    state = {
        "messages": [HumanMessage(content="hi")],
        "agent_results": {},
    }
    result = agg.aggregate(state)
    assert "No results" in result["messages"][0].content


def test_aggregate_failed_results_fallback():
    agg = Aggregator(model_name="test_model")
    state = {
        "messages": [HumanMessage(content="hi")],
        "agent_results": {"fail": {"success": False, "data": "", "error": "Agent something broke"}},
    }
    result = agg.aggregate(state)
    assert "Errors occurred" in result["messages"][0].content


def test_aggregate_multi_merges():
    agg = Aggregator(model_name="test_model")
    state = {
        "messages": [HumanMessage(content="combine these")],
        "agent_results": {"a": "alpha", "b": "beta"},
    }
    result = agg.aggregate(state)
    assert result["aggregation_done"] is True
    assert result["messages"][0].content is not None


def test_aggregate_failed_excluded():
    agg = Aggregator(model_name="test_model")
    state = {
        "messages": [HumanMessage(content="hi")],
        "agent_results": {"good": "valid result", "bad": "Agent error occurred"},
    }
    result = agg.aggregate(state)
    # Should use LLM merger for the single good result or multi
    assert result["messages"][0].content is not None


def test_split_results():
    results = {
        "a": "valid data",
        "b": {"success": False, "data": "", "error": "Agent something broke"},
        "c": {"success": False, "data": "", "error": "error occurred"},
        "d": "good result",
    }
    success, failed = _split_results(results)
    assert "a" in success
    assert "d" in success
    assert "b" in failed
    assert "c" in failed


def test_split_results_structured_executor_results():
    results = {
        "ok": {"success": True, "data": "structured answer", "error": ""},
        "failed": {"success": False, "data": "", "error": "tool failed"},
        "waiting": {
            "success": False,
            "data": "",
            "error": "",
            "status": "waiting_confirmation",
            "reason": "tool_requires_confirmation:delete_file",
        },
        "skipped": {"success": False, "data": "", "error": "Dependency unsatisfied", "status": "skipped"},
    }
    success, failed = _split_results(results)
    assert success == {"ok": "structured answer"}
    assert failed["failed"] == "tool failed"
    assert "waiting_confirmation" in failed["waiting"]
    assert "Dependency unsatisfied" in failed["skipped"]


def test_split_results_extracts_structured_message_result():
    results = {
        "agent": {"messages": [HumanMessage(content="hi"), AIMessage(content="final answer")]},
    }
    success, failed = _split_results(results)
    assert success == {"agent": "final answer"}
    assert failed == {}


def test_no_results_fallback_with_ai_message():
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="previous reply"),
        ],
    }
    result = _no_results_fallback(state, {})
    assert "previous reply" in result


def test_no_results_fallback_with_failed():
    result = _no_results_fallback({}, {"t1": "error"})
    assert "Errors occurred" in result


def test_aggregate_sets_aggregation_done():
    agg = Aggregator(model_name="test_model")
    state = {
        "messages": [HumanMessage(content="hi")],
        "agent_results": {"x": "y"},
    }
    result = agg.aggregate(state)
    assert result["aggregation_done"] is True


# ── Edge case tests ─────────────────────────────────────────


def test_split_results_edge_strings():
    """Plain strings are treated as successful payloads unless they are empty."""
    results = {
        "a": "error_handler_module",   # contains "error" but is legitimate
        "b": "clean_result",
        "c": "AgentSmith_result",      # starts with "Agent" but is legitimate
        "d": "",                        # falsy → failed
    }
    success, failed = _split_results(results)
    assert "a" in success
    assert "b" in success
    assert "c" in success
    assert "d" in failed


def test_split_results_normal():
    """Clean strings are all classified as success."""
    results = {
        "a": "valid data",
        "b": "more valid data",
    }
    success, failed = _split_results(results)
    assert "a" in success
    assert "b" in success
    assert len(failed) == 0


def test_llmmerger_model_not_found():
    """When the model is not registered, LLMMerger falls back to concatenation."""
    clean_registries()  # no model registered
    merger = LLMMerger(model_name="ghost_model")
    result = merger.merge("hello", {"a": "result_a", "b": "result_b"})
    assert "result_a" in result
    assert "result_b" in result


def test_aggregator_custom_merger_injection():
    """A custom ResultMerger can be injected and is used by Aggregator."""

    class UpperMerger(ResultMerger):
        def merge(self, user_request, agent_results):
            return "; ".join(v.upper() for v in agent_results.values())

    agg = Aggregator(model_name="test_model", merger=UpperMerger())
    state = {
        "messages": [HumanMessage(content="hi")],
        "agent_results": {"a": "hello", "b": "world"},
    }
    result = agg.aggregate(state)
    assert "HELLO" in result["messages"][0].content
    assert "WORLD" in result["messages"][0].content


def test_aggregator_merger_fallback_to_concat():
    """When the merger raises, Aggregator falls back to ConcatMerger."""
    import tempfile, json, os

    class BrokenMerger(ResultMerger):
        def merge(self, user_request, agent_results):
            raise RuntimeError("merge failed")

    agg = Aggregator(model_name="test_model", merger=BrokenMerger())
    state = {
        "messages": [HumanMessage(content="combine")],
        "agent_results": {"a": "alpha", "b": "beta"},
    }
    result = agg.aggregate(state)
    # fallback concat should produce section headers
    assert "[a]" in result["messages"][0].content
    assert "[b]" in result["messages"][0].content


def test_aggregate_large_result_dict():
    """Aggregation handles 50+ results without issue."""
    agg = Aggregator(model_name="test_model")
    agent_results = {f"task_{i}": f"result_{i}" for i in range(60)}
    state = {
        "messages": [HumanMessage(content="big")],
        "agent_results": agent_results,
    }
    result = agg.aggregate(state)
    assert result["aggregation_done"] is True
    assert result["messages"][0].content is not None
