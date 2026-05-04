"""Unit tests for ExecutionPolicy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langdeep.core.execution.execution_policy import ExecutionPolicy
from langdeep.core.errors import InvalidPolicyError


def test_default_policy():
    p = ExecutionPolicy()
    assert p.max_concurrency == 5
    assert p.strategy == "gather"
    assert p.retry_on == []


def test_valid_strategies():
    for s in ("gather", "sequential", "priority_queue"):
        p = ExecutionPolicy(strategy=s)
        assert p.strategy == s


def test_invalid_strategy():
    try:
        ExecutionPolicy(strategy="invalid_strat")
        assert False, "Should raise"
    except InvalidPolicyError:
        pass


def test_max_concurrency_validation():
    try:
        ExecutionPolicy(max_concurrency=0)
        assert False, "Should raise"
    except InvalidPolicyError:
        pass
    # Valid
    p = ExecutionPolicy(max_concurrency=1)
    assert p.max_concurrency == 1


def test_from_dict():
    p = ExecutionPolicy.from_dict({"max_concurrency": 3, "strategy": "sequential", "retry_on": ["TimeoutError"]})
    assert p.max_concurrency == 3
    assert p.strategy == "sequential"
    assert "TimeoutError" in p.retry_on


def test_from_dict_ignores_extra_keys():
    p = ExecutionPolicy.from_dict({"max_concurrency": 2, "extra_key": "ignored"})
    assert p.max_concurrency == 2


def test_to_dict_roundtrip():
    p1 = ExecutionPolicy(max_concurrency=4, strategy="priority_queue", retry_on=["ValueError"])
    d = p1.to_dict()
    assert d["max_concurrency"] == 4
    assert d["strategy"] == "priority_queue"
    assert d["retry_on"] == ["ValueError"]

    p2 = ExecutionPolicy.from_dict(d)
    assert p2.max_concurrency == p1.max_concurrency
    assert p2.strategy == p1.strategy
    assert p2.retry_on == p1.retry_on


def test_retry_on_list():
    p = ExecutionPolicy(retry_on=["TimeoutError", "ConnectionError"])
    assert len(p.retry_on) == 2
