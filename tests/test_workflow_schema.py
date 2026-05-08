"""WorkflowPlan schema validation tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langdeep.schemas import validate_workflow_plan
from langdeep.core.errors import PlannerError


def test_validate_workflow_plan_success():
    plan = validate_workflow_plan(
        [
            {"id": "t1", "agent": "a", "tools": ["search"], "status": "pending"},
            {"id": "t2", "agent": "a", "depends_on": ["t1"]},
        ],
        available_agents=["a"],
        available_tools=["search"],
    )
    assert [task.id for task in plan.tasks] == ["t1", "t2"]


def test_validate_workflow_plan_unknown_agent():
    try:
        validate_workflow_plan([{"id": "t1", "agent": "ghost"}], available_agents=["a"])
        assert False, "Should raise"
    except PlannerError as exc:
        assert "unknown agent" in str(exc)


def test_validate_workflow_plan_duplicate_ids():
    try:
        validate_workflow_plan([
            {"id": "t1", "agent": "a"},
            {"id": "t1", "agent": "a"},
        ])
        assert False, "Should raise"
    except PlannerError as exc:
        assert "unique" in str(exc)


def test_validate_workflow_plan_cycle():
    try:
        validate_workflow_plan([
            {"id": "t1", "agent": "a", "depends_on": ["t2"]},
            {"id": "t2", "agent": "a", "depends_on": ["t1"]},
        ])
        assert False, "Should raise"
    except PlannerError as exc:
        assert "cycle" in str(exc)
