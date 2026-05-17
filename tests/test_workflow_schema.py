"""WorkflowPlan schema validation tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langdeep.schemas import (
    EXECUTABLE_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    WORKFLOW_TASK_STATUSES,
    is_executable_task_status,
    status_from_execution_result,
    validate_workflow_plan,
)
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
    assert plan.tasks[1].status == "pending"


def test_workflow_task_status_sets_are_explicit():
    assert WORKFLOW_TASK_STATUSES == (
        "pending",
        "running",
        "completed",
        "failed",
        "skipped",
        "waiting_confirmation",
    )
    assert EXECUTABLE_TASK_STATUSES == {"pending", "running"}
    assert TERMINAL_TASK_STATUSES == {"completed", "failed", "skipped", "waiting_confirmation"}


def test_validate_workflow_plan_invalid_status():
    try:
        validate_workflow_plan([{"id": "t1", "agent": "a", "status": "queued"}])
        assert False, "Should raise"
    except PlannerError as exc:
        assert "schema validation failed" in str(exc)


def test_validate_workflow_plan_missing_status_strict_mode():
    try:
        validate_workflow_plan([{"id": "t1", "agent": "a"}], require_status=True)
        assert False, "Should raise"
    except PlannerError as exc:
        assert "missing required status" in str(exc)


def test_validate_workflow_plan_missing_required_field():
    try:
        validate_workflow_plan([{"id": "t1", "status": "pending"}])
        assert False, "Should raise"
    except PlannerError as exc:
        assert "schema validation failed" in str(exc)


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


def test_validate_workflow_plan_can_skip_dependency_checks_for_runtime_recovery():
    plan = validate_workflow_plan(
        [
            {"id": "t1", "agent": "a", "depends_on": ["ghost"], "status": "pending"},
        ],
        check_dependencies=False,
    )
    assert plan.tasks[0].depends_on == ["ghost"]


def test_status_helpers():
    assert is_executable_task_status("pending") is True
    assert is_executable_task_status("running") is True
    assert is_executable_task_status("completed") is False
    assert is_executable_task_status(None) is True
    assert status_from_execution_result({"success": True, "data": "ok"}) == "completed"
    assert status_from_execution_result({"success": False, "status": "skipped"}) == "skipped"
    assert status_from_execution_result({"success": False, "status": "waiting_confirmation"}) == "waiting_confirmation"
    assert status_from_execution_result({"success": False, "error": "bad"}) == "failed"
