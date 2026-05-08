"""Structured workflow plan schemas and validation helpers."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..core.errors import PlannerError

TaskStatus = Literal["pending", "running", "completed", "failed", "skipped", "waiting_confirmation"]


class WorkflowTask(BaseModel):
    id: str
    name: str = ""
    agent: str
    tools: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    parallel: bool = False
    priority: int = 0
    timeout: Optional[int] = None
    retry: Optional[int] = None
    requires_confirmation: bool = False
    input: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None
    status: TaskStatus = "pending"

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = self.id


class WorkflowPlan(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    tasks: List[WorkflowTask]

    @classmethod
    def from_raw_plan(cls, data: Any) -> "WorkflowPlan":
        if isinstance(data, list):
            return cls.model_validate({"tasks": data})
        if isinstance(data, dict):
            if "tasks" not in data and "steps" in data:
                data = {**data, "tasks": data["steps"]}
            return cls.model_validate(data)
        raise PlannerError("Workflow plan must be a list or object", context={"type": type(data).__name__})

    def to_task_dicts(self) -> List[Dict[str, Any]]:
        return [task.model_dump() for task in self.tasks]


def validate_workflow_plan(
    plan: Any,
    *,
    available_agents: Optional[List[str]] = None,
    available_tools: Optional[List[str]] = None,
) -> WorkflowPlan:
    """Validate structure, references, duplicate IDs, missing dependencies, and cycles."""
    try:
        workflow = WorkflowPlan.from_raw_plan(plan)
    except PlannerError:
        raise
    except Exception as exc:
        raise PlannerError("Workflow plan schema validation failed", cause=exc) from exc

    should_check_agents = available_agents is not None and len(available_agents) > 0
    should_check_tools = available_tools is not None
    agent_names = set(available_agents or [])
    tool_names = set(available_tools or [])
    task_ids = [task.id for task in workflow.tasks]

    duplicates = sorted({tid for tid in task_ids if task_ids.count(tid) > 1})
    if duplicates:
        raise PlannerError("Workflow task ids must be unique", context={"duplicates": duplicates})

    known_tasks = set(task_ids)
    for task in workflow.tasks:
        if should_check_agents and task.agent not in agent_names:
            raise PlannerError(
                f"Workflow task references unknown agent '{task.agent}'",
                context={"task_id": task.id, "available_agents": sorted(agent_names)},
            )
        missing_tools = [tool for tool in task.tools if should_check_tools and tool not in tool_names]
        if missing_tools:
            raise PlannerError(
                "Workflow task references unknown tools",
                context={"task_id": task.id, "missing_tools": missing_tools},
            )
        missing_deps = [dep for dep in task.depends_on if dep not in known_tasks]
        if missing_deps:
            raise PlannerError(
                "Workflow task references missing dependencies",
                context={"task_id": task.id, "missing_dependencies": missing_deps},
            )

    _assert_acyclic(workflow.tasks)
    return workflow


def _assert_acyclic(tasks: List[WorkflowTask]) -> None:
    deps = {task.id: set(task.depends_on) for task in tasks}
    visiting = set()
    visited = set()

    def visit(task_id: str, stack: List[str]) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            cycle = stack[stack.index(task_id):] + [task_id] if task_id in stack else stack + [task_id]
            raise PlannerError("Workflow plan contains a dependency cycle", context={"cycle": cycle})
        visiting.add(task_id)
        for dep in deps.get(task_id, set()):
            visit(dep, stack + [dep])
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in deps:
        visit(task_id, [task_id])
