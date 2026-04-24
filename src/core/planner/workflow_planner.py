"""Declarative workflow planner — loads plans from YAML/JSON and bridges to orchestrator."""

import json
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..logging import get_logger
from ..errors import TemplateNotFoundError, CircularDependencyError

logger = get_logger(__name__)


class NodeType(Enum):
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    HUMAN = "human"
    CUSTOM = "custom"


@dataclass
class WorkflowNode:
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    retry: int = 0
    timeout: int = 60
    priority: int = 0


class WorkflowPlanner:
    """Loads predefined workflows and can bridge them to the orchestrator."""

    def __init__(self, workflow_dir: str = "workflows"):
        self._workflow_dir = workflow_dir
        self._workflows: Dict[str, List[WorkflowNode]] = {}

    def load_workflow(self, name: str) -> List[WorkflowNode]:
        if name in self._workflows:
            return self._workflows[name]

        base = Path(self._workflow_dir)
        for ext in (".yaml", ".yml", ".json"):
            wf_file = base / f"{name}{ext}"
            if wf_file.exists():
                with open(wf_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f) if ext != ".json" else json.load(f)
                nodes = self._parse_nodes(data)
                self._workflows[name] = nodes
                logger.info("Workflow loaded", extra={"name": name, "file": str(wf_file), "node_count": len(nodes)})
                return nodes

        raise TemplateNotFoundError(
            f"Workflow '{name}' not found",
            context={"search_dir": self._workflow_dir, "tried_extensions": [".yaml", ".yml", ".json"]},
        )

    def list_templates(self) -> List[str]:
        base = Path(self._workflow_dir)
        if not base.is_dir():
            return []
        templates = []
        for ext in ("*.yaml", "*.yml", "*.json"):
            for fpath in sorted(base.glob(ext)):
                templates.append(fpath.stem)
        return templates

    def to_plan_dicts(self, nodes: List[WorkflowNode], user_input: str = "") -> List[Dict[str, Any]]:
        plan = []
        for node in nodes:
            task = {
                "id": node.id,
                "name": node.name,
                "depends_on": node.depends_on,
                "status": "pending",
                "priority": node.priority,
            }
            if node.type == NodeType.CUSTOM:
                task["node"] = node.name
                task["config"] = node.config
            else:
                task["agent"] = node.name
                task["tools"] = node.config.get("tools", [])
            for key, value in list(task.items()):
                if isinstance(value, str) and "{{ user_input }}" in value:
                    task[key] = value.replace("{{ user_input }}", user_input)
            plan.append(task)
        return plan

    def topological_sort(self, nodes: List[WorkflowNode]) -> List[List[WorkflowNode]]:
        graph = {node.id: set(node.depends_on) for node in nodes}
        result = []
        remaining = {node.id for node in nodes}
        while remaining:
            ready = [
                node for node in nodes
                if node.id in remaining and not graph[node.id].intersection(remaining)
            ]
            if not ready:
                raise CircularDependencyError(
                    "Workflow contains circular dependencies",
                    context={"remaining_nodes": list(remaining)},
                )
            result.append(ready)
            for node in ready:
                remaining.remove(node.id)
        return result

    def estimate_duration(self, nodes: List[WorkflowNode]) -> int:
        groups = self.topological_sort(nodes)
        return sum(max(node.timeout for node in group) for group in groups)

    def _parse_nodes(self, data: dict) -> List[WorkflowNode]:
        nodes = []
        for nd in data.get("steps", data.get("nodes", [])):
            nodes.append(WorkflowNode(
                id=nd["id"],
                type=NodeType(nd.get("type", "agent")),
                name=nd.get("name", nd["id"]),
                config=nd.get("config", {}),
                depends_on=nd.get("depends_on", []),
                condition=nd.get("condition"),
                retry=nd.get("retry", 0),
                timeout=nd.get("timeout", 60),
                priority=nd.get("priority", 0),
            ))
        return nodes
