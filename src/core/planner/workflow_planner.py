"""Workflow planner with YAML support."""
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import yaml
import json


class NodeType(Enum):
    """Node type"""
    AGENT = "agent"          # Agent node
    TOOL = "tool"            # Tool node
    CONDITION = "condition"  # Conditional branch
    PARALLEL = "parallel"    # Parallel execution
    LOOP = "loop"            # Loop
    HUMAN = "human"          # Human intervention


@dataclass
class WorkflowNode:
    """Workflow node definition"""
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    retry: int = 0
    timeout: int = 60


class WorkflowPlanner:
    """
    Workflow planner

    Supports two planning modes:
    1. Declarative: Load predefined workflows from YAML/JSON files
    2. Dynamic: Generate execution plans dynamically by LLM based on user requests
    """

    def __init__(self, workflow_dir: str = "workflows"):
        self.workflow_dir = workflow_dir
        self._workflows: Dict[str, List[WorkflowNode]] = {}

    def load_workflow(self, name: str) -> List[WorkflowNode]:
        """Load workflow definition from file"""
        if name in self._workflows:
            return self._workflows[name]

        workflow_file = f"{self.workflow_dir}/{name}.yaml"
        with open(workflow_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        nodes = []
        for node_data in data.get("nodes", []):
            node = WorkflowNode(
                id=node_data["id"],
                type=NodeType(node_data["type"]),
                name=node_data["name"],
                config=node_data.get("config", {}),
                depends_on=node_data.get("depends_on", []),
                condition=node_data.get("condition"),
                retry=node_data.get("retry", 0),
                timeout=node_data.get("timeout", 60)
            )
            nodes.append(node)

        self._workflows[name] = nodes
        return nodes

    def plan_dynamically(
        self,
        user_request: str,
        available_agents: List[str],
        llm_model: str
    ) -> List[WorkflowNode]:
        """
        Dynamic planning: Generate execution plan by LLM based on user request
        """
        from ..registry.model_registry import model_registry
        from ..prompt.prompt_loader import prompt_loader

        llm = model_registry.get_model(llm_model)
        planner_prompt = prompt_loader.load_prompt("dynamic_planner")

        response = llm.invoke(
            planner_prompt.format_messages(
                user_request=user_request,
                available_agents=available_agents
            )
        )

        # Parse LLM-generated plan
        plan = self._parse_llm_plan(response.content)
        return plan

    def _parse_llm_plan(self, content: str) -> List[WorkflowNode]:
        """Parse LLM-generated plan (expects JSON format)"""
        try:
            data = json.loads(content)
            nodes = []
            for node_data in data.get("steps", []):
                node = WorkflowNode(
                    id=node_data["id"],
                    type=NodeType(node_data["type"]),
                    name=node_data["name"],
                    config=node_data.get("config", {}),
                    depends_on=node_data.get("depends_on", []),
                    condition=node_data.get("condition")
                )
                nodes.append(node)
            return nodes
        except json.JSONDecodeError:
            # Fallback handling
            return []

    def topological_sort(self, nodes: List[WorkflowNode]) -> List[List[WorkflowNode]]:
        """
        Topological sort - determine execution order and identify parallel executable node groups
        """
        # Build dependency graph
        graph = {node.id: set(node.depends_on) for node in nodes}

        # Kahn's algorithm for topological sort
        result = []
        remaining = set(node.id for node in nodes)

        while remaining:
            # Find all nodes with zero in-degree (can execute in parallel)
            ready = [
                node for node in nodes
                if node.id in remaining and not graph[node.id].intersection(remaining)
            ]

            if not ready:
                # Cycle detected
                raise ValueError("Workflow has circular dependency")

            result.append(ready)
            for node in ready:
                remaining.remove(node.id)

        return result

    def estimate_duration(self, nodes: List[WorkflowNode]) -> int:
        """Estimate workflow execution time"""
        sorted_groups = self.topological_sort(nodes)
        total_time = 0

        for group in sorted_groups:
            # Parallel group takes maximum time
            group_time = max(node.timeout for node in group)
            total_time += group_time

        return total_time