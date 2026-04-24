"""Supervisor routing — keyword fast-path and LLM fallback."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool

from ..logging import get_logger
from ..errors import RoutingError
from ..registry.agent_registry import agent_registry

logger = get_logger(__name__)


# ── Routing strategy interface ───────────────────────────────────────────────────

class RoutingStrategy(ABC):
    """Pluggable routing strategy.

    Subclass and pass to ``FlowOrchestrator(routing_strategy=...)`` to
    change how the supervisor picks the next node.
    """

    @abstractmethod
    def route(self, user_input: str, available_agents: List[Dict[str, Any]]) -> Optional[str]:
        """Return an agent/node name or None to fall through to the next strategy."""
        ...


class KeywordRoutingStrategy(RoutingStrategy):
    """Default keyword-based fast router — no LLM call, near-zero latency."""

    def route(self, user_input: str, available_agents: List[Dict[str, Any]]) -> Optional[str]:
        lower_input = user_input.lower()
        for agent_info in available_agents:
            name = agent_info["name"]
            meta = agent_registry.get_metadata(name)
            keywords = meta.routing_keywords if meta else []
            if any(kw.lower() in lower_input for kw in keywords):
                logger.info("Keyword routing hit", extra={"agent_name": name})
                return name
        return None


class DefaultRouter:
    """Two-tier routing: keyword strategy first, then LLM fallback.

    Parameters:
        model_name: Name of the registered model used for LLM routing.
        routing_strategy: Optional custom RoutingStrategy.
        valid_targets: Pre-resolved list of valid destination node names.
    """

    def __init__(
        self,
        model_name: str,
        routing_strategy: Optional[RoutingStrategy] = None,
        valid_targets: Optional[List[str]] = None,
    ):
        self._model_name = model_name
        self._strategy = routing_strategy or KeywordRoutingStrategy()
        self._valid_targets = valid_targets or []
        self._routing_tool = self._build_tool()

    def set_valid_targets(self, targets: List[str]) -> None:
        self._valid_targets = targets

    @staticmethod
    def _build_tool():
        @lc_tool
        def route_to_node(next_node: str) -> str:
            """Route to the next node in the workflow."""
            return f"Routing to {next_node}"
        return route_to_node

    def route(
        self,
        state: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute routing and return a state update dict with ``next`` key."""
        from ..registry.model_registry import model_registry

        last_human = _last_human_input(state["messages"])

        # 1. Fast keyword path
        fast = self._strategy.route(last_human, available_agents)
        if fast:
            logger.info("Fast-route result", extra={"next_node": fast})
            return {"messages": [], "next": fast}

        # 2. LLM fallback
        return self._llm_route(last_human, available_agents, model_registry)

    def _llm_route(
        self,
        user_input: str,
        available_agents: List[Dict[str, Any]],
        model_registry,
    ) -> Dict[str, Any]:
        agent_list = "\n".join(
            f"- {a['name']}: {a['description']}" for a in available_agents
        )
        targets_str = ", ".join(self._valid_targets)

        system_msg = SystemMessage(content=(
            f"Call route_to_node to select the next step.\n"
            f"Agents:\n{agent_list}\n"
            f"Valid targets: {targets_str}\n"
            f"Rule: simple tasks → best agent; complex multi-step tasks → planner."
        ))

        llm = model_registry.get_model(self._model_name)
        llm_with_tools = llm.bind_tools([self._routing_tool], tool_choice="required")

        try:
            response = llm_with_tools.invoke(
                [system_msg, HumanMessage(content=user_input)]
            )
            next_node = _parse_tool_call(response, self._valid_targets)
        except Exception as exc:
            logger.error("LLM routing call failed", extra={"error": str(exc)})
            next_node = "end"

        logger.info("LLM routing result", extra={"next_node": next_node})
        return {"messages": [], "next": next_node}


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _last_human_input(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""


def _parse_tool_call(response: Any, valid_targets: List[str]) -> str:
    if hasattr(response, "tool_calls") and response.tool_calls:
        next_node = response.tool_calls[0].get("args", {}).get("next_node", "")
        if next_node in valid_targets:
            return next_node
    if hasattr(response, "content") and response.content:
        content = str(response.content).lower()
        for target in valid_targets:
            if target in content:
                return target
    logger.warning("Could not parse routing decision; defaulting to end")
    return "end"
