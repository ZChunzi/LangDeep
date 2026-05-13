"""Supervisor routing — keyword fast-path and LLM fallback."""

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool

from ..logging import get_logger
from ..errors import RoutingError
from ..observability.metrics import MetricsCollector
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
    """Default keyword-based fast router — no LLM call, near-zero latency.

    Uses word-boundary matching for ASCII keywords and minimum-length
    guards for CJK to avoid spurious substring matches (e.g. ``code``
    in ``encode``).
    """

    def route(self, user_input: str, available_agents: List[Dict[str, Any]]) -> Optional[str]:
        lower_input = user_input.lower()
        for agent_info in available_agents:
            name = agent_info["name"]
            meta = agent_registry.get_metadata(name)
            keywords = meta.routing_keywords if meta else []
            for kw in keywords:
                kw_lower = kw.lower()
                # Exact match always wins
                if kw_lower == lower_input.strip():
                    logger.info("Keyword routing hit (exact)", extra={"agent_name": name, "keyword": kw})
                    return name
                if self._match_keyword(lower_input, kw_lower):
                    logger.info("Keyword routing hit", extra={"agent_name": name, "keyword": kw})
                    return name
        return None

    @staticmethod
    def _match_keyword(text: str, keyword: str) -> bool:
        """Check if *keyword* appears as a meaningful token inside *text*."""
        # Empty guard
        if not keyword or len(keyword) < 2:
            return False

        # ASCII → word-boundary regex match
        if all(ord(c) < 0x2000 for c in keyword):
            pattern = re.escape(keyword)
            return bool(re.search(rf'(?<![a-zA-Z]){pattern}(?![a-zA-Z])', text))

        # CJK / non-ASCII → require minimum length to reduce accidental matches
        if len(keyword) >= 2:
            return keyword in text

        return False


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
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self._model_name = model_name
        self._strategy = routing_strategy or KeywordRoutingStrategy()
        self._valid_targets = valid_targets or []
        self._routing_tool = self._build_tool()
        self._metrics = metrics_collector

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

        started = time.monotonic()
        if self._metrics is not None:
            self._metrics.counter("routing.requests")
        last_human = _last_human_input(state["messages"])

        # 1. Fast keyword path
        fast = self._strategy.route(last_human, available_agents)
        if fast:
            logger.info("Fast-route result", extra={"next_node": fast})
            self._record_route(started, path="fast", next_node=fast, status="success")
            return {"messages": [], "next": fast}

        # 2. LLM fallback
        result = self._llm_route(last_human, available_agents, model_registry)
        self._record_route(started, path="llm", next_node=result.get("next", "end"), status="success")
        return result

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

        tool_prompt = (
            "You are a routing agent. Call 'route_to_node' with the correct 'next_node'.\n"
            f"Agents:\n{agent_list}\n"
            f"Valid targets: {targets_str}\n"
            "Rule: simple tasks → best agent; complex multi-step tasks → planner.\n"
            "Respond ONLY by calling route_to_node."
        )

        llm = model_registry.get_model(self._model_name)
        llm_with_tools = llm.bind_tools([self._routing_tool])

        model_started = time.monotonic()
        model_status = "success"
        if self._metrics is not None:
            self._metrics.counter(
                "model.calls",
                tags={"component": "router", "model": self._model_name},
            )

        try:
            response = llm_with_tools.invoke(
                [SystemMessage(content=tool_prompt), HumanMessage(content=user_input)]
            )
            next_node = _parse_tool_call(response, self._valid_targets)
        except Exception as exc:
            model_status = "failure"
            logger.error("LLM routing call failed", extra={"error": str(exc)})
            next_node = "end"

        # Fallback: if tool-based routing failed to find a target and a
        # plain-text response was given, try a second call without tools.
        if next_node == "end" and available_agents:
            text_prompt = (
                "Choose the single best agent for this request.\n"
                f"Agents:\n{agent_list}\n\n"
                "Reply with ONLY the agent name, nothing else."
            )
            try:
                response2 = llm.invoke(
                    [SystemMessage(content=text_prompt), HumanMessage(content=user_input)]
                )
                fallback = _parse_text_routing(response2, self._valid_targets)
                if fallback:
                    logger.info("Fallback text routing succeeded",
                                extra={"next_node": fallback})
                    next_node = fallback
            except Exception:
                model_status = "failure"
                logger.warning("Fallback text routing also failed")

        if self._metrics is not None:
            self._metrics.histogram(
                "model.duration_ms",
                (time.monotonic() - model_started) * 1000,
                tags={
                    "component": "router",
                    "model": self._model_name,
                    "status": model_status,
                },
            )
            if model_status == "failure":
                self._metrics.counter(
                    "model.errors",
                    tags={"component": "router", "model": self._model_name},
                )

        logger.info("LLM routing result", extra={"next_node": next_node})
        return {"messages": [], "next": next_node}

    def _record_route(self, started: float, *, path: str, next_node: str, status: str) -> None:
        if self._metrics is None:
            return
        tags = {"path": path, "next": next_node, "status": status}
        self._metrics.counter("routing.decisions", tags=tags)
        self._metrics.histogram(
            "routing.duration_ms",
            (time.monotonic() - started) * 1000,
            tags=tags,
        )
        if path == "fast":
            self._metrics.counter("routing.fast_path_hits", tags={"next": next_node})


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _last_human_input(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""


def _parse_tool_call(response: Any, valid_targets: List[str]) -> str:
    # 1. Structured tool_call (OpenAI-compatible format)
    if hasattr(response, "tool_calls") and response.tool_calls:
        next_node = response.tool_calls[0].get("args", {}).get("next_node", "")
        if next_node in valid_targets:
            return next_node

    # 2. Content-based match (including reasoning_content from DeepSeek thinking mode)
    content = _collect_response_text(response)
    if content:
        lower = content.lower()
        for target in valid_targets:
            if target in lower:
                return target
        # Fuzzy match: weather_agent → "weather agent"
        for target in valid_targets:
            target_words = target.replace("_", " ").replace("-", " ")
            if target_words in lower:
                return target

    logger.warning("Could not parse routing decision; defaulting to end")
    return "end"


def _parse_text_routing(response: Any, valid_targets: List[str]) -> Optional[str]:
    """Parse a plain-text LLM response to extract a valid target name."""
    text = _collect_response_text(response)
    if not text:
        return None
    lower = text.lower().strip().rstrip(".。!！")
    # Direct match
    if lower in valid_targets:
        return lower
    # Fuzzy match via underscore/space
    for target in valid_targets:
        if target in lower:
            return target
        target_words = target.replace("_", " ").replace("-", " ")
        if target_words in lower:
            return target
    return None


def _collect_response_text(response: Any) -> str:
    """Collect all text from content + additional_kwargs (reasoning_content)."""
    parts = []
    if hasattr(response, "content") and response.content:
        parts.append(str(response.content))
    if hasattr(response, "additional_kwargs") and response.additional_kwargs.get("reasoning_content"):
        parts.append(response.additional_kwargs["reasoning_content"])
    return " ".join(parts)
