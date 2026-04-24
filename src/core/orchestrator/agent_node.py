"""Agent node factory — creates LangGraph nodes for directly-routed agents."""

import time
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage

from ..logging import get_logger
from ..registry.agent_registry import agent_registry

logger = get_logger(__name__)


def make_agent_node(
    agent_name: str,
    max_retries: int = 3,
    clean_messages_fn: Callable = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a LangGraph node function for the named agent with retry logic."""

    def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = state["messages"]
        if clean_messages_fn:
            msgs = clean_messages_fn(msgs)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                instance = agent_registry.get_agent(agent_name)
                resp = instance.invoke({
                    "messages": msgs,
                    "task_context": state.get("task_context", {}),
                })
                content = _extract(resp)
                logger.info(
                    "Direct agent call succeeded",
                    extra={"agent": agent_name, "attempt": attempt},
                )
                return {
                    "messages": [AIMessage(content=content)],
                    "agent_results": {agent_name: content},
                }
            except Exception as exc:
                last_error = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Direct agent call failed",
                    extra={"agent": agent_name, "attempt": attempt, "error": str(exc)},
                )
                if attempt < max_retries:
                    time.sleep(wait)

        err_msg = f"Agent {agent_name}: max retries ({max_retries}) exhausted. Last error: {last_error}"
        logger.error(err_msg)
        return {
            "messages": [AIMessage(content=err_msg)],
            "agent_results": {agent_name: err_msg},
        }

    return agent_node


def _extract(response: Any) -> str:
    if isinstance(response, dict) and "messages" in response:
        for m in reversed(response["messages"]):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                return m.content
        for m in reversed(response["messages"]):
            if isinstance(m, AIMessage) and m.content:
                return m.content
    if isinstance(response, str):
        return response
    return str(response)
