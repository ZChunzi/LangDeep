"""Agent node factory — creates LangGraph nodes for directly-routed agents."""

import asyncio
import time
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage

from ..agent_builder import ainvoke_agent_runnable, invoke_agent_runnable, is_async_only_agent
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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            instance = agent_registry.get_agent(agent_name)
            if is_async_only_agent(instance):
                return _agent_node_async(state)

        return _agent_node_sync(state)

    def _agent_node_sync(state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = state["messages"]
        if clean_messages_fn:
            msgs = clean_messages_fn(msgs)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                instance = agent_registry.get_agent(agent_name)
                resp = invoke_agent_runnable(instance, {
                    "messages": msgs,
                    "task_context": state.get("task_context", {}),
                })
                content, additional_kwargs = _extract(resp)
                logger.info(
                    "Direct agent call succeeded",
                    extra={"agent": agent_name, "attempt": attempt},
                )
                # Ensure content is never None (DeepSeek thinking mode requires non-null content)
                safe_content = content if content is not None else ""
                return {
                    "messages": [AIMessage(content=safe_content, additional_kwargs=additional_kwargs)],
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

    async def _agent_node_async(state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = state["messages"]
        if clean_messages_fn:
            msgs = clean_messages_fn(msgs)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                instance = agent_registry.get_agent(agent_name)
                resp = await ainvoke_agent_runnable(instance, {
                    "messages": msgs,
                    "task_context": state.get("task_context", {}),
                })
                content, additional_kwargs = _extract(resp)
                logger.info(
                    "Direct agent call succeeded",
                    extra={"agent": agent_name, "attempt": attempt},
                )
                safe_content = content if content is not None else ""
                return {
                    "messages": [AIMessage(content=safe_content, additional_kwargs=additional_kwargs)],
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
                    await asyncio.sleep(wait)

        err_msg = f"Agent {agent_name}: max retries ({max_retries}) exhausted. Last error: {last_error}"
        logger.error(err_msg)
        return {
            "messages": [AIMessage(content=err_msg)],
            "agent_results": {agent_name: err_msg},
        }

    return agent_node


def _extract(response: Any):
    """Extract (content, additional_kwargs) from agent response.

    Returns a tuple of (content_string, additional_kwargs_dict).
    additional_kwargs preserves metadata such as reasoning_content from
    providers like DeepSeek.
    """
    if isinstance(response, dict) and "messages" in response:
        for m in reversed(response["messages"]):
            if isinstance(m, AIMessage) and m.content:
                content = m.content
                if isinstance(content, list):
                    texts = [
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    ]
                    return ("".join(texts).strip(), dict(m.additional_kwargs))
                return (content, dict(m.additional_kwargs))
    if isinstance(response, str):
        return (response, {})
    return (str(response), {})
