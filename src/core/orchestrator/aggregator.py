"""Result aggregation — merges multi-agent outputs into a single coherent response."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from ..logging import get_logger
from ..errors import AggregatorError

logger = get_logger(__name__)


# ── Extension point ──────────────────────────────────────────────────────────────

class ResultMerger(ABC):
    """Pluggable result merger — override to change aggregation behaviour."""

    @abstractmethod
    def merge(
        self,
        user_request: str,
        agent_results: Dict[str, str],
    ) -> str:
        ...


class LLMMerger(ResultMerger):
    """Uses an LLM to synthesise multiple agent outputs into one answer."""

    def __init__(self, model_name: str, prompt_loader=None):
        self._model_name = model_name
        self._prompt_loader = prompt_loader

    def merge(self, user_request: str, agent_results: Dict[str, str]) -> str:
        from ..registry.model_registry import model_registry

        try:
            if self._prompt_loader:
                aggregator_prompt = self._prompt_loader.load_prompt("aggregator")
                prompt_msgs = aggregator_prompt.format_messages(
                    user_request=user_request,
                    agent_results=agent_results,
                )
            else:
                raise RuntimeError("No prompt loader configured")
        except Exception:
            import json
            prompt_msgs = [HumanMessage(content=(
                f"Synthesise results for: '{user_request}'\n"
                f"{json.dumps(agent_results, ensure_ascii=False)}"
            ))]

        try:
            llm = model_registry.get_model(self._model_name)
            response = llm.invoke(prompt_msgs)
            return str(response.content)
        except Exception as exc:
            logger.error("LLM aggregation failed", extra={"error": str(exc)})
            return "\n\n".join(str(v) for v in agent_results.values())


class ConcatMerger(ResultMerger):
    """Simple concatenation — no LLM call, just joins results with newlines."""

    def merge(self, user_request: str, agent_results: Dict[str, str]) -> str:
        return "\n\n".join(
            f"[{name}]\n{text}" for name, text in agent_results.items()
        )


# ── Aggregator node logic ────────────────────────────────────────────────────────

class Aggregator:
    """Result aggregation node.

    Parameters:
        model_name: Registered model used for LLM synthesis.
        merger: Optional custom ResultMerger.
        prompt_loader: Optional prompt loader for the aggregator prompt.
    """

    def __init__(
        self,
        model_name: str,
        merger: Optional[ResultMerger] = None,
        prompt_loader=None,
    ):
        self._merger = merger or LLMMerger(model_name, prompt_loader)
        self._concat = ConcatMerger()

    def aggregate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge agent_results into a final response."""
        agent_results: Dict[str, str] = state.get("agent_results", {})

        success_results, failed_results = _split_results(agent_results)

        if failed_results:
            logger.warning(
                "Some tasks failed, excluded from aggregation",
                extra={"failed_tasks": list(failed_results.keys())},
            )

        # Single result — return directly
        if len(success_results) == 1:
            answer = next(iter(success_results.values()))
            return {"messages": [AIMessage(content=answer)], "aggregation_done": True}

        # No success results
        if not success_results:
            answer = _no_results_fallback(state, failed_results)
            return {"messages": [AIMessage(content=answer)], "aggregation_done": True}

        # Multiple results — merge
        user_request = _last_human(state["messages"])
        try:
            answer = self._merger.merge(user_request, success_results)
        except Exception as exc:
            logger.error("Merger failed, falling back to concat", extra={"error": str(exc)})
            answer = self._concat.merge(user_request, success_results)

        return {"messages": [AIMessage(content=answer)], "aggregation_done": True}


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _split_results(results: Dict[str, str]) -> tuple:
    success = {}
    failed = {}
    for k, v in results.items():
        if v and not str(v).startswith("Agent") and "error" not in str(v).lower():
            success[k] = v
        else:
            failed[k] = v
    return success, failed


def _no_results_fallback(state: Dict[str, Any], failed: Dict[str, str]) -> str:
    if failed:
        return "Errors occurred during execution. Please try again later."
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return str(m.content)
    return "No results available. Please try again."


def _last_human(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""
