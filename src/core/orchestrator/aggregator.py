"""Result aggregation — merges multi-agent outputs into a single coherent response."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..logging import get_logger
from ..errors import AggregatorError
from ..observability.metrics import MetricsCollector

logger = get_logger(__name__)


@dataclass(frozen=True)
class AggregationInput:
    """Normalized agent result consumed by result mergers."""

    name: str
    success: bool
    text: str
    status: str = "completed"


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

    def __init__(
        self,
        model_name: str,
        prompt_loader=None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self._model_name = model_name
        self._prompt_loader = prompt_loader
        self._metrics = metrics_collector

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

        started = time.monotonic()
        status = "success"
        if self._metrics is not None:
            self._metrics.counter(
                "model.calls",
                tags={"component": "aggregator", "model": self._model_name},
            )
        try:
            llm = model_registry.get_model(self._model_name)
            response = llm.invoke(prompt_msgs)
            return str(response.content)
        except Exception as exc:
            status = "failure"
            logger.error("LLM aggregation failed", extra={"error": str(exc)})
            return "\n\n".join(str(v) for v in agent_results.values())
        finally:
            if self._metrics is not None:
                self._metrics.histogram(
                    "model.duration_ms",
                    (time.monotonic() - started) * 1000,
                    tags={
                        "component": "aggregator",
                        "model": self._model_name,
                        "status": status,
                    },
                )
                if status == "failure":
                    self._metrics.counter(
                        "model.errors",
                        tags={"component": "aggregator", "model": self._model_name},
                    )


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
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self._metrics = metrics_collector
        self._merger = merger or LLMMerger(
            model_name,
            prompt_loader,
            metrics_collector=metrics_collector,
        )
        self._concat = ConcatMerger()

    def aggregate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge agent_results into a final response."""
        started = time.monotonic()
        if self._metrics is not None:
            self._metrics.counter("aggregation.requests")
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
            self._record_aggregation(started, "single", success_results, failed_results)
            return {"messages": [AIMessage(content=answer)], "aggregation_done": True}

        # No success results
        if not success_results:
            answer = _no_results_fallback(state, failed_results)
            self._record_aggregation(started, "no_success", success_results, failed_results)
            return {"messages": [AIMessage(content=answer)], "aggregation_done": True}

        # Multiple results — merge
        user_request = _last_human(state["messages"])
        try:
            answer = self._merger.merge(user_request, success_results)
            status = "merged"
        except Exception as exc:
            logger.error("Merger failed, falling back to concat", extra={"error": str(exc)})
            answer = self._concat.merge(user_request, success_results)
            status = "fallback"

        self._record_aggregation(started, status, success_results, failed_results)
        return {"messages": [AIMessage(content=answer)], "aggregation_done": True}

    def _record_aggregation(
        self,
        started: float,
        status: str,
        success_results: Dict[str, str],
        failed_results: Dict[str, str],
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.counter("aggregation.results", tags={"status": status})
        self._metrics.histogram(
            "aggregation.duration_ms",
            (time.monotonic() - started) * 1000,
            tags={"status": status},
        )
        self._metrics.histogram("aggregation.success_count", float(len(success_results)))
        self._metrics.histogram("aggregation.failed_count", float(len(failed_results)))


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _split_results(results: Dict[str, Any]) -> tuple:
    success = {}
    failed = {}
    for name, value in results.items():
        item = _normalize_result(name, value)
        if item.success:
            success[name] = item.text
        else:
            failed[name] = item.text
    return success, failed


def _normalize_result(name: str, value: Any) -> AggregationInput:
    """Convert supported agent result shapes into an explicit success/failure."""
    if isinstance(value, dict):
        return _normalize_mapping_result(name, value)

    if isinstance(value, BaseMessage):
        if _is_final_ai_message(value) or isinstance(value, HumanMessage):
            return _text_result(name, value.content)
        return AggregationInput(name=name, success=False, text=str(value.content or ""), status="message")

    return _text_result(name, value)


def _normalize_mapping_result(name: str, value: Dict[str, Any]) -> AggregationInput:
    status = str(value.get("status") or "")
    has_success = "success" in value

    if has_success and value.get("success") is True:
        text = _stringify_result_payload(value.get("data", ""))
        if text:
            return AggregationInput(name=name, success=True, text=text, status=status or "completed")
        return AggregationInput(name=name, success=False, text="Empty successful result", status="empty")

    if has_success and value.get("success") is False:
        return AggregationInput(
            name=name,
            success=False,
            text=_structured_failure_text(value),
            status=status or "failed",
        )

    if "messages" in value:
        text = _extract_final_message_content(value.get("messages") or [])
        if text:
            return AggregationInput(name=name, success=True, text=text, status=status or "completed")

    if "data" in value:
        return _text_result(name, value.get("data"), status=status or "completed")

    return _text_result(name, value, status=status or "completed")


def _text_result(name: str, value: Any, status: str = "completed") -> AggregationInput:
    text = _stringify_result_payload(value)
    if text:
        return AggregationInput(name=name, success=True, text=text, status=status)
    return AggregationInput(name=name, success=False, text="", status="empty")


def _structured_failure_text(value: Dict[str, Any]) -> str:
    status = _stringify_result_payload(value.get("status", "failed")) or "failed"
    for key in ("error", "reason", "message"):
        text = _stringify_result_payload(value.get(key, ""))
        if text:
            return f"{status}: {text}" if status not in {"failed", "failure"} else text

    task_id = _stringify_result_payload(value.get("task_id", ""))
    if task_id:
        return f"{status}: {task_id}"
    return status


def _stringify_result_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseMessage):
        return str(value.content or "")
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _extract_final_message_content(messages: List[Any]) -> str:
    for message in reversed(messages):
        if _is_final_ai_message(message):
            return _stringify_result_payload(message.content)
    return ""


def _is_final_ai_message(message: Any) -> bool:
    return (
        isinstance(message, AIMessage)
        and bool(message.content)
        and not getattr(message, "tool_calls", None)
    )


def _no_results_fallback(state: Dict[str, Any], failed: Dict[str, str]) -> str:
    if failed:
        return "Errors occurred during execution. Please try again later."
    answer = _extract_final_message_content(state.get("messages", []))
    if answer:
        return answer
    return "No results available. Please try again."


def _last_human(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""
