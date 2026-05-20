"""Real-world customer service demo with mock data and no external API key.

Run from the repository root:

    python examples/customer_service_demo/demo.py
"""

import logging
import re
import time
from typing import Any, Dict, List

from langdeep import (
    AssistantMessage,
    FlowOrchestrator,
    UserMessage,
    agent,
    memory,
    model,
    register_tool,
    validate_runtime,
)
from langdeep.core.logging import configure
from langdeep.core.memory import memory_registry
from langdeep.core.observability import MetricsCollector
from langdeep.core.registry.tool_registry import tool_registry
from langdeep.core.tools import ToolExecutionPolicy

configure(level=logging.WARNING)

ORDERS: Dict[str, Dict[str, Any]] = {
    "LD-1001": {
        "customer": "Mia Chen",
        "status": "delivered",
        "item": "Noise-cancelling headphones",
        "days_since_delivery": 12,
        "condition": "unopened",
    },
    "LD-1002": {
        "customer": "Mia Chen",
        "status": "in_transit",
        "item": "USB-C dock",
        "days_since_delivery": None,
        "condition": "new",
    },
}

POLICIES = {
    "refund": "Refunds are available within 30 days for unopened items.",
    "shipping": "Standard delivery takes 3-5 business days after fulfillment.",
    "warranty": "Accessories include a one-year limited warranty after delivery.",
}

CASES: List[Dict[str, str]] = []
METRICS = MetricsCollector()


@model(name="support_demo_mock_chat", provider="mock", model_name="support-demo-mock-chat")
def support_demo_mock_chat():
    pass


@memory(name="customer_demo_memory", description="In-memory session store for the support demo")
def customer_demo_memory():
    pass


@register_tool(
    name="lookup_order",
    description="Look up mock order details by order ID.",
    category="support",
    tags=["customer-service", "knowledge-base"],
    timeout=2,
)
def lookup_order(order_id: str) -> Dict[str, Any]:
    """Look up mock order details by order ID."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return {"found": False, "order_id": order_id.upper()}
    return {"found": True, "order_id": order_id.upper(), **order}


@register_tool(
    name="lookup_policy",
    description="Look up support policy text by topic.",
    category="support",
    tags=["customer-service", "policy"],
    timeout=2,
)
def lookup_policy(topic: str) -> str:
    """Look up support policy text by topic."""
    return POLICIES.get(topic.lower(), "No matching policy was found.")


@register_tool(
    name="create_return_case",
    description="Create a mock return case after user confirmation.",
    category="support",
    tags=["customer-service", "case-management"],
    requires_confirmation=True,
    timeout=2,
)
def create_return_case(order_id: str, reason: str) -> Dict[str, str]:
    """Create a mock return case after user confirmation."""
    case_id = f"CASE-{len(CASES) + 1:04d}"
    record = {"case_id": case_id, "order_id": order_id.upper(), "reason": reason}
    CASES.append(record)
    return record


@agent(
    name="customer_service_demo_agent",
    description="Handle order, refund, shipping, warranty, and return-case questions.",
    routing_keywords=[
        "order",
        "refund",
        "return",
        "shipping",
        "delivery",
        "warranty",
        "case",
        "yes",
    ],
    model="support_demo_mock_chat",
    tools=["lookup_order", "lookup_policy", "create_return_case"],
)
def customer_service_demo_agent():
    class CustomerServiceDemoAgent:
        def invoke(self, state):
            question = _last_user_question(state)
            history = state.get("messages", [])
            order_id = _extract_order_id(question) or _extract_order_id_from_history(history)
            intent = _detect_intent(question)

            if not order_id:
                answer = (
                    "I can help with order status, refunds, shipping, and warranty questions. "
                    "Please provide an order ID such as LD-1001."
                )
                return {"messages": [AssistantMessage(content=answer)]}

            order = _invoke_tool("lookup_order", {"order_id": order_id})
            if not order.get("found"):
                return {
                    "messages": [
                        AssistantMessage(
                            content=f"I could not find order {order_id}. Please verify the order ID."
                        )
                    ]
                }

            if intent == "create_case":
                case = _invoke_tool(
                    "create_return_case",
                    {"order_id": order_id, "reason": "customer confirmed refund request"},
                    confirmed=True,
                )
                answer = (
                    f"Return case {case['case_id']} has been opened for {order_id}.\n"
                    "Next step: A support specialist will email the return label."
                )
            elif intent == "refund":
                policy = _invoke_tool("lookup_policy", {"topic": "refund"})
                eligible = _is_refund_eligible(order)
                answer = (
                    f"Order {order_id}: {order['item']} is {order['status']}.\n"
                    f"Policy: {policy}\n"
                    f"Eligibility: {'eligible' if eligible else 'not eligible'}.\n"
                    "Reply 'yes, open a return case' if you want me to create a case."
                )
            elif intent == "shipping":
                policy = _invoke_tool("lookup_policy", {"topic": "shipping"})
                answer = f"Order {order_id}: {order['status']}.\nPolicy: {policy}"
            elif intent == "warranty":
                policy = _invoke_tool("lookup_policy", {"topic": "warranty"})
                answer = f"Order {order_id}: {order['item']}.\nPolicy: {policy}"
            else:
                answer = (
                    f"Order {order_id}: {order['item']} for {order['customer']} is "
                    f"{order['status']}."
                )

            return {"messages": [AssistantMessage(content=answer)]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return CustomerServiceDemoAgent()


def configure_demo_runtime() -> None:
    """Attach policy and metrics to registered tools."""
    tool_registry.set_policy(ToolExecutionPolicy(enforce_confirmation=True))
    tool_registry.set_metrics_collector(METRICS)


def build_orchestrator() -> FlowOrchestrator:
    configure_demo_runtime()
    validate_runtime(instantiate_agents=True).raise_for_errors()
    return FlowOrchestrator(
        supervisor_model="support_demo_mock_chat",
        memory="customer_demo_memory",
        enable_checkpoint=False,
    )


def run_demo() -> Dict[str, Any]:
    orchestrator = build_orchestrator()
    CASES.clear()
    METRICS.clear()
    tool_registry.get_audit_log().clear()
    memory_backend = memory_registry.get_backend("customer_demo_memory")
    memory_backend.clear()

    session_id = "demo-customer-001"
    turns = [
        "Hi, I need help with order LD-1001.",
        "Can I return it for a refund?",
        "Yes, open a return case.",
    ]

    transcript = []
    for user_text in turns:
        assistant_text = orchestrator.chat_text(user_text, session_id=session_id)
        transcript.append((user_text, assistant_text))

    return {
        "session_id": session_id,
        "transcript": transcript,
        "memory_entries": memory_backend.get_entry_count(session_id),
        "audit_records": tool_registry.get_audit_log().list_records(),
        "tool_metrics": METRICS.get_metrics(),
        "orchestrator_metrics": orchestrator.get_metrics(),
    }


def main() -> None:
    summary = run_demo()

    print("Conversation")
    for idx, (user_text, assistant_text) in enumerate(summary["transcript"], start=1):
        print(f"{idx}. User: {user_text}")
        print(f"{idx}. Assistant: {assistant_text}")

    print(f"Memory entries: {summary['memory_entries']}")
    print("Tool audit")
    for record in summary["audit_records"]:
        print(
            f"- {record.tool_name}: success={record.success}, "
            f"confirmed={record.confirmed}, blocked={record.blocked}"
        )

    print("Tool metrics")
    for name, value in sorted(summary["tool_metrics"]["counters"].items()):
        print(f"- {name}: {int(value)}")


def _invoke_tool(name: str, payload: Dict[str, Any], *, confirmed: bool = False) -> Any:
    tool = tool_registry.get_tool(name)
    config = {"metadata": {"tool_confirmations": {name: True}}} if confirmed else None
    started = time.monotonic()
    try:
        result = tool.invoke(payload, config=config)
        METRICS.counter("customer_service.tool.calls", tags={"tool": name, "status": "success"})
        return result
    except Exception:
        METRICS.counter("customer_service.tool.calls", tags={"tool": name, "status": "failure"})
        raise
    finally:
        duration_ms = int((time.monotonic() - started) * 1000)
        METRICS.histogram("customer_service.tool.duration_ms", duration_ms, tags={"tool": name})


def _last_user_question(state) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, UserMessage):
            return str(message.content)
    return ""


def _extract_order_id(text: str) -> str:
    match = re.search(r"\bLD-\d{4}\b", text.upper())
    return match.group(0) if match else ""


def _extract_order_id_from_history(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            order_id = _extract_order_id(str(message.content))
            if order_id:
                return order_id
    return ""


def _detect_intent(text: str) -> str:
    lowered = text.lower()
    if "yes" in lowered and "case" in lowered:
        return "create_case"
    if any(word in lowered for word in ("refund", "return")):
        return "refund"
    if any(word in lowered for word in ("ship", "delivery", "tracking")):
        return "shipping"
    if "warranty" in lowered:
        return "warranty"
    return "status"


def _is_refund_eligible(order: Dict[str, Any]) -> bool:
    days = order.get("days_since_delivery")
    return (
        order.get("status") == "delivered"
        and order.get("condition") == "unopened"
        and isinstance(days, int)
        and days <= 30
    )


if __name__ == "__main__":
    main()
