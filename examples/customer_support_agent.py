"""Customer support agent example using a local mock knowledge base.

Run from the repository root:

    python examples/customer_support_agent.py
"""

import logging

from langdeep import (
    AssistantMessage,
    FlowOrchestrator,
    UserMessage,
    agent,
    last_assistant_text,
    model,
    register_tool,
    validate_runtime,
)
from langdeep.core.logging import configure

configure(level=logging.WARNING)

POLICIES = {
    "refund": (
        "Refund requests are accepted within 30 days when the order is unused "
        "and includes the original receipt."
    ),
    "shipping": "Standard shipping takes 3-5 business days after fulfillment.",
    "warranty": "Hardware accessories include a one-year limited warranty.",
}


@model(name="support_mock_chat", provider="mock", model_name="support-mock-chat")
def support_mock_chat():
    pass


@register_tool(
    name="lookup_policy",
    description="Look up a customer support policy by topic.",
    category="support",
    tags=["customer-support", "policy"],
)
def lookup_policy(topic: str) -> str:
    """Look up a customer support policy by topic."""
    normalized = topic.strip().lower()
    return POLICIES.get(normalized, "No matching policy was found.")


@agent(
    name="customer_support_agent",
    description="Answer customer questions with local policy lookups.",
    routing_keywords=["refund", "return", "shipping", "warranty", "support", "order"],
    model="support_mock_chat",
    tools=["lookup_policy"],
)
def customer_support_agent():
    class CustomerSupportAgent:
        def invoke(self, state):
            question = _last_user_question(state)
            topic = _detect_topic(question)
            policy = lookup_policy(topic)
            answer = (
                f"Customer question: {question}\n"
                f"Matched topic: {topic}\n"
                f"Policy: {policy}\n"
                "Next step: Ask for the order number and verify eligibility before creating a case."
            )
            return {"messages": [AssistantMessage(content=answer)]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return CustomerSupportAgent()


def _last_user_question(state) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, UserMessage):
            return str(message.content)
    return ""


def _detect_topic(question: str) -> str:
    text = question.lower()
    if any(word in text for word in ("refund", "return", "cancel")):
        return "refund"
    if any(word in text for word in ("ship", "delivery", "tracking")):
        return "shipping"
    if any(word in text for word in ("warranty", "repair", "replace")):
        return "warranty"
    return "refund"


def main() -> None:
    validate_runtime(instantiate_agents=True).raise_for_errors()

    orchestrator = FlowOrchestrator(
        supervisor_model="support_mock_chat",
        enable_checkpoint=False,
    )

    result = orchestrator.invoke("Can I get a refund for an unused order?")
    print(last_assistant_text(result))


if __name__ == "__main__":
    main()
