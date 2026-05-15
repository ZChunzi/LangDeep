"""Minimal LangDeep agent example using the built-in mock provider.

Run from the repository root:

    python examples/basic_mock_agent.py
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


@model(name="mock_chat", provider="mock", model_name="mock-chat")
def mock_chat():
    pass


@register_tool(name="get_weather", description="Return mock weather.")
def get_weather(city: str) -> str:
    """Return mock weather."""
    return f"{city}: sunny, 25C"


@agent(
    name="weather_agent",
    description="Answer simple weather questions.",
    routing_keywords=["weather"],
    model="mock_chat",
    tools=["get_weather"],
)
def weather_agent():
    class WeatherAgent:
        def invoke(self, state):
            question = ""
            for message in reversed(state.get("messages", [])):
                if isinstance(message, UserMessage):
                    question = str(message.content)
                    break

            answer = f"Question: {question}\n{get_weather('Beijing')}"
            return {"messages": [AssistantMessage(content=answer)]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return WeatherAgent()


def main() -> None:
    validate_runtime(instantiate_agents=True).raise_for_errors()

    orchestrator = FlowOrchestrator(
        supervisor_model="mock_chat",
        enable_checkpoint=False,
    )

    result = orchestrator.invoke("How is the weather in Beijing?")
    print(last_assistant_text(result))


if __name__ == "__main__":
    main()
