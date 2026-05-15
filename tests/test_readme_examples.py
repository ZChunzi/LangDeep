"""Smoke tests for README examples that should run without external services."""

import json
from typing import Dict

from langdeep import (
    AssistantMessage,
    FlowOrchestrator,
    ResultMerger,
    UserMessage,
    agent,
    last_assistant_text,
    model,
    register_tool,
    validate_runtime,
)
from conftest import clean_registries


def setup_function():
    clean_registries()


def test_readme_quick_start_runs_without_langchain_core_imports():
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
        routing_keywords=["weather", "天气"],
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

    validate_runtime(instantiate_agents=True).raise_for_errors()

    orchestrator = FlowOrchestrator(
        supervisor_model="mock_chat",
        enable_checkpoint=False,
    )

    result = orchestrator.invoke("How is the weather in Beijing?")

    assert last_assistant_text(result) == (
        "Question: How is the weather in Beijing?\n"
        "Beijing: sunny, 25C"
    )


def test_developer_guide_custom_result_merger_signature_runs():
    class JsonMerger(ResultMerger):
        def merge(self, user_request: str, agent_results: Dict[str, str]) -> str:
            return json.dumps(
                {
                    "request": user_request,
                    "results": agent_results,
                },
                ensure_ascii=False,
                sort_keys=True,
            )

    payload = JsonMerger().merge(
        "Summarize the research",
        {"research_agent": "LangDeep supports custom mergers."},
    )

    assert json.loads(payload) == {
        "request": "Summarize the research",
        "results": {"research_agent": "LangDeep supports custom mergers."},
    }
