# Getting Started

This path runs entirely with the built-in `mock` provider and does not require
an external LLM API key.

## Install

```bash
python -m pip install langdeep
```

For a source checkout:

```bash
python -m pip install -e ".[dev]"
```

## Five-Minute Example

```python
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
            return {"messages": [AssistantMessage(content=f"{question}\n{get_weather('Beijing')}")]}

        async def ainvoke(self, state):
            return self.invoke(state)

    return WeatherAgent()


validate_runtime(instantiate_agents=True).raise_for_errors()
orchestrator = FlowOrchestrator(supervisor_model="mock_chat", enable_checkpoint=False)
print(last_assistant_text(orchestrator.invoke("weather in Beijing")))
```

Runnable versions live in:

- `examples/basic_mock_agent.py`
- `examples/customer_support_agent.py`
