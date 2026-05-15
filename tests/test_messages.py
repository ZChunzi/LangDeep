"""Tests for public LangDeep message helpers."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from langdeep import (
    AssistantMessage,
    FlowOrchestrator,
    UserMessage,
    assistant_message,
    last_assistant_text,
    message_text,
    user_message,
)
from conftest import clean_registries, orch, populate_minimal_registries


def setup_function():
    clean_registries()
    populate_minimal_registries()


def test_public_message_aliases_are_langchain_compatible():
    user = UserMessage(content="hello")
    assistant = AssistantMessage(content="hi")

    assert isinstance(user, HumanMessage)
    assert isinstance(assistant, AIMessage)
    assert isinstance(user_message("hello"), HumanMessage)
    assert isinstance(assistant_message("hi"), AIMessage)


def test_message_text_and_last_assistant_text_helpers():
    result = {
        "messages": [
            user_message("hello"),
            assistant_message(""),
            assistant_message([{"type": "text", "text": "final answer"}]),
        ]
    }

    assert message_text(result["messages"][-1]) == "final answer"
    assert last_assistant_text(result) == "final answer"


def test_chat_text_returns_last_assistant_text():
    o = orch()

    assert len(o.chat_text("你好")) > 5


def test_achat_text_returns_last_assistant_text():
    o = orch()

    async def run():
        return await o.achat_text("你好")

    assert len(asyncio.run(run())) > 5
