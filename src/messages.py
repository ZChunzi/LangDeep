"""Convenience wrappers around LangChain message primitives."""

from typing import Any, Iterable, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

Message = BaseMessage
UserMessage = HumanMessage
AssistantMessage = AIMessage


def user_message(content: str, **kwargs: Any) -> HumanMessage:
    """Create a user message without importing ``langchain_core`` directly."""
    return HumanMessage(content=content, **kwargs)


def assistant_message(content: str, **kwargs: Any) -> AIMessage:
    """Create an assistant message without importing ``langchain_core`` directly."""
    return AIMessage(content=content, **kwargs)


def system_message(content: str, **kwargs: Any) -> SystemMessage:
    """Create a system message without importing ``langchain_core`` directly."""
    return SystemMessage(content=content, **kwargs)


def tool_message(content: str, tool_call_id: str, **kwargs: Any) -> ToolMessage:
    """Create a tool message without importing ``langchain_core`` directly."""
    return ToolMessage(content=content, tool_call_id=tool_call_id, **kwargs)


def message_text(message: BaseMessage) -> str:
    """Extract display text from a LangChain message."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip() if content is not None else ""


def last_assistant_text(result_or_messages: Any) -> str:
    """Return the last non-empty assistant message text from a result or list."""
    messages = (
        result_or_messages.get("messages")
        if isinstance(result_or_messages, dict)
        else result_or_messages
    )
    if messages is None:
        return ""
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            text = message_text(message)
            if text:
                return text
    return ""


def last_user_text(messages: Iterable[BaseMessage]) -> str:
    """Return the last user message text from a message sequence."""
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


__all__ = [
    "AssistantMessage",
    "Message",
    "UserMessage",
    "assistant_message",
    "last_assistant_text",
    "last_user_text",
    "message_text",
    "system_message",
    "tool_message",
    "user_message",
]
