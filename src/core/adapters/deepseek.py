"""DeepSeek v4 thinking mode adapter — custom message serialization.

DeepSeek v4's "thinking mode" returns ``reasoning_content`` alongside (or
instead of) the normal ``content`` field.  The API *requires* that any
assistant message which originally carried ``reasoning_content`` includes it
again when the conversation continues — otherwise a 400 error is returned.

LangChain's built-in ``ChatOpenAI`` does not handle this correctly because:

1. It serialises ``AIMessage(content=None, additional_kwargs={…})`` as
   ``"content": null`` in the JSON body — DeepSeek v4 rejects ``null``
   content.
2. It has no concept of "thinking mode" and therefore does not track whether
   ``reasoning_content`` needs to be carried forward.

Usage
-----
    >>> from langdeep.core.adapters.deepseek import DeepSeekChatModel
    >>> llm = DeepSeekChatModel(model="deepseek-chat", …, extra_body={"thinking": {"type": "enabled"}})

The model registry automatically uses ``DeepSeekChatModel`` when the
provider is ``"deepseek"``.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from ..logging import get_logger

logger = get_logger(__name__)


def _sanitize(message: BaseMessage) -> BaseMessage:
    """Fix a single message for DeepSeek v4 thinking-mode compatibility.

    * Ensures ``content`` is never ``None`` (→ empty string).
    * Preserves ``additional_kwargs`` verbatim (carries ``reasoning_content``).
    * Preserves ``tool_calls`` and ``id``.
    """
    if not isinstance(message, AIMessage):
        return message

    content = message.content if message.content is not None else ""
    kwargs = dict(message.additional_kwargs)
    tool_calls = list(message.tool_calls) if message.tool_calls else []

    msg_id = getattr(message, "id", None) or getattr(message, "id_", None)
    return AIMessage(
        content=content,
        additional_kwargs=kwargs,
        tool_calls=tool_calls,
        id=msg_id,
    )


class DeepSeekChatModel(ChatOpenAI):
    """ChatOpenAI subclass that handles DeepSeek v4 thinking mode.

    Intercepts message serialisation so that ``reasoning_content`` is
    properly carried forward in multi-turn conversations.
    """

    def _create_message_dicts(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Override: sanitise messages before converting to API dicts."""
        sanitized = [_sanitize(m) for m in messages]
        return super()._create_message_dicts(sanitized, stop=stop)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ):
        """Override: sanitise messages before generating."""
        sanitized = [_sanitize(m) for m in messages]
        return super()._generate(sanitized, stop=stop, run_manager=run_manager, **kwargs)


__all__ = ["DeepSeekChatModel", "_sanitize"]
