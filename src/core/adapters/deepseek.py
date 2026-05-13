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

    LangChain's ``_convert_dict_to_message`` only puts ``function_call`` and
    ``audio`` into ``additional_kwargs`` — everything else from the API
    response (including ``reasoning_content``) is silently dropped.

    This class overrides ``_create_chat_result`` to capture
    ``reasoning_content`` from the raw API response and inject it into the
    ``AIMessage.additional_kwargs``, and overrides the message serialisation
    path to ensure it is carried forward in subsequent requests.
    """

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[Dict[str, Any]] = None,
    ):
        """Override: capture ``reasoning_content`` from the raw API response.

        LangChain's ``_convert_dict_to_message`` discards unknown response
        fields.  We extract ``reasoning_content`` from the raw dict *before*
        the parent processes the response, then inject it back into the
        resulting ``AIMessage.additional_kwargs``.
        """
        # Extract reasoning_content from the raw response dict
        reasoning_contents: List[Optional[str]] = []
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}},
            )
        )
        for choice in response_dict.get("choices") or []:
            raw_message = choice.get("message", {}) if isinstance(choice, dict) else {}
            reasoning_contents.append(raw_message.get("reasoning_content"))

        result = super()._create_chat_result(response, generation_info=generation_info)

        # Inject reasoning_content into the AIMessage's additional_kwargs
        for i, rc in enumerate(reasoning_contents):
            if rc is not None and i < len(result.generations):
                msg = result.generations[i].message
                if isinstance(msg, AIMessage):
                    msg.additional_kwargs["reasoning_content"] = rc

        return result

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
