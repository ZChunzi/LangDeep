"""DeepSeek v4 thinking mode adapter - custom message serialization.

DeepSeek v4's thinking mode returns ``reasoning_content`` alongside the normal
``content`` field. For v4 tool-call turns, the API requires that the assistant
message's ``reasoning_content`` is replayed in later requests. The legacy
``deepseek-reasoner`` endpoint has the opposite rule: reasoning content must be
removed from subsequent requests.

LangChain's built-in ``ChatOpenAI`` does not handle this correctly because:

1. It serialises ``AIMessage(content=None, additional_kwargs={...})`` as
   ``"content": null`` in the JSON body - DeepSeek v4 rejects ``null``
   content.
2. It discards provider-specific fields such as ``reasoning_content`` when
   converting LangChain messages to OpenAI-compatible request dictionaries.

Usage
-----
    >>> from langdeep.core.adapters.deepseek import DeepSeekChatModel
    >>> llm = DeepSeekChatModel(
    ...     model="deepseek-v4-pro",
    ...     reasoning_effort="high",
    ...     extra_body={"thinking": {"type": "enabled"}},
    ... )

The model registry automatically uses ``DeepSeekChatModel`` when the
provider is ``"deepseek"``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI

from ..logging import get_logger

logger = get_logger(__name__)

ReasoningContentPolicy = Literal["auto", "preserve", "tool_calls", "drop"]
ResolvedReasoningContentPolicy = Literal["preserve", "tool_calls", "drop"]
_SUPPORTED_POLICIES = {"auto", "preserve", "tool_calls", "drop"}


@dataclass(frozen=True)
class DeepSeekCompatibilityProfile:
    """Message-serialization policy for DeepSeek OpenAI-compatible models."""

    model_name: str = ""
    reasoning_content_policy: ReasoningContentPolicy = "auto"
    thinking_enabled: Optional[bool] = None

    def resolved_reasoning_content_policy(self) -> ResolvedReasoningContentPolicy:
        """Resolve ``auto`` into an explicit replay/drop behavior."""
        policy = self.reasoning_content_policy
        if policy not in _SUPPORTED_POLICIES:
            raise ValueError(
                "reasoning_content_policy must be one of "
                f"{sorted(_SUPPORTED_POLICIES)}"
            )
        if policy != "auto":
            return policy
        if self.thinking_enabled is False:
            return "drop"

        model = self.model_name.lower()
        if "reasoner" in model:
            return "drop"
        if model.startswith("deepseek-v4"):
            return "tool_calls"
        return "preserve"


def configure_deepseek_v4(
    *,
    thinking: Literal["enabled", "disabled"] = "enabled",
    reasoning_effort: Literal["high", "max"] = "high",
    reasoning_content_policy: ReasoningContentPolicy = "auto",
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return ``ModelConfig.extra_params`` for DeepSeek v4 models.

    This small helper keeps application code from depending on OpenAI SDK
    ``extra_body`` details while preserving explicit escape hatches.
    """
    params = dict(extra_params or {})
    extra_body = dict(params.get("extra_body") or {})
    extra_body["thinking"] = {"type": thinking}
    params["extra_body"] = extra_body
    params["reasoning_effort"] = reasoning_effort
    params["reasoning_content_policy"] = reasoning_content_policy
    return params


def extract_reasoning_content(message: BaseMessage) -> Optional[str]:
    """Read DeepSeek reasoning text from a LangChain message or chunk."""
    value = getattr(message, "additional_kwargs", {}).get("reasoning_content")
    return value if isinstance(value, str) else None


def _sanitize(message: BaseMessage, profile: Optional[DeepSeekCompatibilityProfile] = None) -> BaseMessage:
    """Fix a single message for DeepSeek request compatibility.

    * Ensures assistant ``content`` is never ``None``.
    * Drops ``reasoning_content`` when the selected profile requires it.
    * Preserves ``tool_calls`` and ``id``.
    """
    if not isinstance(message, AIMessage):
        return message

    content = message.content if message.content is not None else ""
    kwargs = dict(message.additional_kwargs)
    if profile and profile.resolved_reasoning_content_policy() == "drop":
        kwargs.pop("reasoning_content", None)
    tool_calls = list(message.tool_calls) if message.tool_calls else []

    msg_id = getattr(message, "id", None) or getattr(message, "id_", None)
    return AIMessage(
        content=content,
        additional_kwargs=kwargs,
        tool_calls=tool_calls,
        id=msg_id,
    )


def normalize_deepseek_messages(
    messages: Iterable[BaseMessage],
    profile: Optional[DeepSeekCompatibilityProfile] = None,
) -> List[BaseMessage]:
    """Return LangChain messages normalized for DeepSeek request payloads."""
    return [_sanitize(message, profile=profile) for message in messages]


def _extract_thinking_enabled(extra_body: Optional[Mapping[str, Any]]) -> Optional[bool]:
    if not extra_body:
        return None
    thinking = extra_body.get("thinking")
    if not isinstance(thinking, Mapping):
        return None
    value = thinking.get("type")
    if value == "enabled":
        return True
    if value == "disabled":
        return False
    enabled = thinking.get("enabled")
    return enabled if isinstance(enabled, bool) else None


def _has_tool_call_payload(message: AIMessage) -> bool:
    return bool(
        message.tool_calls
        or message.invalid_tool_calls
        or message.additional_kwargs.get("tool_calls")
        or message.additional_kwargs.get("function_call")
    )


def _should_replay_reasoning(
    message: BaseMessage,
    profile: DeepSeekCompatibilityProfile,
) -> bool:
    if not isinstance(message, AIMessage):
        return False
    if not extract_reasoning_content(message):
        return False

    policy = profile.resolved_reasoning_content_policy()
    if policy == "drop":
        return False
    if policy == "preserve":
        return True
    return _has_tool_call_payload(message)


def _patch_payload_messages(
    payload_messages: Sequence[Dict[str, Any]],
    source_messages: Sequence[BaseMessage],
    profile: DeepSeekCompatibilityProfile,
) -> None:
    """Patch OpenAI-compatible request dicts with DeepSeek-specific fields."""
    for payload_message, source_message in zip(payload_messages, source_messages):
        if payload_message.get("role") != "assistant":
            continue

        if payload_message.get("content") is None:
            payload_message["content"] = ""

        reasoning_content = extract_reasoning_content(source_message)
        if reasoning_content and _should_replay_reasoning(source_message, profile):
            payload_message["reasoning_content"] = reasoning_content
        else:
            payload_message.pop("reasoning_content", None)


def build_deepseek_payload_messages(
    messages: Iterable[BaseMessage],
    profile: Optional[DeepSeekCompatibilityProfile] = None,
) -> List[Dict[str, Any]]:
    """Convert LangChain messages to DeepSeek-compatible request dictionaries.

    The public helper is intentionally small: users that build custom providers
    can reuse LangDeep's DeepSeek rules without subclassing ``ChatOpenAI``.
    """
    from langchain_openai.chat_models.base import _convert_message_to_dict

    active_profile = profile or DeepSeekCompatibilityProfile()
    normalized = normalize_deepseek_messages(messages, profile=active_profile)
    payload_messages = [_convert_message_to_dict(message) for message in normalized]
    _patch_payload_messages(payload_messages, normalized, active_profile)
    return payload_messages


def _extract_delta_reasoning_content(chunk: Mapping[str, Any]) -> Optional[str]:
    choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    reasoning_content = delta.get("reasoning_content")
    return reasoning_content if isinstance(reasoning_content, str) else None


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

    reasoning_content_policy: ReasoningContentPolicy = "auto"

    def _compatibility_profile(
        self,
        *,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> DeepSeekCompatibilityProfile:
        payload = payload or {}
        extra_body = payload.get("extra_body")
        if not isinstance(extra_body, Mapping):
            extra_body = getattr(self, "model_kwargs", {}).get("extra_body")
        if not isinstance(extra_body, Mapping):
            extra_body = None

        return DeepSeekCompatibilityProfile(
            model_name=str(payload.get("model") or getattr(self, "model_name", "") or ""),
            reasoning_content_policy=self.reasoning_content_policy,
            thinking_enabled=_extract_thinking_enabled(extra_body),
        )

    def _get_request_payload(
        self,
        input_,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build a request payload that preserves DeepSeek-specific fields."""
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        profile = self._compatibility_profile(payload=payload)
        normalized = normalize_deepseek_messages(messages, profile=profile)
        payload = super()._get_request_payload(normalized, stop=stop, **kwargs)
        _patch_payload_messages(payload.get("messages", []), normalized, profile)
        return payload

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

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: Dict[str, Any],
        default_chunk_class: type,
        base_generation_info: Optional[Dict[str, Any]],
    ):
        """Override: preserve streaming ``reasoning_content`` deltas."""
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        reasoning_content = _extract_delta_reasoning_content(chunk)
        if generation_chunk is None:
            return None
        if reasoning_content and isinstance(generation_chunk.message, AIMessageChunk):
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content
        return generation_chunk


__all__ = [
    "DeepSeekChatModel",
    "DeepSeekCompatibilityProfile",
    "ReasoningContentPolicy",
    "build_deepseek_payload_messages",
    "configure_deepseek_v4",
    "extract_reasoning_content",
    "normalize_deepseek_messages",
    "_sanitize",
]
