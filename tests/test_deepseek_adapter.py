"""Tests for DeepSeek LangChain compatibility helpers."""

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langdeep.core.adapters.deepseek import (
    DeepSeekChatModel,
    DeepSeekCompatibilityProfile,
    build_deepseek_payload_messages,
    configure_deepseek_v4,
    extract_reasoning_content,
    normalize_deepseek_messages,
)


def _assistant_with_reasoning(*, tool_call: bool = False) -> AIMessage:
    kwargs = {"reasoning_content": "think first"}
    if tool_call:
        kwargs["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
    return AIMessage(content="", additional_kwargs=kwargs)


def test_v4_auto_policy_replays_reasoning_content_for_tool_calls():
    profile = DeepSeekCompatibilityProfile(model_name="deepseek-v4-pro")
    payload = build_deepseek_payload_messages(
        [HumanMessage(content="hi"), _assistant_with_reasoning(tool_call=True)],
        profile,
    )

    assistant_payload = payload[1]
    assert assistant_payload["role"] == "assistant"
    assert assistant_payload["content"] == ""
    assert assistant_payload["reasoning_content"] == "think first"
    assert assistant_payload["tool_calls"][0]["function"]["name"] == "lookup"


def test_v4_auto_policy_drops_reasoning_content_without_tool_calls():
    profile = DeepSeekCompatibilityProfile(model_name="deepseek-v4-pro")
    payload = build_deepseek_payload_messages(
        [_assistant_with_reasoning(tool_call=False)],
        profile,
    )

    assert "reasoning_content" not in payload[0]


def test_reasoner_auto_policy_removes_reasoning_content_from_history():
    profile = DeepSeekCompatibilityProfile(model_name="deepseek-reasoner")
    normalized = normalize_deepseek_messages([_assistant_with_reasoning()], profile)
    payload = build_deepseek_payload_messages(normalized, profile)

    assert extract_reasoning_content(normalized[0]) is None
    assert "reasoning_content" not in payload[0]


def test_disabled_thinking_drops_reasoning_content():
    profile = DeepSeekCompatibilityProfile(
        model_name="deepseek-v4-pro",
        thinking_enabled=False,
    )
    payload = build_deepseek_payload_messages(
        [_assistant_with_reasoning(tool_call=True)],
        profile,
    )

    assert "reasoning_content" not in payload[0]


def test_preserve_policy_replays_reasoning_content_without_tool_calls():
    profile = DeepSeekCompatibilityProfile(
        model_name="deepseek-v4-pro",
        reasoning_content_policy="preserve",
    )
    payload = build_deepseek_payload_messages([_assistant_with_reasoning()], profile)

    assert payload[0]["reasoning_content"] == "think first"


def test_configure_deepseek_v4_returns_langchain_extra_params():
    params = configure_deepseek_v4(
        thinking="enabled",
        reasoning_effort="max",
        extra_params={"timeout": 30},
    )

    assert params["timeout"] == 30
    assert params["reasoning_effort"] == "max"
    assert params["reasoning_content_policy"] == "auto"
    assert params["extra_body"]["thinking"]["type"] == "enabled"


def test_streaming_reasoning_content_is_preserved_on_chunks():
    model = DeepSeekChatModel.model_construct(
        model_name="deepseek-v4-pro",
        model_kwargs={},
        output_version=None,
        reasoning_content_policy="auto",
    )

    chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "streamed thought",
                },
                "finish_reason": None,
            }
        ],
        "model": "deepseek-v4-pro",
    }

    generation = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

    assert generation is not None
    assert generation.message.additional_kwargs["reasoning_content"] == "streamed thought"


def test_chat_model_request_payload_applies_disabled_thinking_policy():
    model = DeepSeekChatModel.model_construct(
        model_name="deepseek-v4-pro",
        model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        output_version=None,
        reasoning_content_policy="auto",
    )
    message = _assistant_with_reasoning(tool_call=True)

    payload = model._get_request_payload([message])

    assert payload["extra_body"]["thinking"]["type"] == "disabled"
    assert "reasoning_content" not in payload["messages"][0]
