"""Unit tests for _clean_messages — the tool-call-chain-preserving message cleaner."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from langdeep.core.orchestrator.executor import _clean_messages


def _tool_call_msg(content: str = "", tool_call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": "test_tool", "args": {"x": "1"}, "id": tool_call_id}],
    )


def _tool_result(content: str = "done", tool_call_id: str = "call_1") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


# ── ReAct chain preservation ──────────────────────────────────────────

def test_preserves_tool_calls_aimessage():
    """AIMessage with tool_calls MUST be kept — ReAct loop invariant."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="write a script"),
        _tool_call_msg(),
        _tool_result(),
        AIMessage(content="Script written."),
    ]
    result = _clean_messages(msgs)
    assert len(result) == 4
    assert isinstance(result[1], AIMessage)
    assert result[1].tool_calls is not None
    assert len(result[1].tool_calls) == 1


def test_preserves_toolmessage():
    """ToolMessage must survive cleaning — tool results needed by next ReAct round."""
    msgs = [HumanMessage(content="do it"), _tool_call_msg(), _tool_result(content="success")]
    result = _clean_messages(msgs)
    assert isinstance(result[2], ToolMessage)
    assert result[2].content == "success"


def test_preserves_human_and_system():
    msgs = [SystemMessage(content="be helpful"), HumanMessage(content="hello")]
    result = _clean_messages(msgs)
    assert len(result) == 2


# ── Empty-message stripping ───────────────────────────────────────────

def test_strips_empty_aimessage():
    """AIMessage with no content and no tool_calls is noise → dropped."""
    msgs = [HumanMessage(content="hi"), AIMessage(content=""), HumanMessage(content="again")]
    result = _clean_messages(msgs)
    assert len(result) == 2
    assert all(isinstance(m, HumanMessage) for m in result)


def test_strips_empty_aimessage2():
    msgs = [AIMessage(content=""), HumanMessage(content="x")]
    result = _clean_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)


# ── Context window capping ────────────────────────────────────────────

def test_caps_at_80_default():
    many = [HumanMessage(content=f"msg_{i}") for i in range(100)]
    result = _clean_messages(many)
    assert len(result) == 80
    # First two preserved
    assert result[0].content == "msg_0"
    assert result[1].content == "msg_1"
    # Last messages preserved
    assert "msg_99" in result[-1].content


def test_custom_max_messages():
    many = [HumanMessage(content=f"m{i}") for i in range(20)]
    result = _clean_messages(many, max_messages=10)
    assert len(result) == 10


def test_cap_keeps_react_chain():
    """Even when capped, tool-call pairs should be preserved at the tail."""
    msgs = [HumanMessage(content=f"msg_{i}") for i in range(80)]
    msgs.append(HumanMessage(content="final request"))
    tc_msg = _tool_call_msg(tool_call_id="final_call")
    tr_msg = _tool_result(content="final result", tool_call_id="final_call")
    msgs.extend([tc_msg, tr_msg, AIMessage(content="all done")])

    result = _clean_messages(msgs, max_messages=10)
    assert len(result) == 10
    # The tool-call pair should be at the end
    assert isinstance(result[-3], AIMessage)  # final request (human)
    assert result[-1].content == "all done"


# ── Edge cases ────────────────────────────────────────────────────────

def test_empty_list():
    assert _clean_messages([]) == []


def test_only_tool_calls():
    msgs = [_tool_call_msg(), _tool_result()]
    result = _clean_messages(msgs)
    assert len(result) == 2


def test_mixed_types():
    msgs: list[BaseMessage] = [
        SystemMessage(content="system"),
        HumanMessage(content="user"),
        _tool_call_msg(),
        _tool_result(),
        AIMessage(content="response"),
        AIMessage(content=""),  # noise
    ]
    result = _clean_messages(msgs)
    assert len(result) == 5  # tool_call msg is preserved (not stripped) + all others
