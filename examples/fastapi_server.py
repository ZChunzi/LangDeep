"""FastAPI server example using LangDeep with the built-in mock provider.

Install server dependencies and run from the repository root:

    python -m pip install "langdeep[server]"
    uvicorn examples.fastapi_server:app --reload
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langdeep import (
    AssistantMessage,
    FlowOrchestrator,
    UserMessage,
    agent,
    model,
    validate_runtime,
)
from langdeep.core.logging import configure

configure(level=logging.WARNING)

app = FastAPI(title="LangDeep FastAPI Example", version="1.0.0")
_orchestrator: Optional[FlowOrchestrator] = None


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    reply: str
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Runtime health response."""

    status: str
    diagnostics: Dict[str, Any]


@model(name="fastapi_mock_chat", provider="mock", model_name="fastapi-mock-chat")
def fastapi_mock_chat():
    pass


@agent(
    name="fastapi_chat_agent",
    description="Answer HTTP chat requests with a local mock response.",
    routing_keywords=["chat", "hello", "help", "question", "refund", "shipping", "weather"],
    model="fastapi_mock_chat",
)
def fastapi_chat_agent():
    class FastAPIChatAgent:
        def invoke(self, state):
            question = _last_user_question(state)
            return {
                "messages": [
                    AssistantMessage(
                        content=(
                            "Mock LangDeep response over HTTP.\n"
                            f"User message: {question}"
                        )
                    )
                ]
            }

        async def ainvoke(self, state):
            return self.invoke(state)

    return FastAPIChatAgent()


def _last_user_question(state) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, UserMessage):
            return str(message.content)
    return ""


def get_orchestrator() -> FlowOrchestrator:
    """Build the orchestrator lazily after registry validation."""
    global _orchestrator
    if _orchestrator is None:
        validate_runtime(instantiate_agents=True).raise_for_errors()
        _orchestrator = FlowOrchestrator(
            supervisor_model="fastapi_mock_chat",
            enable_checkpoint=False,
        )
    return _orchestrator


@app.on_event("startup")
def startup() -> None:
    """Validate LangDeep registry wiring during service startup."""
    get_orchestrator()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    diagnostics = validate_runtime(instantiate_agents=True)
    return HealthResponse(
        status="ok" if diagnostics.ok else "error",
        diagnostics=diagnostics.to_dict(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")
    try:
        reply = get_orchestrator().chat_text(message, session_id=request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply, session_id=request.session_id)


def main() -> None:
    import uvicorn

    uvicorn.run("examples.fastapi_server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
