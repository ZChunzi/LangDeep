"""Webhook receiver for IM platforms.

Framework-agnostic core that accepts raw bytes + headers and returns a dict.
Optional Flask/FastAPI integration helpers are provided as separate functions.
"""

import json
import uuid
from typing import Any, Dict, Optional

from ..logging import get_logger
from .models import IMEvent, PlatformType
from .registry import im_channel_registry

logger = get_logger(__name__)


class WebhookReceiver:
    """Framework-agnostic webhook receiver for IM platforms.

    Usage::

        receiver = WebhookReceiver()
        # In Flask:
        @app.route("/webhook/<platform>", methods=["POST"])
        def handle(request, platform):
            return receiver.handle_request(request.data, request.headers, platform)

        # In FastAPI:
        @app.post("/webhook/{platform}")
        async def handle(request: Request, platform: str):
            return receiver.handle_request(await request.body(), request.headers, platform)
    """

    def __init__(
        self,
        registry=None,
        verify_signature: bool = True,
        default_platform: PlatformType = PlatformType.CUSTOM,
    ):
        self._registry = registry or im_channel_registry
        self._verify_signature = verify_signature
        self._default_platform = default_platform

    def handle_request(
        self,
        raw_body: bytes,
        headers: Optional[Dict[str, str]] = None,
        platform: Optional[PlatformType] = None,
    ) -> Dict[str, Any]:
        """Process an incoming webhook request.

        1. Determine platform (from route or header)
        2. Get the adapter for that platform
        3. Validate signature (if enabled)
        4. Parse payload into IMEvent
        5. Dispatch to registered handler
        6. Return formatted response
        """
        headers = headers or {}
        platform = platform or self._detect_platform(headers)
        adapter = self._registry.get_adapter(platform)

        # Signature validation
        if self._verify_signature and adapter:
            signature = headers.get("X-Signature", headers.get("x-signature", ""))
            timestamp = headers.get("X-Timestamp", headers.get("x-timestamp", ""))
            if not adapter.validate_signature(raw_body, signature, timestamp):
                logger.warning("Invalid signature", extra={"platform": platform.value})
                return {"error": "Invalid signature"}

        # Parse payload
        try:
            raw_data = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Failed to parse webhook body", extra={"error": str(exc)})
            return {"error": "Invalid JSON body"}

        if adapter:
            event = adapter.parse_payload(raw_data, headers)
        else:
            event = self._default_parse(raw_data, platform)

        # Dispatch
        try:
            response = self._registry.dispatch(event)
            if adapter:
                if isinstance(response, list):
                    return adapter.format_response(response)
                if hasattr(response, "msg_type"):
                    return adapter.format_response([response])
                if isinstance(response, dict):
                    return response
            return {"content": str(response)}
        except Exception as exc:
            logger.error("Webhook dispatch failed", extra={"error": str(exc)}, exc_info=True)
            return {"error": str(exc)}

    def _detect_platform(self, headers: Dict[str, str]) -> PlatformType:
        """Detect platform from request headers."""
        ua = headers.get("User-Agent", "").lower()
        if "wecom" in ua or "wxwork" in ua:
            return PlatformType.WECOM
        if "dingtalk" in ua:
            return PlatformType.DINGTALK
        if "feishu" in ua or "lark" in ua:
            return PlatformType.FEISHU
        if "slack" in ua:
            return PlatformType.SLACK
        x_platform = headers.get("X-Platform", "")
        if x_platform:
            try:
                return PlatformType(x_platform.lower())
            except ValueError:
                pass
        return self._default_platform

    def _default_parse(self, raw_data: Dict, platform: PlatformType) -> IMEvent:
        """Default payload parser when no adapter is available."""
        return IMEvent(
            msg_id=raw_data.get("msg_id", uuid.uuid4().hex[:12]),
            session_id=raw_data.get("session_id", raw_data.get("channel", "")),
            platform=platform,
            content=raw_data.get("content", raw_data.get("text", "")),
            raw_data=raw_data,
        )


# ── Optional framework integration helpers ──────────────────────────────────


def create_flask_blueprint(receiver: WebhookReceiver, url_prefix: str = "/webhook"):
    """Create a Flask Blueprint for the webhook receiver.

    Only works if Flask is installed. Returns None otherwise.
    """
    try:
        from flask import Blueprint, request, jsonify
    except ImportError:
        logger.warning("Flask not installed, cannot create webhook blueprint")
        return None

    bp = Blueprint("langdeep_webhook", __name__, url_prefix=url_prefix)

    @bp.route("/<platform>", methods=["POST"])
    def handle_platform(platform):
        try:
            pt = PlatformType(platform.lower())
        except ValueError:
            return jsonify({"error": f"Unknown platform: {platform}"}), 400
        result = receiver.handle_request(request.data, dict(request.headers), pt)
        return jsonify(result)

    @bp.route("", methods=["POST"])
    def handle_default():
        result = receiver.handle_request(request.data, dict(request.headers))
        return jsonify(result)

    return bp


def create_fastapi_router(receiver: WebhookReceiver, url_prefix: str = "/webhook"):
    """Create a FastAPI APIRouter for the webhook receiver.

    Only works if FastAPI is installed. Returns None otherwise.
    """
    try:
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.warning("FastAPI not installed, cannot create webhook router")
        return None

    router = APIRouter(prefix=url_prefix)

    @router.post("/{platform}")
    async def handle_platform(platform: str, request: Request):
        body = await request.body()
        headers = dict(request.headers)
        try:
            pt = PlatformType(platform.lower())
        except ValueError:
            return JSONResponse({"error": f"Unknown platform: {platform}"}, status_code=400)
        result = receiver.handle_request(body, headers, pt)
        return JSONResponse(result)

    @router.post("")
    async def handle_default(request: Request):
        body = await request.body()
        headers = dict(request.headers)
        result = receiver.handle_request(body, headers)
        return JSONResponse(result)

    return router
