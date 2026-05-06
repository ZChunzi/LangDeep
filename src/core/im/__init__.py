"""IM (Instant Messaging) integration module.

Provides abstract adapter interfaces, message models, a registry,
decorator support, and a framework-agnostic webhook receiver.

Platform adapters (WeCom, DingTalk, Feishu, Slack) implement
``IMPlatformAdapter`` and are registered via ``@im_channel``.
"""

from .models import (
    IMMessage,
    IMText,
    IMImage,
    IMInteractive,
    IMEvent,
    MessageType,
    PlatformType,
)
from .base import IMPlatformAdapter
from .registry import im_channel_registry, IMChannelRegistry
from .decorators import im_channel
from .webhook import WebhookReceiver, create_flask_blueprint, create_fastapi_router

__all__ = [
    "IMMessage",
    "IMText",
    "IMImage",
    "IMInteractive",
    "IMEvent",
    "MessageType",
    "PlatformType",
    "IMPlatformAdapter",
    "IMChannelRegistry",
    "im_channel_registry",
    "im_channel",
    "WebhookReceiver",
    "create_flask_blueprint",
    "create_fastapi_router",
]
