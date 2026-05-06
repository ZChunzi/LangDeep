"""Data models for IM (instant messaging) platform messages."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    INTERACTIVE = "interactive"
    EVENT = "event"


class PlatformType(Enum):
    WECOM = "wecom"
    DINGTALK = "dingtalk"
    FEISHU = "feishu"
    SLACK = "slack"
    CUSTOM = "custom"


@dataclass
class IMMessage:
    """Base IM message."""
    msg_id: str
    session_id: str
    platform: PlatformType
    content: str
    msg_type: MessageType = MessageType.TEXT
    raw_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IMText(IMMessage):
    """Text message."""
    msg_type: MessageType = MessageType.TEXT


@dataclass
class IMImage(IMMessage):
    """Image message."""
    msg_type: MessageType = MessageType.IMAGE
    image_url: str = ""
    image_base64: Optional[str] = None
    alt_text: str = ""


@dataclass
class IMInteractive(IMMessage):
    """Interactive message (buttons, forms)."""
    msg_type: MessageType = MessageType.INTERACTIVE
    action: str = ""
    action_data: Dict[str, Any] = field(default_factory=dict)
    callback_id: str = ""


@dataclass
class IMEvent(IMMessage):
    """Non-message events (subscribe, file upload, etc.)."""
    msg_type: MessageType = MessageType.EVENT
    event_type: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)
