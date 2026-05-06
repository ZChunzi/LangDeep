"""Abstract base class for IM platform adapters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import IMEvent, IMMessage, PlatformType


class IMPlatformAdapter(ABC):
    """Interface for IM platform adapters.

    Each platform (WeCom, DingTalk, Feishu, Slack) implements
    this interface to handle platform-specific message formats,
    signature validation, and response formatting.
    """

    @property
    @abstractmethod
    def platform(self) -> PlatformType:
        """The platform type this adapter handles."""
        ...

    @abstractmethod
    def parse_payload(self, raw_data: Dict[str, Any], headers: Optional[Dict] = None) -> IMEvent:
        """Parse an incoming webhook payload into an IMEvent."""
        ...

    @abstractmethod
    def validate_signature(self, raw_body: bytes, signature: str, timestamp: Optional[str] = None) -> bool:
        """Validate the request signature."""
        ...

    @abstractmethod
    def format_response(self, messages: List[IMMessage]) -> Dict[str, Any]:
        """Format outgoing messages into the platform's response format."""
        ...

    @abstractmethod
    def create_reply(self, original: IMEvent, text: str) -> Dict[str, Any]:
        """Create a reply to an incoming message."""
        ...

    def get_webhook_route(self) -> str:
        """Return the webhook path prefix for this platform."""
        return f"/webhook/{self.platform.value}"
