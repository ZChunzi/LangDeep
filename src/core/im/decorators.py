"""@im_channel decorator — registers an IM message handler."""

from typing import Any, Callable, Optional

from ..logging import get_logger
from .models import PlatformType
from .registry import im_channel_registry

logger = get_logger(__name__)


def im_channel(
    name: Optional[str] = None,
    platform: str = "custom",
    description: str = "",
    adapter: Optional[Any] = None,
):
    """Decorator that registers an IM message handler.

    Usage::

        @im_channel(name="helpdesk", platform="wecom", description="Help desk handler")
        def handle_helpdesk(event: IMEvent) -> str:
            return f"Received: {event.content}"

    Args:
        name: Channel name (defaults to function name).
        platform: Platform type (\"wecom\", \"dingtalk\", \"feishu\", \"slack\", \"custom\").
        description: Human-readable description.
        adapter: Optional platform adapter instance.
    """

    def decorator(func: Callable) -> Callable:
        channel_name = name or func.__name__
        platform_type = PlatformType(platform.lower())
        im_channel_registry.register_channel(
            name=channel_name,
            handler=func,
            platform=platform_type,
            description=description or func.__doc__ or "",
            adapter=adapter,
        )
        logger.info(
            "IM channel registered via decorator",
            extra={"channel": channel_name, "platform": platform},
        )
        return func

    return decorator
