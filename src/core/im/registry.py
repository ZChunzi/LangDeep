"""Registry for IM channel handlers."""

import copy
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..logging import get_logger
from ..errors import ConfigurationError
from .models import IMEvent, PlatformType

if TYPE_CHECKING:
    from ..orchestrator import FlowOrchestrator

logger = get_logger(__name__)


class IMChannelRegistry:
    """Singleton registry for IM message handlers.

    Each handler is a callable that accepts an IMEvent and returns
    a response (str, dict, or List[IMMessage]).
    Follows the same singleton pattern as AgentRegistry / MemoryRegistry.
    """

    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._channels: Dict[str, Dict] = {}
                    cls._instance._adapters: Dict[str, Any] = {}
                    cls._instance._registry_lock = threading.RLock()
        return cls._instance

    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register a platform adapter by name."""
        with self._registry_lock:
            self._adapters[name] = adapter
        logger.info("IM adapter registered", extra={"adapter": name})

    def register_channel(
        self,
        name: str,
        handler: Callable[[IMEvent], Any],
        platform: PlatformType = PlatformType.CUSTOM,
        description: str = "",
        adapter: Optional[Any] = None,
    ) -> None:
        """Register a message handler for an IM channel."""
        with self._registry_lock:
            self._channels[name] = {
                "handler": handler,
                "platform": platform,
                "adapter": adapter,
                "description": description,
                "created_at": datetime.now(),
            }
        logger.info(
            "IM channel registered",
            extra={"channel": name, "platform": platform.value},
        )

    def dispatch(self, event: IMEvent) -> Any:
        """Find the best matching channel and dispatch the event."""
        with self._registry_lock:
            channels = list(self._channels.items())

        for name, info in channels:
            if info["platform"] == event.platform:
                handler = info["handler"]
                adapter = info.get("adapter")
                result = handler(event)
                if adapter and hasattr(result, "msg_type"):
                    return adapter.format_response([result])
                return result

        raise ConfigurationError(
            f"No channel handler registered for platform {event.platform.value}",
            context={"available_channels": [name for name, _ in channels]},
        )

    def connect_orchestrator(
        self,
        orchestrator: "FlowOrchestrator",
        channel_name: str = "im_default",
        platform: PlatformType = PlatformType.CUSTOM,
    ) -> None:
        """Route IM messages through FlowOrchestrator."""
        def handler(event: IMEvent) -> Any:
            return orchestrator.invoke(
                user_input=event.content,
                context={
                    "im_session": event.session_id,
                    "im_platform": event.platform.value,
                    "im_msg_id": event.msg_id,
                },
            )
        self.register_channel(
            f"orchestrator_{channel_name}",
            handler,
            platform=platform,
            description=f"Routed through FlowOrchestrator ({channel_name})",
        )

    def list_channels(self) -> List[Dict[str, str]]:
        with self._registry_lock:
            return [
                {
                    "name": name,
                    "platform": info["platform"].value,
                    "description": info["description"],
                }
                for name, info in self._channels.items()
            ]

    def get_adapter(self, platform: PlatformType) -> Optional[Any]:
        with self._registry_lock:
            channels = list(self._channels.values())

        for info in channels:
            adapter = info.get("adapter")
            if adapter and getattr(adapter, "platform", None) == platform:
                return adapter
        return None

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow runtime snapshot with copied channel metadata."""
        with self._registry_lock:
            return {
                "channels": copy.deepcopy(self._channels),
                "adapters": dict(self._adapters),
            }

    def reset(self) -> None:
        """Clear registered channels and adapters."""
        with self._registry_lock:
            self._channels.clear()
            self._adapters.clear()

    clear = reset


# Global singleton
im_channel_registry = IMChannelRegistry()
