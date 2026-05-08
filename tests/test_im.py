"""Unit tests for the IM module: models, registry, @im_channel decorator, WebhookReceiver."""

from langdeep.core.im import (
    IMMessage,
    IMText,
    IMImage,
    IMInteractive,
    IMEvent,
    MessageType,
    PlatformType,
    IMChannelRegistry,
    im_channel_registry,
    im_channel,
    WebhookReceiver,
)
from langdeep.core.im.base import IMPlatformAdapter


def setup_function():
    """Reset singleton registries before each test."""
    from tests.conftest import clean_registries
    clean_registries()


# ── Message models ──────────────────────────────────────────────────────────


def test_im_message_basic():
    """Create a basic IMMessage with required fields."""
    msg = IMMessage(msg_id="m1", session_id="room1", platform=PlatformType.CUSTOM, content="hello")
    assert msg.msg_id == "m1"
    assert msg.session_id == "room1"
    assert msg.platform == PlatformType.CUSTOM
    assert msg.content == "hello"
    assert msg.msg_type == MessageType.TEXT


def test_im_text():
    """Create an IMText message."""
    msg = IMText(msg_id="t1", session_id="room1", platform=PlatformType.WECOM, content="text msg")
    assert msg.msg_type == MessageType.TEXT


def test_im_image():
    """Create an IMImage message."""
    msg = IMImage(
        msg_id="img1", session_id="room1", platform=PlatformType.DINGTALK,
        content="image", image_url="http://example.com/img.png", alt_text="photo",
    )
    assert msg.msg_type == MessageType.IMAGE
    assert msg.image_url == "http://example.com/img.png"


def test_im_interactive():
    """Create an IMInteractive message."""
    msg = IMInteractive(
        msg_id="act1", session_id="room1", platform=PlatformType.FEISHU,
        content="button", action="click", callback_id="cb_1",
    )
    assert msg.msg_type == MessageType.INTERACTIVE
    assert msg.action == "click"


def test_im_event():
    """Create an IMEvent (subscribe, etc.)."""
    msg = IMEvent(
        msg_id="ev1", session_id="room1", platform=PlatformType.SLACK,
        content="", event_type="subscribe", event_data={"user": "u1"},
    )
    assert msg.msg_type == MessageType.EVENT
    assert msg.event_type == "subscribe"


def test_platform_type_values():
    """All platform enum values exist."""
    assert PlatformType.WECOM.value == "wecom"
    assert PlatformType.DINGTALK.value == "dingtalk"
    assert PlatformType.FEISHU.value == "feishu"
    assert PlatformType.SLACK.value == "slack"
    assert PlatformType.CUSTOM.value == "custom"


# ── IMChannelRegistry ───────────────────────────────────────────────────────


def test_im_registry_register_and_dispatch():
    """Register a handler and dispatch an event."""
    reg = IMChannelRegistry()

    def my_handler(event):
        return f"Handled: {event.content}"

    reg.register_channel("ch1", my_handler, PlatformType.CUSTOM, "test channel")
    event = IMEvent(msg_id="m1", session_id="r1", platform=PlatformType.CUSTOM, content="ping")
    result = reg.dispatch(event)
    assert result == "Handled: ping"


def test_im_registry_dispatch_not_found():
    """Dispatch with no matching handler raises ConfigurationError."""
    from langdeep.core.errors import ConfigurationError
    reg = IMChannelRegistry()
    event = IMEvent(msg_id="m1", session_id="r1", platform=PlatformType.WECOM, content="x")
    try:
        reg.dispatch(event)
        assert False, "Expected ConfigurationError"
    except ConfigurationError:
        pass


def test_im_registry_list_channels():
    """list_channels returns registered channel info."""
    reg = IMChannelRegistry()
    reg.register_channel("ch_a", lambda e: "a", PlatformType.CUSTOM, "channel a")
    reg.register_channel("ch_b", lambda e: "b", PlatformType.WECOM, "channel b")
    channels = reg.list_channels()
    assert len(channels) == 2
    names = [c["name"] for c in channels]
    assert "ch_a" in names
    assert "ch_b" in names


def test_im_registry_connect_orchestrator():
    """connect_orchestrator routes IM messages through the orchestrator."""
    reg = IMChannelRegistry()

    class FakeOrch:
        def invoke(self, user_input="", context=None):
            return {"result": f"processed: {user_input}"}

    reg.connect_orchestrator(FakeOrch(), "my_channel", PlatformType.CUSTOM)

    event = IMEvent(msg_id="m1", session_id="r1", platform=PlatformType.CUSTOM, content="hello")
    result = reg.dispatch(event)
    assert "hello" in str(result)


# ── @im_channel decorator ───────────────────────────────────────────────────


def test_im_decorator():
    """@im_channel registers a handler."""

    @im_channel(name="helpdesk", platform="wecom", description="Help desk")
    def helpdesk(event):
        return f"help: {event.content}"

    channels = im_channel_registry.list_channels()
    assert any(c["name"] == "helpdesk" for c in channels)

    event = IMEvent(msg_id="m1", session_id="r1", platform=PlatformType.WECOM, content="help me")
    result = im_channel_registry.dispatch(event)
    assert result == "help: help me"


def test_im_decorator_no_name():
    """@im_channel without name uses function name."""

    @im_channel(platform="custom")
    def auto_name(event):
        return "ok"

    assert "auto_name" in [c["name"] for c in im_channel_registry.list_channels()]


# ── WebhookReceiver ─────────────────────────────────────────────────────────


def test_webhook_basic():
    """WebhookReceiver parses and dispatches correctly."""
    receiver = WebhookReceiver(verify_signature=False)

    @im_channel(name="wh", platform="custom")
    def wh_handler(event):
        return f"got: {event.content}"

    result = receiver.handle_request(
        b'{"content": "hello", "session_id": "room1"}',
        platform=PlatformType.CUSTOM,
    )
    assert result["content"] == "got: hello"


def test_webhook_invalid_json():
    """WebhookReceiver returns error on invalid JSON."""
    receiver = WebhookReceiver(verify_signature=False)
    result = receiver.handle_request(b"not json", platform=PlatformType.CUSTOM)
    assert "error" in result


def test_webhook_platform_detection_wecom():
    """Detect WeChat Work from User-Agent header."""
    receiver = WebhookReceiver(verify_signature=False)
    # No handler registered for wecom, should error
    result = receiver.handle_request(
        b'{"content": "test"}',
        headers={"User-Agent": "WeCom"},
    )
    assert "error" in result  # no handler, but we get an error, not a crash


def test_webhook_platform_detection_slack():
    """Detect Slack from User-Agent header."""
    receiver = WebhookReceiver(verify_signature=False)
    result = receiver.handle_request(
        b'{"content": "test"}',
        headers={"User-Agent": "SlackBot 2.0"},
    )
    assert "error" in result


def test_webhook_detect_via_x_platform_header():
    """Detect platform from X-Platform header."""
    receiver = WebhookReceiver(verify_signature=False)

    @im_channel(name="ding_handler", platform="dingtalk")
    def ding_handler(event):
        return f"ding: {event.content}"

    result = receiver.handle_request(
        b'{"content": "alert", "session_id": "r1"}',
        headers={"X-Platform": "dingtalk"},
    )
    assert "ding: alert" in result["content"]


def test_webhook_adapter_not_required_for_basic():
    """WebhookReceiver works without a platform adapter."""
    receiver = WebhookReceiver(verify_signature=False)

    @im_channel(name="simple", platform="custom")
    def simple(event):
        return event.content

    result = receiver.handle_request(
        b'{"content": "direct"}',
        platform=PlatformType.CUSTOM,
    )
    assert result["content"] == "direct"


def test_webhook_adapter_signature_rejection():
    """Registered adapters can reject invalid signatures before parsing."""

    class RejectingAdapter(IMPlatformAdapter):
        @property
        def platform(self):
            return PlatformType.CUSTOM

        def parse_payload(self, raw_data, headers=None):
            raise AssertionError("parse_payload should not run after signature failure")

        def validate_signature(self, raw_body, signature, timestamp=None):
            return False

        def format_response(self, messages):
            return {"messages": [m.content for m in messages]}

        def create_reply(self, original, text):
            return {"content": text}

    reg = IMChannelRegistry()
    reg.register_channel(
        "signed",
        lambda event: event.content,
        PlatformType.CUSTOM,
        adapter=RejectingAdapter(),
    )
    receiver = WebhookReceiver(registry=reg, verify_signature=True)

    result = receiver.handle_request(
        b'{"content": "hello"}',
        headers={"X-Signature": "bad", "X-Timestamp": "1"},
        platform=PlatformType.CUSTOM,
    )

    assert result == {"error": "Invalid signature"}


def test_webhook_adapter_parse_and_format_list_response():
    """Adapters parse payloads and format list responses."""

    class FormattingAdapter(IMPlatformAdapter):
        @property
        def platform(self):
            return PlatformType.CUSTOM

        def parse_payload(self, raw_data, headers=None):
            return IMEvent(
                msg_id="adapter-msg",
                session_id=raw_data["room"],
                platform=PlatformType.CUSTOM,
                content=raw_data["text"],
            )

        def validate_signature(self, raw_body, signature, timestamp=None):
            return True

        def format_response(self, messages):
            return {"formatted": [m.content for m in messages]}

        def create_reply(self, original, text):
            return {"content": text}

    adapter = FormattingAdapter()
    reg = IMChannelRegistry()

    def handler(event):
        assert event.msg_id == "adapter-msg"
        return [
            IMText(
                msg_id="reply",
                session_id=event.session_id,
                platform=event.platform,
                content=f"reply: {event.content}",
            )
        ]

    reg.register_channel("adapter", handler, PlatformType.CUSTOM, adapter=adapter)
    receiver = WebhookReceiver(registry=reg, verify_signature=True)

    result = receiver.handle_request(
        b'{"room": "r1", "text": "hello"}',
        headers={"X-Signature": "ok"},
        platform=PlatformType.CUSTOM,
    )

    assert result == {"formatted": ["reply: hello"]}


def test_webhook_adapter_formats_single_message_response():
    class SingleMessageAdapter(IMPlatformAdapter):
        @property
        def platform(self):
            return PlatformType.CUSTOM

        def parse_payload(self, raw_data, headers=None):
            return IMEvent(
                msg_id="incoming",
                session_id="room",
                platform=PlatformType.CUSTOM,
                content=raw_data["content"],
            )

        def validate_signature(self, raw_body, signature, timestamp=None):
            return True

        def format_response(self, messages):
            return {"single": messages[0].content}

        def create_reply(self, original, text):
            return {"content": text}

    reg = IMChannelRegistry()
    reg.register_channel(
        "single",
        lambda event: IMText(
            msg_id="reply",
            session_id=event.session_id,
            platform=event.platform,
            content=f"single: {event.content}",
        ),
        PlatformType.CUSTOM,
        adapter=SingleMessageAdapter(),
    )
    receiver = WebhookReceiver(registry=reg, verify_signature=True)

    result = receiver.handle_request(
        b'{"content": "hello"}',
        headers={"X-Signature": "ok"},
        platform=PlatformType.CUSTOM,
    )

    assert result == {"single": "single: hello"}


def test_webhook_unknown_x_platform_falls_back_to_default():
    receiver = WebhookReceiver(verify_signature=False)

    @im_channel(name="fallback", platform="custom")
    def fallback(event):
        return event.platform.value

    result = receiver.handle_request(
        b'{"content": "hello"}',
        headers={"X-Platform": "unknown"},
    )

    assert result["content"] == "custom"


def test_webhook_platform_detection_feishu_and_dingtalk():
    receiver = WebhookReceiver(verify_signature=False)
    assert receiver._detect_platform({"User-Agent": "Feishu Bot"}) == PlatformType.FEISHU
    assert receiver._detect_platform({"User-Agent": "DingTalk Callback"}) == PlatformType.DINGTALK


def test_im_platform_adapter_default_route():
    class RouteAdapter(IMPlatformAdapter):
        @property
        def platform(self):
            return PlatformType.SLACK

        def parse_payload(self, raw_data, headers=None):
            raise NotImplementedError

        def validate_signature(self, raw_body, signature, timestamp=None):
            return True

        def format_response(self, messages):
            return {}

        def create_reply(self, original, text):
            return {}

    assert RouteAdapter().get_webhook_route() == "/webhook/slack"
