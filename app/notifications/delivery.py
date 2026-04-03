"""
Notification Delivery Handlers

Subscribe to notifier topics and deliver messages via:
- Discord webhook
- Email (stub — actual SES delivery later)
- WebSocket push to connected clients

Call `register_delivery_handlers()` on app startup to wire subscriptions.
"""

import logging
from typing import Any, Optional

from app.infra.notify import Notification, get_notifier

logger = logging.getLogger(__name__)


# ===================================
# Discord Webhook Delivery
# ===================================

class DiscordWebhookDelivery:
    """Sends formatted notifications to a Discord webhook URL."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url

    async def handle(self, notification: Notification) -> None:
        """Format and send notification to Discord."""
        if not self.webhook_url:
            logger.debug("Discord webhook URL not configured, skipping delivery")
            return

        payload = notification.payload
        message_text = payload.get("message", str(payload))
        subject = notification.subject or notification.topic

        discord_payload = {
            "embeds": [
                {
                    "title": subject,
                    "description": message_text,
                    "color": self._color_for_topic(notification.topic, payload),
                    "footer": {"text": f"quant-ai | {notification.topic}"},
                    "timestamp": notification.timestamp.isoformat(),
                }
            ]
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.webhook_url, json=discord_payload)
                resp.raise_for_status()
            logger.info(f"Discord notification sent: {subject}")
        except Exception as e:
            logger.warning(f"Failed to send Discord notification: {e}")

    @staticmethod
    def _color_for_topic(topic: str, payload: dict[str, Any]) -> int:
        """Return Discord embed color based on notification type."""
        event = payload.get("event", payload.get("type", ""))
        if "failed" in event:
            return 0xFF0000  # Red
        if "completed" in event or "promoted" in event:
            return 0x00FF00  # Green
        if "started" in event:
            return 0x0099FF  # Blue
        if "overbought" in payload.get("condition", "") or "bearish" in event:
            return 0xFF6600  # Orange
        return 0x7289DA  # Discord blurple default


# ===================================
# Email Delivery (Stub)
# ===================================

class EmailDelivery:
    """Email delivery stub — logs message, actual SES integration later."""

    def __init__(self, recipient: Optional[str] = None):
        self.recipient = recipient

    async def handle(self, notification: Notification) -> None:
        """Log the email that would be sent."""
        payload = notification.payload
        subject = notification.subject or notification.topic
        message = payload.get("message", str(payload))

        logger.info(
            f"[EMAIL STUB] To: {self.recipient or 'not-configured'} | "
            f"Subject: {subject} | Body: {message}"
        )


# ===================================
# WebSocket Delivery
# ===================================

class WebSocketDelivery:
    """Pushes notifications to connected WebSocket clients."""

    def __init__(self) -> None:
        self._connections: set[Any] = set()

    def add_connection(self, ws: Any) -> None:
        """Register a WebSocket connection."""
        self._connections.add(ws)

    def remove_connection(self, ws: Any) -> None:
        """Unregister a WebSocket connection."""
        self._connections.discard(ws)

    async def handle(self, notification: Notification) -> None:
        """Broadcast notification to all connected WebSocket clients."""
        import json

        message = json.dumps({
            "topic": notification.topic,
            "subject": notification.subject,
            "payload": notification.payload,
            "timestamp": notification.timestamp.isoformat(),
        }, default=str)

        dead_connections = set()
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead_connections.add(ws)

        # Clean up dead connections
        self._connections -= dead_connections
        if dead_connections:
            logger.debug(f"Removed {len(dead_connections)} dead WebSocket connections")

        if self._connections:
            logger.debug(
                f"Broadcast notification to {len(self._connections)} WebSocket clients"
            )


# ===================================
# Singleton instances
# ===================================

discord_delivery = DiscordWebhookDelivery()
email_delivery = EmailDelivery()
websocket_delivery = WebSocketDelivery()


# ===================================
# Startup Registration
# ===================================

async def register_delivery_handlers(
    discord_webhook_url: Optional[str] = None,
    email_recipient: Optional[str] = None,
) -> None:
    """
    Register all delivery handlers to subscribe to notification topics.
    Call this on app startup.
    """
    notifier = get_notifier()

    # Configure delivery instances
    if discord_webhook_url:
        discord_delivery.webhook_url = discord_webhook_url
    if email_recipient:
        email_delivery.recipient = email_recipient

    # Subscribe to both topics
    for topic in ("alerts", "training"):
        await notifier.subscribe(topic, discord_delivery.handle)
        await notifier.subscribe(topic, email_delivery.handle)
        await notifier.subscribe(topic, websocket_delivery.handle)

    logger.info("Notification delivery handlers registered for topics: alerts, training")
