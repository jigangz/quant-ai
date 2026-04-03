from __future__ import annotations

"""
Training Lifecycle Notifications

Publishes training events via the configured notifier backend.
All publish calls are non-blocking for graceful degradation.

Topics:
- "training" — training lifecycle events (started, completed, failed, promoted)
"""

import asyncio
import logging
from typing import Any, Optional

from app.infra.notify import get_notifier

logger = logging.getLogger(__name__)

TRAINING_TOPIC = "training"


async def _safe_publish(topic: str, payload: dict[str, Any], subject: Optional[str] = None) -> None:
    """Publish a notification, logging and swallowing errors for graceful degradation."""
    try:
        notifier = get_notifier()
        await notifier.publish(topic, payload, subject=subject)
    except Exception as e:
        logger.warning(f"Failed to publish notification to {topic}: {e}")


# ===================================
# Training Lifecycle Events
# ===================================

async def publish_training_started(
    training_run_id: str,
    tickers: list[str],
    model_type: str,
) -> None:
    """Publish training started event."""
    payload = {
        "event": "training_started",
        "training_run_id": training_run_id,
        "tickers": tickers,
        "model_type": model_type,
        "message": f"Training started: {model_type} for {', '.join(tickers)}",
    }
    await _safe_publish(TRAINING_TOPIC, payload, subject="Training Started")


async def publish_training_completed(
    model_id: str,
    training_run_id: str,
    metrics: dict[str, Any],
) -> None:
    """Publish training completed event."""
    payload = {
        "event": "training_completed",
        "model_id": model_id,
        "training_run_id": training_run_id,
        "metrics": metrics,
        "message": f"Training completed: model {model_id}",
    }
    await _safe_publish(TRAINING_TOPIC, payload, subject="Training Completed")


async def publish_training_failed(
    training_run_id: str,
    error: str,
) -> None:
    """Publish training failed event."""
    payload = {
        "event": "training_failed",
        "training_run_id": training_run_id,
        "error": error,
        "message": f"Training failed: {error}",
    }
    await _safe_publish(TRAINING_TOPIC, payload, subject="Training Failed")


async def publish_model_promoted(
    model_id: str,
    model_name: str,
    promoted_by: str = "system",
) -> None:
    """Publish model promoted to production event."""
    payload = {
        "event": "model_promoted",
        "model_id": model_id,
        "model_name": model_name,
        "promoted_by": promoted_by,
        "message": f"Model promoted to production: {model_name} ({model_id})",
    }
    await _safe_publish(TRAINING_TOPIC, payload, subject="Model Promoted")
