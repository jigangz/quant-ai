#!/usr/bin/env python3
from __future__ import annotations
"""
SQS Worker — Standalone async worker that polls SQS and runs training tasks.

Usage:
    python -m app.workers.sqs_worker

Requires QUEUE_BACKEND=sqs and SQS_ENDPOINT_URL set (for LocalStack).
"""

import asyncio
import logging
import signal
import sys
from typing import Any

from app.infra.queue import Job, JobStatus, SQSQueue, get_queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def handle_training_job(job: Job) -> dict[str, Any]:
    """Execute a training job from the SQS queue."""
    from app.jobs.tasks import run_training_task

    logger.info(f"Processing training job {job.id} (attempt {job.attempt})")

    # run_training_task is sync — run in thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_training_task, job.payload)

    if not result.get("success"):
        raise RuntimeError(result.get("error", "Training failed"))

    # Publish completion notification (best-effort)
    try:
        from app.notifications.training_events import publish_training_completed
        await publish_training_completed(
            model_id=result.get("model_id", ""),
            training_run_id=result.get("training_run_id", ""),
            metrics=result.get("metrics", {}),
        )
    except Exception as e:
        logger.warning(f"Failed to publish training completion notification: {e}")

    logger.info(f"Training job {job.id} completed: model_id={result.get('model_id')}")
    return result


async def handle_training_failure(job: Job) -> None:
    """Publish failure notification when a training job fails."""
    try:
        from app.notifications.training_events import publish_training_failed
        await publish_training_failed(
            training_run_id=job.id,
            error=job.error or "Unknown error",
        )
    except Exception as e:
        logger.warning(f"Failed to publish training failure notification: {e}")


async def main() -> None:
    """Start the SQS worker."""
    queue = get_queue()

    if not isinstance(queue, SQSQueue):
        logger.error(
            f"SQS worker requires QUEUE_BACKEND=sqs, got {type(queue).__name__}. "
            "Set QUEUE_BACKEND=sqs in your environment."
        )
        sys.exit(1)

    # Register training handler
    await queue.register_handler("training", handle_training_job)

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("SQS worker starting...")
    await queue.start_worker()
    logger.info("SQS worker running. Waiting for jobs...")

    await stop_event.wait()

    logger.info("Stopping SQS worker...")
    await queue.stop()
    logger.info("SQS worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
