"""
Jobs Module - Async task queue for training and other long-running tasks.

Uses Redis + RQ (Redis Queue) for job management.
"""

from app.jobs.queue import get_queue, enqueue_training_job
from app.jobs.tasks import run_training_task

__all__ = ["get_queue", "enqueue_training_job", "run_training_task"]
