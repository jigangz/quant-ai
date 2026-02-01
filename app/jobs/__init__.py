"""
Jobs Module - Async task queue for training and other long-running tasks.

Uses Redis + RQ (Redis Queue) for job management.
Includes experiment tracking for reproducibility.
"""

from app.jobs.queue import get_queue, enqueue_training_job
from app.jobs.tasks import run_training_task
from app.jobs.experiment import (
    get_git_info,
    compute_data_hash,
    compute_config_hash,
    collect_experiment_metadata,
)

__all__ = [
    "get_queue",
    "enqueue_training_job",
    "run_training_task",
    "get_git_info",
    "compute_data_hash",
    "compute_config_hash",
    "collect_experiment_metadata",
]
