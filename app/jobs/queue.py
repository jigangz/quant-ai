"""
Job Queue - Redis + RQ setup

Provides:
- Redis connection
- RQ queue instance
- Helper to enqueue training jobs
"""

import logging
import os
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job

from app.core.settings import settings

logger = logging.getLogger(__name__)

# Redis connection (lazy init)
_redis_conn: Redis | None = None
_queue: Queue | None = None


def get_redis() -> Redis:
    """Get Redis connection (singleton)."""
    global _redis_conn
    
    if _redis_conn is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        logger.info(f"Connecting to Redis: {redis_url}")
        _redis_conn = Redis.from_url(redis_url)
        # Test connection
        _redis_conn.ping()
        logger.info("Redis connected")
    
    return _redis_conn


def get_queue(name: str = "default") -> Queue:
    """Get RQ queue instance."""
    global _queue
    
    if _queue is None or _queue.name != name:
        _queue = Queue(name, connection=get_redis())
    
    return _queue


def enqueue_training_job(request_dict: dict[str, Any]) -> Job:
    """
    Enqueue a training job.
    
    Args:
        request_dict: Training request as dict (serializable)
        
    Returns:
        RQ Job instance with job.id
    """
    from app.jobs.tasks import run_training_task
    
    queue = get_queue("training")
    
    job = queue.enqueue(
        run_training_task,
        request_dict,
        job_timeout="30m",  # Max 30 minutes
        result_ttl=86400,   # Keep result for 24h
        failure_ttl=86400,  # Keep failures for 24h
    )
    
    logger.info(f"Enqueued training job: {job.id}")
    return job


def get_job(job_id: str) -> Job | None:
    """Get job by ID."""
    try:
        return Job.fetch(job_id, connection=get_redis())
    except Exception as e:
        logger.warning(f"Job not found: {job_id} - {e}")
        return None


def get_job_status(job_id: str) -> dict[str, Any]:
    """
    Get job status and result.
    
    Returns:
        {
            "job_id": str,
            "status": "queued" | "started" | "finished" | "failed",
            "result": Any (if finished),
            "error": str (if failed),
            "enqueued_at": datetime,
            "started_at": datetime,
            "ended_at": datetime,
        }
    """
    job = get_job(job_id)
    
    if job is None:
        return {
            "job_id": job_id,
            "status": "not_found",
            "error": "Job not found",
        }
    
    status_map = {
        "queued": "pending",
        "started": "running",
        "finished": "completed",
        "failed": "failed",
        "deferred": "pending",
        "scheduled": "pending",
        "stopped": "failed",
        "canceled": "failed",
    }
    
    result = {
        "job_id": job_id,
        "status": status_map.get(job.get_status(), job.get_status()),
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }
    
    # Add result or error
    if job.is_finished:
        result["result"] = job.result
    elif job.is_failed:
        result["error"] = str(job.exc_info) if job.exc_info else "Unknown error"
    
    return result
