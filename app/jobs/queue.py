"""
Job Queue - Redis + RQ setup (optional)

Falls back to sync execution if Redis is not available.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Try to import Redis/RQ
try:
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None
    Queue = None
    Job = None
    logger.info("Redis/RQ not installed, async jobs disabled")

# Redis connection (lazy init)
_redis_conn = None
_queue = None


def get_redis():
    """Get Redis connection (singleton)."""
    global _redis_conn
    
    if not REDIS_AVAILABLE:
        raise RuntimeError("Redis not installed")
    
    if _redis_conn is None:
        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            raise RuntimeError("REDIS_URL not configured")
        
        logger.info(f"Connecting to Redis: {redis_url}")
        _redis_conn = Redis.from_url(redis_url)
        _redis_conn.ping()
        logger.info("Redis connected")
    
    return _redis_conn


def get_queue(name: str = "default"):
    """Get RQ queue instance."""
    global _queue
    
    if not REDIS_AVAILABLE:
        raise RuntimeError("Redis not installed")
    
    if _queue is None or _queue.name != name:
        _queue = Queue(name, connection=get_redis())
    
    return _queue


def is_redis_available() -> bool:
    """Check if Redis is available and configured."""
    if not REDIS_AVAILABLE:
        return False
    
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return False
    
    try:
        get_redis().ping()
        return True
    except Exception:
        return False


def enqueue_training_job(request_dict: dict[str, Any]):
    """Enqueue a training job."""
    if not is_redis_available():
        raise RuntimeError("Redis not available for async jobs")
    
    from app.jobs.tasks import run_training_task
    
    queue = get_queue("training")
    
    job = queue.enqueue(
        run_training_task,
        request_dict,
        job_timeout="30m",
        result_ttl=86400,
        failure_ttl=86400,
    )
    
    logger.info(f"Enqueued training job: {job.id}")
    return job


def get_job(job_id: str):
    """Get job by ID."""
    if not REDIS_AVAILABLE:
        return None
    
    try:
        return Job.fetch(job_id, connection=get_redis())
    except Exception as e:
        logger.warning(f"Job not found: {job_id} - {e}")
        return None


def get_job_status(job_id: str) -> dict[str, Any]:
    """Get job status and result."""
    if not REDIS_AVAILABLE:
        return {
            "job_id": job_id,
            "status": "not_found",
            "error": "Redis not available",
        }
    
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
    
    if job.is_finished:
        result["result"] = job.result
    elif job.is_failed:
        result["error"] = str(job.exc_info) if job.exc_info else "Unknown error"
    
    return result
