#!/usr/bin/env python3
"""
RQ Worker - Background job processor

Usage:
    python -m app.jobs.worker
    
Or via Docker:
    docker-compose up worker
"""

import logging
import os

from redis import Redis
from rq import Worker, Connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_worker():
    """Start the RQ worker."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    logger.info(f"Connecting to Redis: {redis_url}")
    
    conn = Redis.from_url(redis_url)
    
    # Listen on these queues (priority order)
    queues = ["training", "default"]
    
    with Connection(conn):
        worker = Worker(queues)
        logger.info(f"Worker started, listening on queues: {queues}")
        worker.work()


if __name__ == "__main__":
    run_worker()
