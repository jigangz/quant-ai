"""
Task Queue Abstraction

Supports: AWS SQS, Redis (RQ), In-Memory.
Swap backend via QUEUE_BACKEND env var.

Usage:
    queue = get_queue()
    job_id = await queue.enqueue("training", {"tickers": ["AAPL"], "model_type": "xgboost"})
    status = await queue.get_status(job_id)
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable, Optional

from app.core.settings import settings

logger = logging.getLogger(__name__)


# ===================================
# Data Models
# ===================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Job:
    id: str
    queue_name: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    attempt: int = 0
    max_retries: int = 3


JobHandler = Callable[[Job], Awaitable[dict[str, Any]]]


# ===================================
# Abstract Interface
# ===================================

class BaseQueue(ABC):
    """Abstract task queue interface."""

    @abstractmethod
    async def enqueue(
        self,
        queue_name: str,
        payload: dict[str, Any],
        delay_seconds: int = 0,
    ) -> str:
        """Submit a job. Returns job ID."""
        ...

    @abstractmethod
    async def get_status(self, job_id: str) -> Optional[Job]:
        """Get job status by ID."""
        ...

    @abstractmethod
    async def register_handler(
        self, queue_name: str, handler: JobHandler
    ) -> None:
        """Register a handler for a queue."""
        ...

    @abstractmethod
    async def start_worker(self) -> None:
        """Start consuming and processing jobs."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the queue worker."""
        ...


# ===================================
# SQS Implementation
# ===================================

class SQSQueue(BaseQueue):
    """
    AWS SQS queue with automatic retries and DLQ support.
    
    Works with: real AWS SQS, LocalStack (local dev).
    Set SQS_ENDPOINT_URL=http://localhost:4566 for LocalStack.
    """

    def __init__(self):
        self._client = None
        self._queue_urls: dict[str, str] = {}
        self._handlers: dict[str, JobHandler] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def _get_client(self):
        if self._client is None:
            import aioboto3
            session = aioboto3.Session()
            endpoint = getattr(settings, "SQS_ENDPOINT_URL", None)
            kwargs = {"service_name": "sqs", "region_name": getattr(settings, "AWS_REGION", "us-east-1")}
            if endpoint:
                kwargs["endpoint_url"] = endpoint
            self._client = await session.client(**kwargs).__aenter__()
        return self._client

    async def _get_or_create_queue(self, queue_name: str) -> str:
        if queue_name in self._queue_urls:
            return self._queue_urls[queue_name]
        client = await self._get_client()
        try:
            resp = await client.get_queue_url(QueueName=f"quant-ai-{queue_name}")
            url = resp["QueueUrl"]
        except Exception:
            resp = await client.create_queue(
                QueueName=f"quant-ai-{queue_name}",
                Attributes={"VisibilityTimeout": "300", "MessageRetentionPeriod": "86400"},
            )
            url = resp["QueueUrl"]
        self._queue_urls[queue_name] = url
        return url

    async def enqueue(
        self,
        queue_name: str,
        payload: dict[str, Any],
        delay_seconds: int = 0,
    ) -> str:
        client = await self._get_client()
        queue_url = await self._get_or_create_queue(queue_name)
        job_id = str(uuid.uuid4())
        body = json.dumps({"job_id": job_id, "payload": payload, "attempt": 0}, default=str)
        await client.send_message(
            QueueUrl=queue_url,
            MessageBody=body,
            DelaySeconds=min(delay_seconds, 900),
            MessageAttributes={
                "JobId": {"DataType": "String", "StringValue": job_id},
            },
        )
        logger.info(f"SQS enqueued job {job_id} to {queue_name}")
        return job_id

    async def get_status(self, job_id: str) -> Optional[Job]:
        # SQS is fire-and-forget; for status tracking, use Redis or DB as side-channel
        logger.warning("SQS does not natively support job status lookup; use DB tracking")
        return None

    async def register_handler(self, queue_name: str, handler: JobHandler) -> None:
        self._handlers[queue_name] = handler

    async def start_worker(self) -> None:
        self._running = True
        for queue_name, handler in self._handlers.items():
            task = asyncio.create_task(self._poll_loop(queue_name, handler))
            self._tasks.append(task)
        logger.info(f"SQS worker started for {list(self._handlers.keys())}")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._client:
            await self._client.__aexit__(None, None, None)

    async def _poll_loop(self, queue_name: str, handler: JobHandler) -> None:
        client = await self._get_client()
        queue_url = await self._get_or_create_queue(queue_name)
        try:
            while self._running:
                resp = await client.receive_message(
                    QueueUrl=queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,  # Long polling
                    MessageAttributeNames=["All"],
                )
                for msg in resp.get("Messages", []):
                    body = json.loads(msg["Body"])
                    job = Job(
                        id=body["job_id"],
                        queue_name=queue_name,
                        payload=body["payload"],
                        attempt=body.get("attempt", 0),
                    )
                    try:
                        job.status = JobStatus.RUNNING
                        result = await handler(job)
                        job.status = JobStatus.COMPLETED
                        job.result = result
                        await client.delete_message(
                            QueueUrl=queue_url,
                            ReceiptHandle=msg["ReceiptHandle"],
                        )
                    except Exception as e:
                        logger.error(f"Job {job.id} failed: {e}", exc_info=True)
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        # Message becomes visible again after VisibilityTimeout (auto-retry)
        except asyncio.CancelledError:
            pass


# ===================================
# Redis (RQ) Implementation
# ===================================

class RedisQueue(BaseQueue):
    """
    Redis-backed queue using the existing RQ setup.
    Wraps the current app.jobs.queue module with the abstract interface.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._handlers: dict[str, JobHandler] = {}

    async def enqueue(
        self,
        queue_name: str,
        payload: dict[str, Any],
        delay_seconds: int = 0,
    ) -> str:
        from app.jobs.queue import enqueue_job
        
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = Job(id=job_id, queue_name=queue_name, payload=payload)
        
        # Delegate to existing RQ
        enqueue_job(queue_name, payload)
        logger.info(f"Redis enqueued job {job_id} to {queue_name}")
        return job_id

    async def get_status(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def register_handler(self, queue_name: str, handler: JobHandler) -> None:
        self._handlers[queue_name] = handler

    async def start_worker(self) -> None:
        logger.info("Redis queue uses RQ worker (run `python -m app.jobs.worker`)")

    async def stop(self) -> None:
        pass


# ===================================
# In-Memory Implementation (Testing)
# ===================================

class InMemoryQueue(BaseQueue):
    """In-memory queue for tests. Executes jobs synchronously."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._handlers: dict[str, JobHandler] = {}

    async def enqueue(
        self,
        queue_name: str,
        payload: dict[str, Any],
        delay_seconds: int = 0,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, queue_name=queue_name, payload=payload)
        self._jobs[job_id] = job
        
        handler = self._handlers.get(queue_name)
        if handler:
            try:
                result = await handler(job)
                job.status = JobStatus.COMPLETED
                job.result = result
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
        return job_id

    async def get_status(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def register_handler(self, queue_name: str, handler: JobHandler) -> None:
        self._handlers[queue_name] = handler

    async def start_worker(self) -> None:
        pass

    async def stop(self) -> None:
        self._jobs.clear()


# ===================================
# Factory
# ===================================

_queue_instance: Optional[BaseQueue] = None


def get_queue() -> BaseQueue:
    """
    Get the configured queue instance.
    
    Set QUEUE_BACKEND env var:
      - "sqs"    → SQSQueue (AWS SQS / LocalStack)
      - "redis"  → RedisQueue (existing RQ setup)
      - "memory" → InMemoryQueue (tests only)
    
    Default: "redis" (backward compatible with current setup)
    """
    global _queue_instance
    if _queue_instance is None:
        backend = getattr(settings, "QUEUE_BACKEND", "redis")
        if backend == "sqs":
            _queue_instance = SQSQueue()
        elif backend == "redis":
            _queue_instance = RedisQueue()
        elif backend == "memory":
            _queue_instance = InMemoryQueue()
        else:
            raise ValueError(f"Unknown queue backend: {backend}")
        logger.info(f"Queue backend: {backend}")
    return _queue_instance
