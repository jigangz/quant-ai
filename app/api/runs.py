"""
Training Runs API

GET /runs - List training runs
GET /runs/{run_id} - Get run status and result
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.jobs.queue import get_job_status

logger = logging.getLogger(__name__)
router = APIRouter()


# ===================================
# Response Schemas
# ===================================
class RunStatus(BaseModel):
    """Status of a training run."""
    
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "not_found"]
    
    # Timing
    enqueued_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    
    # Result (if completed)
    result: dict[str, Any] | None = None
    
    # Error (if failed)
    error: str | None = None


# ===================================
# GET /runs/{run_id}
# ===================================
@router.get("/runs/{run_id}", response_model=RunStatus)
def get_run_status(run_id: str):
    """
    Get the status of a training run.
    
    Status values:
    - pending: Job is queued, waiting for worker
    - running: Job is being processed
    - completed: Job finished successfully
    - failed: Job failed with error
    - not_found: Job ID not found
    """
    status = get_job_status(run_id)
    
    return RunStatus(
        job_id=status["job_id"],
        status=status["status"],
        enqueued_at=status.get("enqueued_at"),
        started_at=status.get("started_at"),
        ended_at=status.get("ended_at"),
        result=status.get("result") if status["status"] == "completed" else None,
        error=status.get("error"),
    )


# ===================================
# GET /runs (list recent runs)
# ===================================
class RunListResponse(BaseModel):
    """List of training runs."""
    
    runs: list[RunStatus]
    total: int


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
):
    """
    List recent training runs.
    
    Note: This queries the database training_runs table, not Redis.
    For real-time job status, use GET /runs/{run_id}.
    """
    from app.db.model_registry import get_model_registry
    
    registry = get_model_registry()
    runs = registry.list_runs(limit=limit)
    
    # Convert to RunStatus format
    run_statuses = []
    for run in runs:
        run_status = "completed" if run.success else "failed" if run.error else "pending"
        run_statuses.append(
            RunStatus(
                job_id=run.id,
                status=run_status,
                result={
                    "model_id": run.model_id,
                    "metrics": run.metrics,
                    "training_time_seconds": run.training_time_seconds,
                } if run.success else None,
                error=run.error,
            )
        )
    
    return RunListResponse(runs=run_statuses, total=len(run_statuses))
