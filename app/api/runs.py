"""
Training Runs API - Experiment Tracking

GET /runs - List training runs with experiment metadata
GET /runs/{run_id} - Get full run details
GET /runs/{run_id}/reproduce - Get reproduction command
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
class ExperimentInfo(BaseModel):
    """Experiment reproducibility info."""
    
    git_sha: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    data_hash: str | None = None
    config_hash: str | None = None
    python_version: str | None = None
    packages: dict[str, str] = {}


class DataInfo(BaseModel):
    """Training data info."""
    
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_date_range: tuple[str, str] | None = None
    val_date_range: tuple[str, str] | None = None
    test_date_range: tuple[str, str] | None = None


class RunStatus(BaseModel):
    """Status of a training run with experiment tracking."""
    
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "not_found"]
    
    # Timing
    enqueued_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    training_time_seconds: float = 0.0
    
    # Config
    tickers: list[str] = []
    model_type: str = ""
    feature_groups: list[str] = []
    model_params: dict[str, Any] = {}
    horizon_days: int = 5
    
    # Experiment tracking
    experiment: ExperimentInfo | None = None
    data_info: DataInfo | None = None
    
    # Result (if completed)
    model_id: str | None = None
    metrics: dict[str, float] = {}
    
    # Error (if failed)
    error: str | None = None


class RunListResponse(BaseModel):
    """List of training runs."""
    
    runs: list[RunStatus]
    total: int


class ReproduceCommand(BaseModel):
    """Command to reproduce an experiment."""
    
    run_id: str
    git_checkout: str | None = None
    curl_command: str
    python_command: str
    notes: list[str] = []


# ===================================
# GET /runs/{run_id}
# ===================================
@router.get("/runs/{run_id}", response_model=RunStatus)
def get_run_status(run_id: str):
    """
    Get the status of a training run with full experiment metadata.
    
    Includes:
    - Job status (pending/running/completed/failed)
    - Git SHA, branch, dirty flag
    - Config hash for dedup
    - Data split info
    - Python + package versions
    """
    # First check RQ job status
    job_status = get_job_status(run_id)
    
    # Also check database for persisted run record
    from app.db.model_registry import get_model_registry
    registry = get_model_registry()
    run_record = registry.get_run(run_id)
    
    if run_record:
        # Use database record (more complete)
        return RunStatus(
            job_id=run_id,
            status="completed" if run_record.success else "failed" if run_record.error else "pending",
            training_time_seconds=run_record.training_time_seconds,
            tickers=run_record.tickers,
            model_type=run_record.model_type,
            feature_groups=run_record.feature_groups,
            model_params=run_record.model_params,
            horizon_days=run_record.horizon_days,
            experiment=ExperimentInfo(
                git_sha=run_record.git_sha,
                git_branch=run_record.git_branch,
                git_dirty=run_record.git_dirty,
                data_hash=run_record.data_hash,
                config_hash=run_record.config_hash,
                python_version=run_record.python_version,
                packages=run_record.packages,
            ),
            data_info=DataInfo(
                train_samples=run_record.train_samples,
                val_samples=run_record.val_samples,
                test_samples=run_record.test_samples,
                train_date_range=run_record.train_date_range,
                val_date_range=run_record.val_date_range,
                test_date_range=run_record.test_date_range,
            ),
            model_id=run_record.model_id,
            metrics=run_record.metrics,
            error=run_record.error,
        )
    
    # Fall back to RQ job status
    return RunStatus(
        job_id=job_status["job_id"],
        status=job_status["status"],
        enqueued_at=job_status.get("enqueued_at"),
        started_at=job_status.get("started_at"),
        ended_at=job_status.get("ended_at"),
        model_id=job_status.get("result", {}).get("model_id") if job_status.get("result") else None,
        metrics=job_status.get("result", {}).get("metrics", {}) if job_status.get("result") else {},
        error=job_status.get("error"),
    )


# ===================================
# GET /runs (list)
# ===================================
@router.get("/runs", response_model=RunListResponse)
def list_runs(
    status: str | None = Query(None, description="Filter by status"),
    model_type: str | None = Query(None, description="Filter by model type"),
    ticker: str | None = Query(None, description="Filter by ticker"),
    limit: int = Query(20, ge=1, le=100),
):
    """
    List recent training runs with experiment metadata.
    
    Useful for:
    - Comparing experiments
    - Finding runs to reproduce
    - Tracking experiment history
    """
    from app.db.model_registry import get_model_registry
    
    registry = get_model_registry()
    runs = registry.list_runs(limit=limit)
    
    # Apply filters
    if model_type:
        runs = [r for r in runs if r.model_type == model_type]
    if ticker:
        runs = [r for r in runs if ticker in r.tickers]
    
    # Convert to response
    run_statuses = []
    for run in runs:
        run_status = "completed" if run.success else "failed" if run.error else "pending"
        
        # Skip if status filter doesn't match
        if status and run_status != status:
            continue
            
        run_statuses.append(
            RunStatus(
                job_id=run.id,
                status=run_status,
                training_time_seconds=run.training_time_seconds,
                tickers=run.tickers,
                model_type=run.model_type,
                feature_groups=run.feature_groups,
                horizon_days=run.horizon_days,
                experiment=ExperimentInfo(
                    git_sha=run.git_sha,
                    git_branch=run.git_branch,
                    git_dirty=run.git_dirty,
                    config_hash=run.config_hash,
                ),
                data_info=DataInfo(
                    train_samples=run.train_samples,
                    val_samples=run.val_samples,
                    test_samples=run.test_samples,
                ),
                model_id=run.model_id,
                metrics=run.metrics,
                error=run.error,
            )
        )
    
    return RunListResponse(runs=run_statuses, total=len(run_statuses))


# ===================================
# GET /runs/{run_id}/reproduce
# ===================================
@router.get("/runs/{run_id}/reproduce", response_model=ReproduceCommand)
def get_reproduce_command(run_id: str):
    """
    Get commands to reproduce an experiment.
    
    Returns:
    - Git checkout command (if git_sha available)
    - curl command to re-run training
    - Python command for local training
    - Notes about reproducibility
    """
    from app.db.model_registry import get_model_registry
    import json
    
    registry = get_model_registry()
    run = registry.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    
    notes = []
    
    # Git checkout command
    git_checkout = None
    if run.git_sha:
        git_checkout = f"git checkout {run.git_sha}"
        if run.git_dirty:
            notes.append("⚠️ Original run had uncommitted changes - may not be fully reproducible")
    else:
        notes.append("⚠️ No git SHA recorded - cannot guarantee exact code version")
    
    # Build request body
    request_body = {
        "tickers": run.tickers,
        "model_type": run.model_type,
        "feature_groups": run.feature_groups,
        "model_params": run.model_params,
        "horizon_days": run.horizon_days,
    }
    if run.start_date:
        request_body["start_date"] = run.start_date
    if run.end_date:
        request_body["end_date"] = run.end_date
    
    # Curl command
    request_json = json.dumps(request_body)
    curl_command = f"""curl -X POST http://localhost:8000/train \\
  -H "Content-Type: application/json" \\
  -d '{request_json}'"""
    
    # Python command
    python_command = f"""from app.services.training_service import TrainRequest, TrainingService

request = TrainRequest(**{request_body})
service = TrainingService()
result = service.train(request)
print(result)"""
    
    # Add notes about data
    if run.train_samples > 0:
        notes.append(f"Original: {run.train_samples} train / {run.val_samples} val / {run.test_samples} test samples")
    
    if run.packages:
        notes.append(f"Key packages: {', '.join(f'{k}=={v}' for k, v in list(run.packages.items())[:3])}")
    
    return ReproduceCommand(
        run_id=run_id,
        git_checkout=git_checkout,
        curl_command=curl_command,
        python_command=python_command,
        notes=notes,
    )
