"""
Training API - POST /train

Supports two modes:
1. Async (default): Enqueue job → return run_id → poll GET /runs/{run_id}
2. Sync: Set async=false → wait for result → return model

Use async=false for quick tests, async=true for production.
"""

import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db.model_registry import (
    ModelRecord,
    TrainingRunRecord,
    get_model_registry,
)
from app.services.training_service import TrainRequest, TrainingService

logger = logging.getLogger(__name__)
router = APIRouter()


# ===================================
# Response Schemas
# ===================================
class TrainAsyncResponse(BaseModel):
    """Response for async training - just the run_id."""
    
    run_id: str
    status: Literal["pending", "queued"]
    message: str = "Training job queued. Poll GET /runs/{run_id} for status."


class TrainSyncResponse(ModelRecord):
    """Response for sync training - full model record."""

    training_run_id: str
    training_time_seconds: float = 0.0


# ===================================
# POST /train
# ===================================
@router.post("/train")
def train_model(
    request: TrainRequest,
    sync: bool = Query(
        default=False,
        alias="async",
        description="If false (default), run async and return run_id. If true, wait for result.",
    ),
):
    """
    Train a model.
    
    **Async mode (default):**
    - Enqueues a training job
    - Returns immediately with run_id
    - Poll GET /runs/{run_id} to check status
    
    **Sync mode (?async=false):**
    - Waits for training to complete
    - Returns full model record with metrics
    - Use for quick tests only
    
    Example:
    ```bash
    # Async (recommended)
    curl -X POST /train -d '{"tickers": ["AAPL"]}'
    # Returns: {"run_id": "abc123", "status": "pending"}
    
    # Check status
    curl /runs/abc123
    
    # Sync (for testing)
    curl -X POST "/train?async=false" -d '{"tickers": ["AAPL"]}'
    # Returns: full model record
    ```
    """
    # Note: Query param is "async" but Python uses "sync" (inverted)
    # FastAPI maps async=true → sync=True, async=false → sync=False
    run_async = not sync
    
    # Check if Redis is available for async mode
    redis_available = _check_redis()
    
    if run_async and not redis_available:
        logger.warning("Redis not available, falling back to sync mode")
        run_async = False
    
    if run_async:
        return _train_async(request)
    else:
        return _train_sync(request)


def _check_redis() -> bool:
    """Check if Redis is available."""
    try:
        from app.jobs.queue import is_redis_available
        return is_redis_available()
    except Exception as e:
        logger.debug(f"Redis not available: {e}")
        return False


def _train_async(request: TrainRequest) -> TrainAsyncResponse:
    """Enqueue training job and return immediately."""
    from app.jobs.queue import enqueue_training_job
    
    # Convert request to dict for serialization
    request_dict = request.model_dump()
    
    # Enqueue job
    job = enqueue_training_job(request_dict)
    
    logger.info(f"Training job enqueued: {job.id}")
    
    return TrainAsyncResponse(
        run_id=job.id,
        status="queued",
        message=f"Training job queued. Poll GET /runs/{job.id} for status.",
    )


def _train_sync(request: TrainRequest) -> TrainSyncResponse:
    """Run training synchronously and return result."""
    registry = get_model_registry()

    # 1. Create training run record
    run_record = TrainingRunRecord(
        tickers=request.tickers,
        model_type=request.model_type,
        feature_groups=request.feature_groups,
        model_params=request.model_params,
        horizon_days=request.horizon_days,
    )
    
    start_time = datetime.utcnow()
    logger.info(f"Training run started (sync): {run_record.id}")

    # 2. Run training
    service = TrainingService()
    result = service.train(request)
    
    # Calculate time
    end_time = datetime.utcnow()
    training_time = (end_time - start_time).total_seconds()

    # 3. Handle failure
    if not result.success:
        logger.error(f"Training failed: {result.error}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Training failed",
                "message": result.error,
                "training_run_id": run_record.id,
            },
        )

    # 4. Register the model
    model_record = ModelRecord(
        id=result.model_id,
        name=request.model_name or f"{request.model_type}_{result.model_id[:8]}",
        model_type=result.model_type,
        tickers=result.tickers,
        feature_groups=result.feature_groups,
        feature_names=result.feature_names,
        n_features=result.n_features,
        train_samples=result.train_samples,
        val_samples=result.val_samples,
        test_samples=result.test_samples,
        train_date_range=result.train_date_range,
        metrics=result.metrics,
        artifact_path=result.model_path,
        storage_backend="local",
        status="active",
    )
    registry.insert_model(model_record)
    logger.info(f"Model registered: {model_record.id}")

    # 5. Update run record
    run_record.success = True
    run_record.model_id = model_record.id
    run_record.metrics = result.metrics
    run_record.training_time_seconds = training_time
    registry.insert_run(run_record)

    # 6. Return response
    return TrainSyncResponse(
        **model_record.model_dump(),
        training_run_id=run_record.id,
        training_time_seconds=training_time,
    )
