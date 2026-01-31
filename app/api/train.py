"""
Training API - POST /train

Synchronous training endpoint that:
1. Creates a training run record
2. Runs training via TrainingService
3. Registers the model in the registry
4. Returns model_id and metrics
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.db.model_registry import (
    ModelRecord,
    TrainingRunRecord,
    get_model_registry,
)
from app.services.training_service import TrainRequest, TrainingService

logger = logging.getLogger(__name__)
router = APIRouter()


# ===================================
# Request/Response Schemas
# ===================================
class TrainResponse(ModelRecord):
    """Response for POST /train - extends ModelRecord with training info."""

    training_run_id: str
    training_time_seconds: float = 0.0


# ===================================
# POST /train
# ===================================
@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    """
    Train a model synchronously.

    - Creates a training run record
    - Trains the model using TrainingService
    - Registers the model in the registry
    - Returns model metadata with model_id

    Note: This is a synchronous endpoint. For large datasets,
    consider using a background job queue.
    """
    registry = get_model_registry()

    # 1. Create training run record
    run_record = TrainingRunRecord(
        tickers=request.tickers,
        model_type=request.model_type,
        feature_groups=request.feature_groups,
        model_params=request.model_params,
        horizon_days=request.horizon_days,
        started_at=datetime.utcnow(),
    )
    registry.insert_run(run_record)
    logger.info(f"Training run started: {run_record.id}")

    # 2. Run training
    service = TrainingService()
    result = service.train(request)

    # 3. Update run record with results
    run_record.success = result.success
    run_record.error = result.error
    run_record.metrics = result.metrics
    run_record.training_time_seconds = result.training_time_seconds
    run_record.completed_at = datetime.utcnow()

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
        name=request.model_name or f"{request.model_type}_{result.model_id}",
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

    # 5. Link run to model
    run_record.model_id = model_record.id
    # Note: In a real implementation, we'd update the run record in the DB

    # 6. Return response
    return TrainResponse(
        **model_record.model_dump(),
        training_run_id=run_record.id,
        training_time_seconds=result.training_time_seconds,
    )
