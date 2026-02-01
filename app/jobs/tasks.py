"""
Training Task - Actual training logic run by worker

This is executed in the worker process, not the API process.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def run_training_task(request_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Execute training job.
    
    Args:
        request_dict: Training request parameters
        
    Returns:
        Training result with model_id, metrics, etc.
    """
    from app.services.training_service import TrainRequest, TrainingService
    from app.db.model_registry import (
        ModelRecord,
        TrainingRunRecord,
        get_model_registry,
    )
    
    logger.info(f"Starting training task: {request_dict}")
    start_time = datetime.utcnow()
    
    try:
        # Parse request
        request = TrainRequest(**request_dict)
        
        # Create training run record
        registry = get_model_registry()
        run_record = TrainingRunRecord(
            tickers=request.tickers,
            model_type=request.model_type,
            feature_groups=request.feature_groups,
            model_params=request.model_params,
            horizon_days=request.horizon_days,
        )
        registry.insert_run(run_record)
        
        # Run training
        service = TrainingService()
        result = service.train(request)
        
        # Calculate training time
        end_time = datetime.utcnow()
        training_time = (end_time - start_time).total_seconds()
        
        if not result.success:
            # Update run record with failure
            run_record.success = False
            run_record.error = result.error
            run_record.training_time_seconds = training_time
            registry.insert_run(run_record)
            
            return {
                "success": False,
                "error": result.error,
                "training_run_id": run_record.id,
                "training_time_seconds": training_time,
            }
        
        # Create model record
        model_record = ModelRecord(
            name=result.model_id or f"{request.model_type}_{'_'.join(request.tickers)}",
            model_type=request.model_type,
            tickers=request.tickers,
            feature_groups=request.feature_groups,
            feature_names=result.feature_names,
            n_features=result.n_features,
            train_samples=result.train_samples,
            val_samples=result.val_samples,
            test_samples=result.test_samples,
            train_date_range=result.train_date_range,
            metrics=result.metrics,
            artifact_path=result.model_path,
            storage_backend="local",
        )
        
        registry.insert_model(model_record)
        
        # Update run record
        run_record.success = True
        run_record.model_id = model_record.id
        run_record.metrics = result.metrics
        run_record.training_time_seconds = training_time
        registry.insert_run(run_record)
        
        logger.info(f"Training completed: model_id={model_record.id}")
        
        return {
            "success": True,
            "model_id": model_record.id,
            "model_type": model_record.model_type,
            "tickers": model_record.tickers,
            "metrics": model_record.metrics,
            "training_run_id": run_record.id,
            "training_time_seconds": training_time,
        }
    
    except Exception as e:
        logger.exception(f"Training task failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }
