"""
Training Task - Actual training logic run by worker

This is executed in the worker process, not the API process.
Captures full experiment metadata for reproducibility.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def run_training_task(request_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Execute training job with full experiment tracking.
    
    Args:
        request_dict: Training request parameters
        
    Returns:
        Training result with model_id, metrics, experiment metadata
    """
    from app.services.training_service import TrainRequest, TrainingService
    from app.db.model_registry import (
        ModelRecord,
        TrainingRunRecord,
        get_model_registry,
    )
    from app.jobs.experiment import collect_experiment_metadata
    
    logger.info(f"Starting training task: {request_dict}")
    start_time = datetime.utcnow()
    
    try:
        # Parse request
        request = TrainRequest(**request_dict)
        
        # Collect experiment metadata (git, env, config hash)
        exp_metadata = collect_experiment_metadata(request_dict)
        
        # Create training run record with full metadata
        registry = get_model_registry()
        run_record = TrainingRunRecord(
            # Request params
            tickers=request.tickers,
            model_type=request.model_type,
            feature_groups=request.feature_groups,
            model_params=request.model_params,
            horizon_days=request.horizon_days,
            label_type=getattr(request, "label_type", "direction"),
            train_ratio=getattr(request, "train_ratio", 0.7),
            val_ratio=getattr(request, "val_ratio", 0.15),
            start_date=str(request.start_date) if request.start_date else None,
            end_date=str(request.end_date) if request.end_date else None,
            # Reproducibility
            git_sha=exp_metadata.get("git_sha"),
            git_branch=exp_metadata.get("git_branch"),
            git_dirty=exp_metadata.get("git_dirty", False),
            config_hash=exp_metadata.get("config_hash"),
            python_version=exp_metadata.get("python_version"),
            packages=exp_metadata.get("packages", {}),
        )
        registry.insert_run(run_record)
        logger.info(f"Training run created: {run_record.id}")
        
        # Run training (captures data info)
        service = TrainingService()
        result = service.train(request)
        
        # Calculate training time
        end_time = datetime.utcnow()
        training_time = (end_time - start_time).total_seconds()
        
        # Update run record with data info
        run_record.train_samples = result.train_samples
        run_record.val_samples = result.val_samples
        run_record.test_samples = result.test_samples
        run_record.train_date_range = result.train_date_range
        run_record.training_time_seconds = training_time
        
        if not result.success:
            # Update run record with failure
            run_record.success = False
            run_record.error = result.error
            registry.insert_run(run_record)
            
            logger.error(f"Training failed: {result.error}")
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
        
        # Final update to run record
        run_record.success = True
        run_record.model_id = model_record.id
        run_record.metrics = result.metrics
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
            # Experiment metadata
            "experiment": {
                "git_sha": run_record.git_sha,
                "git_branch": run_record.git_branch,
                "git_dirty": run_record.git_dirty,
                "config_hash": run_record.config_hash,
                "train_samples": run_record.train_samples,
                "val_samples": run_record.val_samples,
                "test_samples": run_record.test_samples,
            },
        }
    
    except Exception as e:
        logger.exception(f"Training task failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }
