from __future__ import annotations

"""
Optimization API Endpoints

- POST /api/optimize/model — multi-objective model hyperparameter optimization
- POST /api/optimize/strategy — strategy parameter optimization
- GET /api/optimize/runs — list optimization history
- GET /api/optimize/runs/{id} — get single optimization run
"""

import logging
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.services.optimization_service import OptimizationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimize", tags=["optimization"])

VALID_MODEL_TYPES = {"logistic", "random_forest", "xgboost", "lightgbm", "catboost"}
VALID_METRICS = {"sharpe_ratio", "total_return", "win_rate", "max_drawdown"}


class OptimizeModelRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    model_type: str
    n_trials: int = Field(default=50, ge=5, le=200)
    timeout: Optional[int] = Field(default=300, ge=10, le=3600)
    feature_groups: list[str] = Field(default=["ta_basic", "momentum"])

    model_config = ConfigDict(extra="forbid")


class OptimizeStrategyRequest(BaseModel):
    strategy_name: str
    ticker: str
    n_trials: int = Field(default=100, ge=5, le=500)
    timeout: Optional[int] = Field(default=300, ge=10, le=3600)
    metric: str = "sharpe_ratio"
    param_overrides: Optional[dict[str, dict]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    model_config = ConfigDict(extra="forbid")


@router.post("/model")
def optimize_model(request: OptimizeModelRequest):
    """Run multi-objective model hyperparameter optimization."""
    if request.model_type not in VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {request.model_type}. "
            f"Valid types: {sorted(VALID_MODEL_TYPES)}",
        )

    try:
        service = OptimizationService()
        run = service.optimize_model(
            tickers=request.tickers,
            model_type=request.model_type,
            n_trials=request.n_trials,
            timeout=request.timeout,
            feature_groups=request.feature_groups,
        )
        return run.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Model optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy")
def optimize_strategy(request: OptimizeStrategyRequest):
    """Run strategy parameter optimization."""
    if request.metric not in VALID_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric: {request.metric}. "
            f"Valid metrics: {sorted(VALID_METRICS)}",
        )

    try:
        service = OptimizationService()
        run = service.optimize_strategy(
            strategy_name=request.strategy_name,
            ticker=request.ticker,
            n_trials=request.n_trials,
            timeout=request.timeout,
            metric=request.metric,
            param_overrides=request.param_overrides,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return run.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Strategy optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
def list_optimization_runs(type: Optional[str] = None, limit: int = 20):
    """List optimization run history."""
    service = OptimizationService()
    runs = service.list_runs(type=type, limit=limit)
    return [r.model_dump(mode="json") for r in runs]


@router.get("/runs/{run_id}")
def get_optimization_run(run_id: str):
    """Get a single optimization run by ID."""
    service = OptimizationService()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Optimization run not found")
    return run.model_dump(mode="json")
