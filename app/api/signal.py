"""
Signal API — V4 Phase 3 meta-labeling endpoints.

POST /api/meta-label/train   — Train a meta-model for a ticker + primary source.
POST /api/signal-score        — Score a primary signal's reliability.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.primary_signal_service import PrimarySignalSpec
from app.services import meta_label_service, signal_scoring_service

router = APIRouter()


# =====================================
# POST /api/meta-label/train
# =====================================

class _BarrierIn(BaseModel):
    tp_k: float = Field(gt=0)
    sl_k: float = Field(gt=0)
    timeout_days: int = Field(ge=1, le=30)
    vol_source: Literal["p1_model", "realized_sigma"] = "realized_sigma"


class _CVIn(BaseModel):
    n_splits: int = Field(default=5, ge=2, le=10)
    embargo_pct: float = Field(default=0.01, ge=0.0, le=0.5)


class _ModelIn(BaseModel):
    type: Literal["xgboost", "lightgbm", "ensemble"] = "xgboost"
    ensemble_mode: Optional[str] = None
    search_mode: Literal["default", "optuna"] = "default"


class _WindowIn(BaseModel):
    lookback_days: int = Field(default=730, ge=60)
    feature_group: str = "ta_basic"


class MetaLabelTrainRequestAPI(BaseModel):
    ticker: str
    primary: PrimarySignalSpec
    barrier: _BarrierIn
    cv: _CVIn = Field(default_factory=_CVIn)
    model: _ModelIn = Field(default_factory=_ModelIn)
    window: _WindowIn = Field(default_factory=_WindowIn)


@router.post("/api/meta-label/train")
def meta_label_train(req: MetaLabelTrainRequestAPI):
    internal_req = meta_label_service.MetaLabelTrainRequest(
        ticker=req.ticker,
        primary=req.primary,
        tp_k=req.barrier.tp_k, sl_k=req.barrier.sl_k,
        timeout_days=req.barrier.timeout_days, vol_source=req.barrier.vol_source,
        cv_n_splits=req.cv.n_splits, cv_embargo_pct=req.cv.embargo_pct,
        model_type=req.model.type, ensemble_mode=req.model.ensemble_mode,
        search_mode=req.model.search_mode,
        lookback_days=req.window.lookback_days, feature_group=req.window.feature_group,
    )
    try:
        return meta_label_service.train_meta_label_model(internal_req)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("insufficient_") or msg.startswith("no_usable_folds"):
            raise HTTPException(status_code=400, detail=msg)
        if "not found" in msg or msg.startswith("meta_model_not_found"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


# =====================================
# POST /api/signal-score
# =====================================

class SignalScoreRequestAPI(BaseModel):
    ticker: str
    meta_model_id: str
    signal: Optional[Literal[-1, 1]] = None
    timestamp: Optional[str] = None
    strategy_name: Optional[str] = None
    strategy_params: Optional[dict] = None


@router.post("/api/signal-score")
def signal_score(req: SignalScoreRequestAPI):
    internal_req = signal_scoring_service.SignalScoreRequest(
        ticker=req.ticker, meta_model_id=req.meta_model_id,
        signal=req.signal, timestamp=req.timestamp,
        strategy_name=req.strategy_name, strategy_params=req.strategy_params,
    )
    try:
        return signal_scoring_service.score_signal(internal_req)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("meta_model_not_found") or msg.startswith("primary_model_not_found"):
            raise HTTPException(status_code=404, detail=msg)
        if msg.startswith("timestamp_out_of_range"):
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


# =====================================
# GET /api/meta-label/coverage
# =====================================

@router.get("/api/meta-label/coverage")
def meta_label_coverage(strategy: str):
    try:
        return signal_scoring_service.compute_coverage(strategy)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("strategy_not_found"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
