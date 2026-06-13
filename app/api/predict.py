from __future__ import annotations

"""
Prediction API

GET /predict                - Legacy endpoint (backward compatible)
POST /predict               - JSON-based direction prediction with model selection
POST /predict/volatility    - [V4 P2] Forward-looking realized volatility (regression)
"""

import threading
from datetime import date

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.services.predict_service import predict
from app.services.ranking_service import RankingService
from app.services.volatility_predict_service import predict_volatility

router = APIRouter()

# Daily cache for the (expensive) full-universe ranking — keyed by
# (model_id, today). The board only changes once per close. Guarded by a lock
# so concurrent first-requests (FastAPI runs sync routes in a threadpool) don't
# both recompute; stale-day entries are pruned rather than clearing everything,
# so distinct model_ids coexist.
_RANKING_CACHE: dict = {}
_RANKING_LOCK = threading.Lock()


# ===================================
# Request Schemas
# ===================================
class PredictRequest(BaseModel):
    """Request for POST /predict."""

    ticker: str
    horizons: list[int] = [5]  # Reserved for future use
    features: dict = {}  # Reserved for future use
    model_id: str | None = None  # Specific model to use


class PredictVolatilityRequest(BaseModel):
    """Request for POST /predict/volatility (V4 Phase 2)."""

    ticker: str
    model_id: str | None = None
    horizon_days: int = Field(default=5, ge=1, le=60)


# ===================================
# GET /predict (Legacy)
# ===================================
@router.get("/predict")
def predict_api_get(
    ticker: str,
    lookback: int = Query(500, ge=50, le=2000),
    model_id: str | None = Query(None, description="Model ID to use"),
):
    """
    Legacy GET endpoint for prediction.

    Args:
        ticker: Stock ticker symbol
        lookback: Number of historical data points (default 500)
        model_id: Optional model ID (defaults to legacy model)
    """
    return predict(
        ticker=ticker,
        lookback=lookback,
        model_id=model_id,
    )


# ===================================
# POST /predict
# ===================================
@router.post("/predict")
def predict_api_post(request: PredictRequest):
    """
    JSON-based prediction endpoint.

    Supports model selection via model_id.

    Example:
        POST /predict
        {
            "ticker": "AAPL",
            "model_id": "xgboost_AAPL_20240131_120000"
        }
    """
    return predict(
        ticker=request.ticker,
        model_id=request.model_id,
    )


# ===================================
# POST /predict/volatility  [V4 Phase 2]
# ===================================
@router.post("/predict/volatility")
def predict_volatility_api(request: PredictVolatilityRequest):
    """
    Forward-looking realized volatility prediction (V4 Pivot Phase 2).

    Accepts a regression model trained with label_type='volatility'.

    Example:
        POST /predict/volatility
        {
            "ticker": "AAPL",
            "model_id": "xgboost_vol_AAPL_v1",
            "horizon_days": 5
        }

    Returns:
        {
            "success": true,
            "ticker": "AAPL",
            "predicted_volatility": 0.28,   # 28% annualized
            "annualized": true,
            "horizon_days": 5,
            ...
        }
    """
    return predict_volatility(
        ticker=request.ticker,
        model_id=request.model_id,
        horizon_days=request.horizon_days,
    )


# ===================================
# GET /predict/ranking  [V5 Phase D]
# ===================================
@router.get("/predict/ranking")
def predict_ranking_api(
    top_n: int = Query(20, ge=1, le=100, description="How many top names to return"),
    model_id: str | None = Query(None, description="xs_strong model id (default: latest)"),
):
    """Cross-sectional Top-N strength ranking (V5 Phase D).

    Scores the whole universe as of the latest close with the xs_strong model
    (per-date normalized, sorted by strength score), returns the Top-N. Result
    is cached per (model_id, day) since the board only moves once per close.

    Returns:
        { success, model_id, as_of, universe_size, scored, top_n, rankings:[
          {rank, ticker, score, percentile} ], score_semantics }
    """
    today = date.today().isoformat()
    key = (model_id, today)
    full = _RANKING_CACHE.get(key)
    if full is None or not full.get("success"):
        with _RANKING_LOCK:
            full = _RANKING_CACHE.get(key)  # re-check: another thread may have filled it
            if full is None or not full.get("success"):
                # Compute a deep ranking once, then slice per request.
                full = RankingService().rank(model_id=model_id, top_n=100)
                if full.get("success"):
                    # Prune stale-day entries; keep distinct model_ids for today.
                    for k in [k for k in _RANKING_CACHE if k[1] != today]:
                        _RANKING_CACHE.pop(k, None)
                    _RANKING_CACHE[key] = full

    out = dict(full)
    out["rankings"] = (full.get("rankings") or [])[:top_n]
    out["top_n"] = top_n
    return out
