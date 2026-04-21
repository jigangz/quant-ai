from __future__ import annotations

"""
Prediction API

GET /predict                - Legacy endpoint (backward compatible)
POST /predict               - JSON-based direction prediction with model selection
POST /predict/volatility    - [V4 P2] Forward-looking realized volatility (regression)
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.services.predict_service import predict
from app.services.volatility_predict_service import predict_volatility

router = APIRouter()


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
