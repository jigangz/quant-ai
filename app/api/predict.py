"""
Prediction API

GET /predict - Legacy endpoint (backward compatible)
POST /predict - JSON-based prediction with model selection
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.services.predict_service import predict

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
        lookback=500,
        model_id=request.model_id,
    )
