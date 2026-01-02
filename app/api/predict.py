from fastapi import APIRouter
from pydantic import BaseModel

from app.services.predict_service import predict

router = APIRouter()

# ============================
# v2 Request Schema
# Used by frontend (POST /predict)
# ============================
class PredictRequest(BaseModel):
    ticker: str
    horizons: list[int] = [5]
    features: dict = {}

# ============================
# v1: Legacy GET endpoint
# Kept for backward compatibility
# ============================
@router.get("/predict")
def predict_api_get(
    ticker: str,
    lookback: int = 500,
):
    return predict(
        ticker=ticker,
        lookback=lookback,
    )

# ============================
# v2: POST endpoint for UI
# Allows structured JSON input
# ============================
@router.post("/predict")
def predict_api_post(request: PredictRequest):
    # Currently ignoring horizons / features
    # These will be used in v2 / v3 iterations
    return predict(
        ticker=request.ticker,
        lookback=500,
    )
