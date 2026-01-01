from fastapi import APIRouter
from app.services.predict_service import predict

router = APIRouter()


@router.get("/predict")
def predict_api(
    ticker: str,
    lookback: int = 500,
):
    return predict(
        ticker=ticker,
        lookback=lookback,
    )
