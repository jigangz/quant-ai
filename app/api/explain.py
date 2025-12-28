from fastapi import APIRouter, Query
from app.explain.shap_explainer import ShapExplainer

router = APIRouter()

explainer = ShapExplainer("artifacts/model.joblib")


@router.get("/explain")
def explain(
    ticker: str = Query(...),
    lookback: int = Query(1000),
    top_k: int = Query(10),
):
    return explainer.explain(
        ticker=ticker,
        lookback=lookback,
        top_k=top_k,
    )
