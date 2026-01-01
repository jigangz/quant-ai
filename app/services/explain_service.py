from typing import Dict

from app.explain.shap_explainer import ShapExplainer

_explainer = ShapExplainer(model_path="artifacts/model.joblib")


def explain(
    ticker: str,
    lookback: int = 1000,
    top_k: int = 10,
) -> Dict:
    """
    Core explain service (no HTTP here).
    """
    result = _explainer.explain(
        ticker=ticker,
        lookback=lookback,
        top_k=top_k,
    )

    return {
        "status": "ok",
        "data": result,
    }
