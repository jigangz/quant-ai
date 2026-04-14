from __future__ import annotations

from typing import Dict

from app.explain.shap_explainer import ShapExplainer

_explainer: ShapExplainer | None = None


def _get_explainer() -> ShapExplainer:
    global _explainer
    if _explainer is None:
        _explainer = ShapExplainer()
    return _explainer


def explain(ticker: str, lookback: int = 1000, top_k: int = 10) -> Dict:
    explainer = _get_explainer()
    result = explainer.explain(ticker=ticker, lookback=lookback, top_k=top_k)
    return {"status": "ok", "data": result}
