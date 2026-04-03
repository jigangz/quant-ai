"""
SHAP Explainer Function

Wraps the existing ShapExplainer to generate SHAP explanations
as a serverless function.
"""

from __future__ import annotations

import logging
from typing import Any

from app.explain.shap_explainer import ShapExplainer

logger = logging.getLogger(__name__)


async def handle_shap_generation(payload: dict[str, Any]) -> dict[str, Any]:
    """
    SHAP generation function handler.

    Payload:
        ticker (str): Stock ticker symbol.
        model_id (str|None): Model ID to explain (uses promoted model if omitted).
        lookback (int): Number of historical samples (default 500).
        top_k (int): Number of top features to return (default 10).

    Returns:
        SHAP explanation dict from ShapExplainer.explain().
    """
    ticker = payload.get("ticker", "")
    if not ticker:
        return {"success": False, "error": "Missing 'ticker' in payload"}

    model_id = payload.get("model_id")
    lookback = payload.get("lookback", 500)
    top_k = payload.get("top_k", 10)

    explainer = ShapExplainer()
    result = explainer.explain(
        ticker=ticker,
        model_id=model_id,
        lookback=lookback,
        top_k=top_k,
    )

    return result
