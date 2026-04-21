"""
Volatility Prediction Service — V4 Pivot Phase 2.

Wraps a regression model trained with label_type='volatility' to produce
forward-looking realized volatility estimates for a ticker.

Usage:
    from app.services.volatility_predict_service import predict_volatility
    result = predict_volatility(ticker="AAPL", model_id="xgboost_vol_AAPL_v1")
"""

from __future__ import annotations

import logging
from typing import Optional

from app.services.predict_service import PredictionService, get_model

logger = logging.getLogger(__name__)

# Reuse feature builder from classification service (same TA pipeline).
_feature_builder = PredictionService()


def predict_volatility(
    ticker: str,
    model_id: Optional[str] = None,
    horizon_days: int = 5,
) -> dict:
    """Predict forward-looking realized volatility for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        model_id: Specific model ID, or None to use promoted model.
        horizon_days: Horizon for which vol was trained (metadata pass-through).

    Returns:
        dict with keys:
            success: bool
            ticker: str
            predicted_volatility: float (annualized stdev, e.g., 0.25 = 25%)
            model_id: str
            model_label_type: str
            horizon_days: int
            annualized: bool
        On error: {"success": False, "error": <msg>, "ticker": ticker}
    """
    model = get_model(model_id)
    if model is None:
        return {
            "success": False,
            "error": (
                "No volatility model available. Train one with "
                "label_type='volatility' then promote, or pass an explicit model_id."
            ),
            "ticker": ticker,
        }

    # Soft check: warn if the loaded model is clearly a classifier (direction).
    model_label_type = getattr(model, "label_type", None)
    if model_label_type not in (None, "volatility"):
        return {
            "success": False,
            "error": (
                f"Model label_type is '{model_label_type}', not 'volatility'. "
                f"Use /predict for direction models."
            ),
            "ticker": ticker,
        }

    # Build the most-recent feature row via the shared TA pipeline.
    X = _feature_builder._build_features(ticker)
    if X is None or len(X) == 0:
        return {
            "success": False,
            "error": f"Could not build features for {ticker}",
            "ticker": ticker,
        }

    try:
        # Regression output: single scalar per row (annualized vol if trained so).
        raw_pred = model.predict(X)
        predicted_vol = float(raw_pred[-1]) if hasattr(raw_pred, "__len__") else float(raw_pred)

        return {
            "success": True,
            "ticker": ticker,
            "model_id": model_id or "promoted",
            "predicted_volatility": predicted_vol,
            "model_label_type": model_label_type or "volatility",
            "horizon_days": horizon_days,
            "annualized": True,
        }

    except Exception as e:
        logger.error(f"Volatility prediction failed for {ticker}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "ticker": ticker,
        }
