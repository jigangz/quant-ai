from __future__ import annotations

"""
Prediction Service

Supports:
- Promoted model (production) as default
- Loading specific models by model_id from registry
- LRU caching via ModelCache
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.core.metrics import MODEL_INFERENCE_SECONDS, PREDICT_CONFIDENCE, PREDICT_TOTAL
from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.services.model_cache import get_model_cache
from app.services.prediction_event_publisher import PredictionEvent, publish_prediction_event

logger = logging.getLogger(__name__)


def get_model(model_id: Optional[str] = None):
    """
    Get a model for prediction.

    Args:
        model_id: Specific model ID, or None to use promoted model

    Returns:
        Loaded model or None
    """
    cache = get_model_cache()

    if model_id:
        # Load specific model
        return cache.get(model_id)
    else:
        # Use promoted model
        _, model = cache.get_promoted()
        if model:
            return model

        # Fallback: try to load legacy default model
        return _get_legacy_default_model()


def _get_legacy_default_model():
    """Load legacy default model (backward compatibility)."""
    import joblib

    default_path = Path("artifacts/model.joblib")
    if default_path.exists():
        try:
            model = joblib.load(default_path)
            logger.info(f"Loaded legacy default model from {default_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load legacy model: {e}")

    return None


class PredictionService:
    """
    Service for making predictions.

    Usage:
        service = PredictionService()
        result = service.predict(ticker="AAPL", model_id="abc123")
    """

    def predict(
        self,
        ticker: str,
        model_id: Optional[str] = None,
        horizons: Optional[list] = None,
        features: Optional[dict] = None,
    ) -> dict:
        """
        Make predictions for a ticker.

        Args:
            ticker: Stock ticker
            model_id: Model ID (uses promoted if not specified)
            horizons: Prediction horizons in days
            features: Pre-computed features (optional)

        Returns:
            Prediction result with probabilities
        """
        horizons = horizons or [5]

        # Get model
        model = get_model(model_id)
        if model is None:
            return {
                "success": False,
                "error": "No model available. Train one or promote a model.",
                "ticker": ticker,
            }

        try:
            # Get features
            if features:
                # Use provided features
                X = pd.DataFrame([features])
            else:
                # Build features from market data
                X = self._build_features(ticker)

            if X is None or len(X) == 0:
                return {
                    "success": False,
                    "error": f"Could not build features for {ticker}",
                    "ticker": ticker,
                }

            model_type = getattr(model, "model_type", "unknown")

            # Make prediction — timed for Prometheus histogram
            with MODEL_INFERENCE_SECONDS.labels(model_type=model_type).time():
                proba = model.predict_proba(X)
                pred = model.predict(X)

            # Get the last row (most recent)
            latest_proba = proba[-1] if len(proba.shape) > 1 else proba
            latest_pred = pred[-1] if hasattr(pred, "__len__") else pred

            prob_up = float(latest_proba[1])

            # Prometheus metrics
            PREDICT_TOTAL.labels(ticker=ticker, model_type=model_type).inc()
            PREDICT_CONFIDENCE.labels(ticker=ticker).observe(prob_up)

            # Fire-and-forget Kafka publish (no-op in sync context or non-kafka mode)
            event = PredictionEvent(
                ticker=ticker,
                prediction=int(latest_pred),
                confidence=prob_up,
                model_id=model_id or "promoted",
                model_type=model_type,
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(publish_prediction_event(event))
            except RuntimeError:
                pass  # Sync context — no running event loop, skip publish

            return {
                "success": True,
                "ticker": ticker,
                "model_id": model_id or "promoted",
                "prediction": int(latest_pred),
                "probability": {
                    "down": float(latest_proba[0]),
                    "up": prob_up,
                },
                "signal": "LONG" if latest_pred == 1 else "SHORT",
                "confidence": float(max(latest_proba)),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker,
            }

    def _build_features(self, ticker: str) -> Optional[pd.DataFrame]:
        """Build features from market data."""
        try:
            # Get recent price data
            df = get_prices(ticker, limit=100)

            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            # Add technical features
            df = add_technical_features(df)

            # Get feature columns (exclude non-features)
            exclude_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
            feature_cols = [c for c in df.columns if c not in exclude_cols]

            # Return last row with features
            return df[feature_cols].tail(1)

        except Exception as e:
            logger.error(f"Failed to build features for {ticker}: {e}")
            return None


# Stub for monkeypatching in tests (V4 P5)
def _run_legacy_predict(**kwargs) -> dict:
    return {}


# Convenience function
def predict(ticker: str, model_id: Optional[str] = None, **kwargs) -> dict:
    """Make a prediction (convenience function)."""
    service = PredictionService()
    return service.predict(ticker=ticker, model_id=model_id)


# V4 P5: prediction_log write helper. Non-blocking — log failures must
# not break a prediction response.
def _write_prediction_log(
    *,
    ticker: str,
    model_id: str,
    model_type: str,
    label_type: str,
    horizon_days: int,
    predicted_value: float,
    predicted_signal: Optional[int],
    feature_group: str,
) -> None:
    try:
        from datetime import datetime, timedelta, timezone
        from app.db.prediction_log import PredictionLogRecord, get_prediction_log_repo

        repo = get_prediction_log_repo()
        now = datetime.now(timezone.utc)
        repo.insert(
            PredictionLogRecord(
                model_id=model_id,
                ticker=ticker,
                label_type=label_type,
                horizon_days=horizon_days,
                predicted_value=float(predicted_value),
                predicted_signal=predicted_signal,
                predicted_extras={"feature_group": feature_group, "model_type": model_type},
                resolve_at=now + timedelta(days=horizon_days),
            )
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "prediction_log write failed (non-blocking): %s", e
        )
