"""
Prediction Service

Supports:
- Promoted model (production) as default
- Loading specific models by model_id from registry
- LRU caching via ModelCache
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.services.model_cache import get_model_cache

logger = logging.getLogger(__name__)


def get_model(model_id: str | None = None):
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
        model_id: str | None = None,
        horizons: list[int] | None = None,
        features: dict | None = None,
    ) -> dict[str, Any]:
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
            
            # Make prediction
            proba = model.predict_proba(X)
            pred = model.predict(X)
            
            # Get the last row (most recent)
            latest_proba = proba[-1] if len(proba.shape) > 1 else proba
            latest_pred = pred[-1] if hasattr(pred, '__len__') else pred
            
            return {
                "success": True,
                "ticker": ticker,
                "model_id": model_id or "promoted",
                "prediction": int(latest_pred),
                "probability": {
                    "down": float(latest_proba[0]),
                    "up": float(latest_proba[1]),
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
    
    def _build_features(self, ticker: str) -> pd.DataFrame | None:
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


# Convenience function
def predict(ticker: str, model_id: str | None = None, **kwargs) -> dict:
    """Make a prediction (convenience function)."""
    service = PredictionService()
    return service.predict(ticker=ticker, model_id=model_id, **kwargs)
