"""
Prediction Service

Supports:
- Default model (artifacts/model.joblib) for backward compatibility
- Loading specific models by model_id from registry
"""

import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from app.core.settings import settings
from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.ml.labels.returns import add_future_return_label
from app.ml.features.build import build_xy

logger = logging.getLogger(__name__)

# ===================================
# Model Cache
# ===================================
_model_cache: Dict[str, object] = {}
_default_model = None


def _get_default_model():
    """Load the default model (backward compatibility)."""
    global _default_model
    if _default_model is None:
        default_path = Path("artifacts/model.joblib")
        if default_path.exists():
            _default_model = joblib.load(default_path)
            logger.info(f"Loaded default model from {default_path}")
        else:
            logger.warning(f"Default model not found at {default_path}")
    return _default_model


def load_model(model_id: str):
    """
    Load a model by ID from the registry.

    Caches loaded models for performance.
    """
    if model_id in _model_cache:
        return _model_cache[model_id]

    from app.db.model_registry import get_model_registry

    registry = get_model_registry()
    model_record = registry.get_model(model_id)

    if not model_record:
        logger.error(f"Model not found in registry: {model_id}")
        return None

    if not model_record.artifact_path:
        logger.error(f"Model has no artifact path: {model_id}")
        return None

    # Load from artifact path
    artifact_path = Path(model_record.artifact_path)
    if not artifact_path.exists():
        # Try with .joblib extension
        artifact_path = Path(f"{model_record.artifact_path}.joblib")

    if not artifact_path.exists():
        logger.error(f"Model artifact not found: {artifact_path}")
        return None

    try:
        model = joblib.load(artifact_path)
        _model_cache[model_id] = model
        logger.info(f"Loaded model {model_id} from {artifact_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        return None


def clear_model_cache():
    """Clear the model cache (for testing/reloading)."""
    global _model_cache, _default_model
    _model_cache = {}
    _default_model = None


# ===================================
# Prediction
# ===================================
def predict(
    ticker: str,
    lookback: int = 500,
    model_id: str | None = None,
) -> Dict:
    """
    Core prediction service.

    Args:
        ticker: Stock ticker symbol
        lookback: Number of historical data points
        model_id: Optional model ID to use (defaults to legacy model)

    Returns:
        Prediction result with probability and signal
    """
    # === 1. Load model ===
    if model_id:
        model = load_model(model_id)
        if not model:
            return {
                "status": "error",
                "message": f"Model not found: {model_id}",
                "model_id": model_id,
            }
    else:
        model = _get_default_model()
        if not model:
            return {
                "status": "error",
                "message": "No default model available. Train a model first.",
            }

    # === 2. Load data ===
    rows = get_prices(ticker, lookback)
    if not rows:
        return {
            "status": "error",
            "message": "No price data found",
        }

    df = pd.DataFrame(rows)

    # === 3. Feature engineering ===
    df_feat = add_technical_features(df)
    df_labeled = add_future_return_label(df_feat)

    X, _ = build_xy(df_labeled)

    if X.empty:
        return {
            "status": "error",
            "message": "Not enough data after feature engineering",
        }

    # === 4. Predict on last row only (most recent) ===
    X_last = X.tail(1)

    try:
        prob_up = float(model.predict_proba(X_last)[0, 1])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
        }

    signal = "LONG" if prob_up > 0.55 else "SHORT" if prob_up < 0.45 else "HOLD"

    result = {
        "status": "ok",
        "ticker": ticker,
        "samples": int(len(X)),
        "prob_up": prob_up,
        "signal": signal,
    }

    if model_id:
        result["model_id"] = model_id

    return result
