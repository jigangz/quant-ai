import joblib
import pandas as pd
from typing import Dict

from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.ml.labels.returns import add_future_return_label
from app.ml.features.build import build_xy


# ===== Load model once (singleton style) =====
_model = joblib.load("artifacts/model.joblib")


def predict(
    ticker: str,
    lookback: int = 500,
) -> Dict:
    """
    Core prediction service.

    Returns probability-based signal.
    """

    # === 1. Load data ===
    rows = get_prices(ticker, lookback)
    if not rows:
        return {
            "status": "error",
            "message": "No price data found",
        }

    df = pd.DataFrame(rows)

    # === 2. Feature engineering ===
    df_feat = add_technical_features(df)
    df_labeled = add_future_return_label(df_feat)

    X, _ = build_xy(df_labeled)

    if X.empty:
        return {
            "status": "error",
            "message": "Not enough data after feature engineering",
        }

    # === 3. Predict on last row only (most recent) ===
    X_last = X.tail(1)

    prob_up = float(_model.predict_proba(X_last)[0, 1])

    signal = (
        "LONG" if prob_up > 0.55
        else "SHORT" if prob_up < 0.45
        else "HOLD"
    )

    return {
        "status": "ok",
        "ticker": ticker,
        "samples": int(len(X)),
        "prob_up": prob_up,
        "signal": signal,
    }
