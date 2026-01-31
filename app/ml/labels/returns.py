import pandas as pd
from app.ml.config import HORIZON_DAYS


def add_future_return_label(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Add future return + direction label.

    Label definition:
        future_return = (close[t + HORIZON] - close[t]) / close[t]
        label = 1 if future_return > 0 else 0

    Notes:
    - Uses shift(-HORIZON_DAYS) to avoid leakage
    - Drops rows without future data
    """

    df = df.sort_values("date").copy()

    df["future_price"] = df[price_col].shift(-HORIZON_DAYS)
    df["future_return"] = (df["future_price"] - df[price_col]) / df[price_col]

    df["label"] = (df["future_return"] > 0).astype(int)

    df = df.dropna(subset=["future_return"])

    return df
