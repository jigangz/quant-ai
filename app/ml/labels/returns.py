"""
Return label generator — regression target (raw future return).

future_return = (close[t + horizon] - close[t]) / close[t]
label = future_return (float; NaN for rows without future data)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from app.ml.config import HORIZON_DAYS

if TYPE_CHECKING:
    from app.ml.dataset.schemas import LabelConfig


def add_return_label(df: pd.DataFrame, cfg: "LabelConfig") -> pd.DataFrame:
    """Add regression label = raw future_return.

    V4 Pivot registry-compatible signature.

    Args:
        df: DataFrame with at least ["date", "close"] columns.
        cfg: LabelConfig with label_type="return" (horizon_days read).

    Returns:
        DataFrame with added columns: future_price, future_return, label (float, NaN in tail).
    """
    df = df.sort_values("date").copy()

    horizon = cfg.horizon_days

    df["future_price"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_price"] - df["close"]) / df["close"]

    # Label = raw return (NaN preserved for tail rows without future)
    df["label"] = df["future_return"]

    return df


def add_future_return_label(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    [Legacy] Add future return + direction label in one pass.

    Kept for backward compatibility with any caller importing this directly.
    New code should use `add_return_label` or `add_direction_label` via the registry.

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
