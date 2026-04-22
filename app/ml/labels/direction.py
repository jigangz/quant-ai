"""
Direction label generator — binary classification target.

future_return = (close[t + horizon] - close[t]) / close[t]
label = (future_return > threshold).astype(int)

Note: For rows where future data is not available (last `horizon_days` rows),
`future_return` is NaN. The astype(int) conversion makes NaN > threshold → False → 0.
This quirk is preserved for backward compatibility with pre-V4-Pivot behavior.
Callers that need strict "valid-only" labels should filter by future_return.notna().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # Imported only for type hints. Importing at runtime creates a circular
    # dependency (app.ml.dataset.__init__ imports builder → labels.registry).
    from app.ml.dataset.schemas import LabelConfig


def add_direction_label(df: pd.DataFrame, cfg: "LabelConfig") -> pd.DataFrame:
    """Add direction label (binary 0/1) based on sign of future return.

    Args:
        df: DataFrame with at least ["date", "close"] columns.
        cfg: LabelConfig with label_type="direction" (only horizon_days + threshold read).

    Returns:
        DataFrame with added columns: future_price, future_return, label (int 0/1).
    """
    df = df.sort_values("date").copy()

    horizon = cfg.horizon_days
    threshold = cfg.threshold

    df["future_price"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_price"] - df["close"]) / df["close"]

    # Preserve pre-refactor behavior: NaN > threshold → False → 0
    df["label"] = (df["future_return"] > threshold).astype(int)

    return df
