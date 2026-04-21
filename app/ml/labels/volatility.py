"""
Volatility label generator — V4 Pivot Phase 2 target.

Predicts realized volatility over the next N trading days. Leverages the well-
documented volatility clustering phenomenon (high-vol days follow high-vol days
and vice versa), which GARCH family models have exploited for 50 years.

Label definition:
    realized_vol[t] = std(daily_returns[t+1], ..., daily_returns[t+horizon])
    if cfg.volatility_annualize: realized_vol *= sqrt(252)

Use case:
- Options pricing (implied volatility input)
- Position sizing (inverse-vol weighting)
- Risk management (VaR, max drawdown estimation)

Note: Unlike direction prediction, volatility has real-world predictability
(López de Prado, Cochrane). This is why V4 Pivot treats it as a first-class
target alongside direction/return baselines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.ml.dataset.schemas import LabelConfig


def add_volatility_label(df: pd.DataFrame, cfg: "LabelConfig") -> pd.DataFrame:
    """Add realized future volatility label (regression target).

    Computes std of the next `cfg.horizon_days` daily returns. Rows near the
    tail (insufficient future data) receive NaN label and are dropped by the
    caller via `dropna(subset=["label"])`.

    Args:
        df: DataFrame with at least ["date", "close"] columns.
        cfg: LabelConfig. Reads horizon_days + volatility_annualize.

    Returns:
        DataFrame with added columns: future_price, future_return, label (float
        or NaN). When volatility_annualize=True (default), label is in units of
        annualized standard deviation (e.g., 0.20 = 20% annualized).
    """
    df = df.sort_values("date").copy()

    horizon = cfg.horizon_days

    # Daily simple returns. First row is NaN (pct_change of first close).
    returns = df["close"].pct_change()

    # For each row t, realized vol = std(returns[t+1], ..., returns[t+horizon]).
    # Implementation: shift-then-reverse-rolling-then-reverse — a pandas idiom
    # for forward-looking rolling aggregations.
    # shifted[t] = returns[t+1]
    shifted = returns.shift(-1)
    # Reverse, compute trailing rolling std over horizon, reverse back:
    # at original row t, value = std of returns[t+1 : t+horizon+1].
    realized_vol = shifted.iloc[::-1].rolling(window=horizon).std().iloc[::-1]

    if cfg.volatility_annualize:
        realized_vol = realized_vol * np.sqrt(252)

    # Parity columns with direction/return generators (downstream may use them).
    df["future_price"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_price"] - df["close"]) / df["close"]
    df["label"] = realized_vol

    return df
