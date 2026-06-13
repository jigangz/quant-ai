"""Cross-sectional "strong group" label — V5 Phase C.

Two-stage by necessity: forward return is per-ticker (computable on one name's
time series), but the strong/weak cut is *cross-sectional* — each date's top
`top_pct` of forward returns. A single ticker is its own degenerate
cross-section, so the binary label cannot be assigned per-ticker.

  1. add_xs_forward_return(df, cfg)  — per ticker, in DatasetBuilder._process_ticker.
     Adds future_price / future_return + a provisional `label` (0.0 on valid
     rows, NaN on the unlabelable tail) so the builder's dropna(subset=["label"])
     trims the tail exactly like the single-ticker labels do.

  2. add_xs_strong_label(panel, cfg) — once, on the concatenated multi-ticker
     panel in DatasetBuilder.build(). Overwrites `label` with the real per-date
     top-top_pct 0/1, keeping future_return for cross-sectional evaluation.

Mirrors the validated PoC scripts/xs_eval.py:65-77.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.ml.dataset.schemas import LabelConfig


def add_xs_forward_return(df: pd.DataFrame, cfg: "LabelConfig") -> pd.DataFrame:
    """Per-ticker: add future_price / future_return + a provisional label.

    Args:
        df: one ticker's frame with ["date", "close"].
        cfg: LabelConfig (horizon_days read).

    Returns:
        df with future_price, future_return, and provisional `label` (0.0 where
        future_return is valid, NaN in the last `horizon_days` rows). The real
        0/1 label is assigned cross-sectionally post-concat.
    """
    df = df.sort_values("date").copy()
    horizon = cfg.horizon_days

    df["future_price"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_price"] - df["close"]) / df["close"]
    # Provisional: 0 where labelable, NaN in the tail so the builder drops it.
    df["label"] = np.where(df["future_return"].notna(), 0.0, np.nan)
    return df


def add_xs_strong_label(panel: pd.DataFrame, cfg: "LabelConfig") -> pd.DataFrame:
    """Cross-sectional: per date, label the top `top_pct` forward returns as 1.

    Args:
        panel: concatenated multi-ticker frame with ["date", "future_return"].
        cfg: LabelConfig (top_pct read).

    Returns:
        panel copy with `label` = 1 for names whose future_return is >= that
        date's (1 - top_pct) quantile, else 0. future_return is preserved.
    """
    panel = panel.copy()
    top = cfg.top_pct
    grp = panel.groupby("date")["future_return"]
    cutoff = grp.transform(lambda s: s.quantile(1.0 - top))
    panel["label"] = (panel["future_return"] >= cutoff).astype(int)
    # Degenerate date with no return spread (every name equal) has no strong
    # group — `>= constant` would otherwise label all of them 1. Force 0.
    nuniq = grp.transform("nunique")
    panel.loc[nuniq <= 1, "label"] = 0
    return panel
