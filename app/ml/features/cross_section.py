"""Per-date cross-sectional normalization — the standardization fix (V5).

For cross-sectional ranking, a raw factor like ma_5=270 (AAPL) vs 12 (F) lets
the model learn *price level* instead of factor signal. The fix is to
winsorize + z-score each factor WITHIN each day's cross-section, so every
factor is comparable across names.

Critical, counter-intuitive vs single-name models: there is NO stored
scaler. Each date is normalized against its OWN cross-section — at train
time per training date, at predict time against today's cross-section. So
training and serving MUST call this exact same function (DRY) or they
silently diverge. That's why it lives here once and is imported by both
paths, with tests pinning the contract.
"""

from __future__ import annotations

import pandas as pd


def cross_section_normalize(
    df: pd.DataFrame,
    factor_cols: list[str],
    date_col: str = "date",
    winsor: float = 0.01,
) -> pd.DataFrame:
    """Winsorize (clip tails) then z-score each factor within each date group.

    Args:
        df: long frame with a date column + factor columns (one row per
            ticker-date).
        factor_cols: columns to normalize in place.
        date_col: the grouping column (the cross-section key).
        winsor: tail fraction to clip on each side (0.01 = 1%/99%).

    Returns a copy; rows whose date-group has a single name or zero variance
    get 0.0 for that factor (no information to rank on that day).
    """
    out = df.copy()
    groups = out.groupby(date_col)

    def _winsor_z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = s.quantile(winsor), s.quantile(1.0 - winsor)
        s = s.clip(lower=lo, upper=hi)
        sd = s.std()
        if sd and sd == sd:  # non-zero, non-NaN
            return (s - s.mean()) / sd
        return pd.Series(0.0, index=s.index)

    # transform stays per-date and never touches the grouping column.
    for c in factor_cols:
        out[c] = groups[c].transform(_winsor_z)
    return out
