"""
Label generator registry — V4 Pivot multi-task ML dispatcher.

Exposes a uniform `add_labels(df, cfg)` API over all label_type targets:
- direction  (V1, binary classification)
- return     (V1, regression)
- volatility (V4 Phase 2, regression — NotImplementedError placeholder, Day 3+)
- meta_label (V4 Phase 3, binary/regression — NotImplementedError placeholder)

Usage:
    from app.ml.labels.registry import add_labels
    df_labeled = add_labels(df, cfg=LabelConfig(label_type="direction", horizon_days=5))
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from app.ml.dataset.schemas import LabelConfig
from app.ml.labels.direction import add_direction_label
from app.ml.labels.returns import add_return_label
from app.ml.labels.volatility import add_volatility_label


def _not_implemented_meta_label(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    raise NotImplementedError(
        "meta_label generator is a V4 Phase 3 target (López de Prado triple-barrier). "
        "Tracking: ml-pivot-task-plan.md Phase 3."
    )


LABEL_GENERATORS: dict[str, Callable[[pd.DataFrame, LabelConfig], pd.DataFrame]] = {
    "direction": add_direction_label,
    "return": add_return_label,
    "volatility": add_volatility_label,
    "meta_label": _not_implemented_meta_label,
}


def add_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """Dispatch label generation based on cfg.label_type.

    Args:
        df: DataFrame with at least ["date", "close"] columns.
        cfg: LabelConfig specifying target type + parameters.

    Returns:
        DataFrame with added columns: future_price, future_return, label.

    Raises:
        NotImplementedError: for volatility / meta_label (V4 future phases).
        KeyError: if cfg.label_type not in LABEL_GENERATORS (pydantic should prevent this).
    """
    generator = LABEL_GENERATORS[cfg.label_type]
    return generator(df, cfg)
