"""Label generators for multi-task ML (V4 Pivot)."""

from __future__ import annotations

from app.ml.labels.direction import add_direction_label
from app.ml.labels.registry import LABEL_GENERATORS, add_labels
from app.ml.labels.returns import add_future_return_label, add_return_label
from app.ml.labels.volatility import add_volatility_label

__all__ = [
    "add_direction_label",
    "add_return_label",
    "add_future_return_label",  # legacy
    "add_volatility_label",
    "LABEL_GENERATORS",
    "add_labels",
]
