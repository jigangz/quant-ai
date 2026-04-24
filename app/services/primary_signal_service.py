"""
Primary Signal Service — dispatcher for meta-labeling's "main model" input.

Accepts either:
  - source="strategy" + strategy_name (one of the 4 rule strategies)
  - source="model" + primary_model_id (ML model from ModelRegistry; must be
    a direction classifier — regression/vol models rejected)

Returns a (signals, strengths) tuple:
  - signals: Series[int] in {-1, 0, +1}, same length as input OHLC
  - strengths: Series[float] in [0, 1]
      = 1.0 for rules (binary triggers)
      = |proba − 0.5| × 2 for ML models (0 at chance, 1 at certainty)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class PrimarySignalSpec(BaseModel):
    """Parameters describing which primary source to use."""

    source: Literal["strategy", "model"]
    strategy_name: Optional[str] = None
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    primary_model_id: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "PrimarySignalSpec":
        if self.source == "strategy":
            if not self.strategy_name or self.primary_model_id:
                raise ValueError(
                    "source=strategy requires exactly one of "
                    "strategy_name (given primary_model_id must be None)"
                )
        elif self.source == "model":
            if not self.primary_model_id or self.strategy_name:
                raise ValueError(
                    "source=model requires exactly one of "
                    "primary_model_id (given strategy_name must be None)"
                )
        return self


# ---- Rule strategy dispatch ----

_STRATEGY_REGISTRY: dict[str, Any] = {}


def _get_strategy_registry() -> dict[str, Any]:
    """Lazy-load the 4 rule strategies (import here to avoid top-level cycles)."""
    global _STRATEGY_REGISTRY
    if _STRATEGY_REGISTRY:
        return _STRATEGY_REGISTRY
    try:
        from app.strategies.templates.ma_cross import MACrossStrategy
        _STRATEGY_REGISTRY["ma_cross"] = MACrossStrategy
    except ImportError:
        pass
    try:
        from app.strategies.templates.rsi_strategy import RSIStrategy
        _STRATEGY_REGISTRY["rsi_strategy"] = RSIStrategy
    except ImportError:
        pass
    try:
        from app.strategies.templates.bollinger_breakout import BollingerBreakoutStrategy
        _STRATEGY_REGISTRY["bollinger_breakout"] = BollingerBreakoutStrategy
    except ImportError:
        pass
    try:
        from app.strategies.templates.sentiment_driven import SentimentDrivenStrategy
        _STRATEGY_REGISTRY["sentiment_driven"] = SentimentDrivenStrategy
    except ImportError:
        pass
    return _STRATEGY_REGISTRY


# ---- Model loader (wrapped for monkeypatchability) ----

def _load_model_for_inference(model_id: str) -> tuple[Any, dict]:
    """Load a registered model + its metadata. Wrapped so tests can monkeypatch."""
    from app.services.model_cache import get_model_cache
    cache = get_model_cache()
    model_info = cache.load(model_id)
    if model_info is None:
        raise ValueError(f"model {model_id!r} not found in registry")
    return model_info.model, model_info.metadata


# ---- Public API ----

def generate_primary_signals(
    spec: PrimarySignalSpec,
    ohlc: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Generate primary signals + strengths aligned to ohlc's index.

    Args:
        spec: PrimarySignalSpec describing the dispatcher choice.
        ohlc: DataFrame with columns ["date", "open", "high", "low", "close", "volume"].

    Returns:
        (signals, strengths) both pd.Series with len == len(ohlc).
    """
    if spec.source == "strategy":
        registry = _get_strategy_registry()
        if spec.strategy_name not in registry:
            raise ValueError(
                f"unknown strategy {spec.strategy_name!r}. "
                f"Known: {sorted(registry)}"
            )
        StrategyCls = registry[spec.strategy_name]
        strategy = (
            StrategyCls(**spec.strategy_params) if spec.strategy_params
            else StrategyCls()
        )
        signals = strategy.generate_signals(ohlc)
        signals = signals.reindex(ohlc.index).fillna(0).astype(int)
        strengths = pd.Series(1.0, index=ohlc.index)
        return signals, strengths

    # source == "model"
    model, metadata = _load_model_for_inference(spec.primary_model_id)
    task = metadata.get("task") or "classification"
    label_type = metadata.get("label_type") or "direction"
    if task != "classification" or label_type not in {"direction", "meta_label"}:
        raise ValueError(
            f"primary_model_id {spec.primary_model_id!r} is not a direction classifier "
            f"(task={task}, label_type={label_type}). "
            "Meta-labeling requires a direction primary."
        )
    features = _build_features_for_model(ohlc, metadata)
    probas = model.predict_proba(features)
    if probas.ndim != 2 or probas.shape[1] < 2:
        raise ValueError("primary model predict_proba must return 2-D array with ≥2 classes")
    p_up = probas[:, 1]
    signals = pd.Series(np.where(p_up >= 0.5, 1, -1), index=ohlc.index, dtype=int)
    strengths = pd.Series(np.abs(p_up - 0.5) * 2, index=ohlc.index)
    return signals, strengths


def _build_features_for_model(ohlc: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Reuse the same feature pipeline the model was trained with."""
    from app.ml.features.technical import add_technical_features
    df = ohlc.copy()
    df = add_technical_features(df)
    feature_cols = [
        c for c in df.columns
        if c not in {"date", "open", "high", "low", "close", "volume"}
    ]
    df_features = df[feature_cols].fillna(0.0)
    return df_features
