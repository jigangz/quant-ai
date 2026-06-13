from __future__ import annotations

"""
Dataset Schemas

Defines configuration and result structures for dataset building.
"""

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class LabelConfig(BaseModel):
    """
    Configuration for label generation.

    V4 Pivot (2026-04-22): Extended from 2 types (direction, return) to 4 types
    to support multi-task ML. Direction/return are production-ready; volatility
    and meta_label are V4 Phase 2/3 targets (Day 3+ implementation).

    Types:
    - direction: Binary classification — sign(future_return > threshold).
    - return:    Regression — raw future_return.
    - volatility: Regression — realized volatility over next N days. [V4 P2]
    - meta_label: Binary/regression — signal quality score for rule triggers. [V4 P3]
    - xs_strong: Cross-sectional classification — each date's top `top_pct`
                 forward-return names = strong group (1). The per-date label is
                 assigned post-concat in DatasetBuilder, not per-ticker. [V5 Phase C]
    """

    label_type: Literal[
        "direction", "return", "volatility", "meta_label", "xs_strong"
    ] = "direction"
    horizon_days: int = Field(default=5, ge=1, le=60)
    threshold: float = Field(default=0.0, description="Threshold for direction labels")

    # V5 Phase C · xs_strong: fraction of each date's cross-section labeled
    # "strong" (1). Ignored for other label_types.
    top_pct: float = Field(
        default=0.30,
        gt=0,
        lt=1,
        description="xs_strong per-date strong-group fraction (top X% forward return).",
    )

    # V4 Phase 2 · Volatility options (ignored for other label_types)
    volatility_annualize: bool = Field(
        default=True,
        description="Annualize realized vol by sqrt(252). Only used when label_type='volatility'.",
    )

    model_config = ConfigDict(extra="forbid")


class SplitConfig(BaseModel):
    """Configuration for train/val/test split."""

    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.3)

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio

    model_config = ConfigDict(extra="forbid")


class DatasetConfig(BaseModel):
    """Full configuration for dataset building."""

    # Tickers
    tickers: list[str] = Field(min_length=1)

    # Date range
    start_date: date | None = None
    end_date: date | None = None

    # Features
    feature_groups: list[str] = Field(default=["ta_basic"])
    # Explicit feature override (V5 Phase C): when set, these columns are used
    # verbatim instead of resolving feature_groups — lets xs_strong train on the
    # Phase-B-selected factor set (incl. factors not exposed by any group).
    feature_names: list[str] | None = None

    # Labels
    label_config: LabelConfig = Field(default_factory=LabelConfig)

    # Split
    split_config: SplitConfig = Field(default_factory=SplitConfig)

    # Data source (V5 Phase C): None → settings.MARKET_PROVIDER (yahoo). Set to
    # "db" for cross-sectional training off the backfilled prices table.
    market_provider: str | None = None

    # Options
    drop_na_features: bool = False  # If True, drop rows with NaN features
    min_samples_per_ticker: int = Field(default=100, ge=10)

    model_config = ConfigDict(extra="forbid")


class TickerDataset(BaseModel):
    """Dataset for a single ticker."""

    ticker: str
    n_samples: int
    n_features: int
    date_range: tuple[str, str]  # (start, end)
    label_distribution: dict[str, int]  # {0: count, 1: count}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatasetResult(BaseModel):
    """Result of dataset building."""

    # Metadata
    config: DatasetConfig
    tickers_processed: list[str]
    tickers_skipped: list[str] = []
    total_samples: int
    n_features: int
    feature_names: list[str]

    # Per-ticker info
    ticker_stats: list[TickerDataset]

    # Split info
    train_samples: int
    val_samples: int
    test_samples: int
    train_date_range: tuple[str, str]
    val_date_range: tuple[str, str]
    test_date_range: tuple[str, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatasetOutput:
    """
    Container for dataset output (not Pydantic - holds DataFrames).

    Attributes:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        metadata: DatasetResult with all metadata
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metadata: DatasetResult,
        groups_train: pd.DataFrame | None = None,
        groups_val: pd.DataFrame | None = None,
        groups_test: pd.DataFrame | None = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.metadata = metadata
        # V5 Phase C: per-split [date, future_return] for cross-sectional eval
        # (Rank IC / precision@top_pct). Populated only for label_type='xs_strong';
        # None for the single-ticker label paths.
        self.groups_train = groups_train
        self.groups_val = groups_val
        self.groups_test = groups_test

    def __repr__(self) -> str:
        return (
            f"DatasetOutput("
            f"train={len(self.X_train)}, "
            f"val={len(self.X_val)}, "
            f"test={len(self.X_test)}, "
            f"features={len(self.X_train.columns)})"
        )
