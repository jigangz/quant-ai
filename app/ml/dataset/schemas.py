"""
Dataset Schemas

Defines configuration and result structures for dataset building.
"""

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class LabelConfig(BaseModel):
    """Configuration for label generation."""
    
    label_type: Literal["direction", "return"] = "direction"
    horizon_days: int = Field(default=5, ge=1, le=60)
    threshold: float = Field(default=0.0, description="Threshold for direction labels")
    
    class Config:
        extra = "forbid"


class SplitConfig(BaseModel):
    """Configuration for train/val/test split."""
    
    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.3)
    
    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio
    
    class Config:
        extra = "forbid"


class DatasetConfig(BaseModel):
    """Full configuration for dataset building."""
    
    # Tickers
    tickers: list[str] = Field(min_length=1)
    
    # Date range
    start_date: date | None = None
    end_date: date | None = None
    
    # Features
    feature_groups: list[str] = Field(default=["ta_basic"])
    
    # Labels
    label_config: LabelConfig = Field(default_factory=LabelConfig)
    
    # Split
    split_config: SplitConfig = Field(default_factory=SplitConfig)
    
    # Options
    drop_na_features: bool = False  # If True, drop rows with NaN features
    min_samples_per_ticker: int = Field(default=100, ge=10)
    
    class Config:
        extra = "forbid"


class TickerDataset(BaseModel):
    """Dataset for a single ticker."""
    
    ticker: str
    n_samples: int
    n_features: int
    date_range: tuple[str, str]  # (start, end)
    label_distribution: dict[str, int]  # {0: count, 1: count}
    
    class Config:
        arbitrary_types_allowed = True


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
    
    class Config:
        arbitrary_types_allowed = True


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
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.metadata = metadata
    
    def __repr__(self) -> str:
        return (
            f"DatasetOutput("
            f"train={len(self.X_train)}, "
            f"val={len(self.X_val)}, "
            f"test={len(self.X_test)}, "
            f"features={len(self.X_train.columns)})"
        )
