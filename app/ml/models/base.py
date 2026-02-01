"""
Base Model Interface

Defines the interface that all ML models must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel as PydanticModel, Field


class ModelMetadata(PydanticModel):
    """Metadata for a trained model."""

    model_type: str
    model_name: str
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Training info
    feature_names: list[str] = []
    feature_groups: list[str] = []
    tickers: list[str] = []
    train_samples: int = 0
    val_samples: int = 0

    # Date range
    train_start_date: str | None = None
    train_end_date: str | None = None

    # Hyperparameters
    params: dict[str, Any] = {}

    # Metrics
    metrics: dict[str, float] = {}

    class Config:
        arbitrary_types_allowed = True


class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    All models must implement:
    - fit(X, y) -> self
    - predict(X) -> array
    - predict_proba(X) -> array
    - save(path) -> None
    - load(path) -> cls
    """

    model_type: str = "base"

    def __init__(self, **params):
        self.params = params
        self.model = None
        self.metadata = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Fit the model to training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability array (n_samples, n_classes)
        """
        pass

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / "model.joblib"
        joblib.dump(self.model, model_path)

        # Save metadata
        if self.metadata:
            metadata_path = path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(
                    self.metadata.model_dump(mode="json"), f, indent=2, default=str
                )

        # Save params
        params_path = path / "params.json"
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: Directory containing saved model

        Returns:
            Loaded model instance
        """
        path = Path(path)

        # Load params
        params_path = path / "params.json"
        with open(params_path, "r") as f:
            params = json.load(f)

        # Create instance
        instance = cls(**params)

        # Load model
        model_path = path / "model.joblib"
        instance.model = joblib.load(model_path)
        instance.is_fitted = True

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            instance.metadata = ModelMetadata(**metadata_dict)

        return instance

    def set_metadata(
        self,
        feature_names: list[str],
        feature_groups: list[str],
        tickers: list[str],
        train_samples: int,
        val_samples: int,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Set model metadata after training."""
        self.metadata = ModelMetadata(
            model_type=self.model_type,
            model_name=f"{self.model_type}_v1",
            feature_names=feature_names,
            feature_groups=feature_groups,
            tickers=tickers,
            train_samples=train_samples,
            val_samples=val_samples,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            params=self.params,
            metrics=metrics or {},
        )
