from __future__ import annotations

"""
Ensemble Model

Combines predictions from N base models using voting or stacking.

Modes:
- voting_soft:    mean(base.predict_proba)
- voting_hard:    majority vote of base.predict()
- stacking_logistic: LogisticRegression meta-learner over K-fold OOF predictions
- stacking_xgboost:  XGBoost meta-learner over K-fold OOF predictions

Integration:
- Inherits BaseModel — plugs into TrainingService + predict API as model_type="ensemble"
- Persistence: base_models.joblib + meta_model.joblib + metadata/params JSON
"""

import json
import logging
from pathlib import Path
from typing import Literal, Union

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel, ConfigDict, Field

from app.ml.models.base import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)


EnsembleMode = Literal[
    "voting_soft",
    "voting_hard",
    "stacking_logistic",
    "stacking_xgboost",
]


class EnsembleConfig(PydanticModel):
    """Configuration for an EnsembleModel."""

    mode: EnsembleMode
    base_models: list[str] = Field(min_length=2)
    base_model_params: dict[str, dict] = Field(default_factory=dict)
    cv_folds: int = Field(default=5, ge=2, le=10)

    model_config = ConfigDict(extra="forbid")


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models via voting or stacking."""

    model_type = "ensemble"

    def __init__(self, ensemble_config: Union[EnsembleConfig, dict], **kwargs):
        # Normalize to pydantic
        if isinstance(ensemble_config, dict):
            config = EnsembleConfig(**ensemble_config)
        else:
            config = ensemble_config

        # Store as dict in self.params for BaseModel.save() to serialize
        super().__init__(ensemble_config=config.model_dump())

        self.config: EnsembleConfig = config
        self.base_models: list[BaseModel] = []
        self.meta_model: Union[BaseModel, None] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        from app.ml.models import ModelFactory  # local import to avoid circular

        if self.config.mode in ("voting_soft", "voting_hard"):
            self._fit_voting(X, y, ModelFactory)
        else:
            self._fit_stacking(X, y, ModelFactory)

        self.is_fitted = True
        return self

    def _fit_voting(self, X: pd.DataFrame, y: pd.Series, factory) -> None:
        """Fit each base model on the full training set."""
        self.base_models = []
        for model_type in self.config.base_models:
            params = self.config.base_model_params.get(model_type, {})
            base = factory.create(model_type, **params)
            base.fit(X, y)
            self.base_models.append(base)
        logger.info(
            f"Voting ensemble fit: {len(self.base_models)} base models "
            f"({self.config.base_models})"
        )

    def _fit_stacking(self, X: pd.DataFrame, y: pd.Series, factory) -> None:
        """Placeholder — implemented in Task 3."""
        raise NotImplementedError("stacking implemented in Task 3")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EnsembleModel not fitted — call fit() first")

        mode = self.config.mode

        if mode == "voting_soft":
            # mean of predict_proba positive column across base models
            probs = np.stack([m.predict_proba(X)[:, 1] for m in self.base_models], axis=1)
            pos = probs.mean(axis=1)
            return np.column_stack([1.0 - pos, pos])

        if mode == "voting_hard":
            # majority vote of predict() across base models
            preds = np.stack([m.predict(X) for m in self.base_models], axis=1)
            # Majority: positive class wins if > half of base models predict 1
            pos = (preds.sum(axis=1) > len(self.base_models) / 2).astype(float)
            return np.column_stack([1.0 - pos, pos])

        if mode.startswith("stacking"):
            raise NotImplementedError("stacking predict_proba implemented in Task 3")

        raise ValueError(f"Unknown ensemble mode: {mode}")
