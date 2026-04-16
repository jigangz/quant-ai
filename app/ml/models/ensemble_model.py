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
        raise NotImplementedError("fit() will be implemented in Task 2 / Task 3")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("predict_proba() will be implemented in Task 2 / Task 3")
