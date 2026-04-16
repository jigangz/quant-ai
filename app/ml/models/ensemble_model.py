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
        """
        Fit via K-fold out-of-fold predictions.

        Steps:
        1. K-fold split (shuffle=False, preserves time order)
        2. For each fold: train bases on train side, predict on val side
        3. Collect OOF matrix [N, n_base]
        4. Train meta-learner on OOF matrix
        5. Retrain base models on full data (used at inference)
        """
        from sklearn.model_selection import KFold

        n = len(X)
        n_base = len(self.config.base_models)
        oof_preds = np.zeros((n, n_base))

        kf = KFold(n_splits=self.config.cv_folds, shuffle=False)
        fold_idx = 0
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            for i, model_type in enumerate(self.config.base_models):
                params = self.config.base_model_params.get(model_type, {})
                base = factory.create(model_type, **params)
                base.fit(X_tr, y_tr)
                oof_preds[val_idx, i] = base.predict_proba(X_val)[:, 1]
            fold_idx += 1

        logger.info(
            f"Stacking OOF complete: {fold_idx} folds, OOF shape {oof_preds.shape}"
        )

        # Train meta-learner on OOF predictions
        meta_type = "logistic" if self.config.mode == "stacking_logistic" else "xgboost"
        self.meta_model = factory.create(meta_type)
        self.meta_model.fit(
            pd.DataFrame(oof_preds, columns=[f"base_{i}" for i in range(n_base)]),
            y.reset_index(drop=True),
        )

        # Retrain base models on full data — these are used at inference
        self.base_models = []
        for model_type in self.config.base_models:
            params = self.config.base_model_params.get(model_type, {})
            base = factory.create(model_type, **params)
            base.fit(X, y)
            self.base_models.append(base)

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
            # Stack base positive-class probabilities → feed to meta-learner
            base_probs = np.stack(
                [m.predict_proba(X)[:, 1] for m in self.base_models], axis=1
            )
            stacked = pd.DataFrame(
                base_probs, columns=[f"base_{i}" for i in range(len(self.base_models))]
            )
            return self.meta_model.predict_proba(stacked)

        raise ValueError(f"Unknown ensemble mode: {mode}")

    # ------------------------------------------------------------------
    # Persistence — custom save/load (does NOT use BaseModel.save which
    # expects self.model; EnsembleModel has a list of base models instead)
    # ------------------------------------------------------------------

    def save(self, path: "str | Path") -> None:
        """
        Save ensemble to disk.

        Layout::

            <path>/
                base_models.joblib   — list of fitted base model instances
                meta_model.joblib    — fitted meta model (stacking only)
                params.json          — EnsembleConfig as dict (for load())
                metadata.json        — ModelMetadata (optional)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Serialize list of base model instances
        joblib.dump(self.base_models, path / "base_models.joblib")

        # Meta model only exists for stacking modes
        if self.meta_model is not None:
            joblib.dump(self.meta_model, path / "meta_model.joblib")

        # params.json — only the ensemble_config dict is needed to reconstruct
        with open(path / "params.json", "w") as f:
            json.dump({"ensemble_config": self.config.model_dump()}, f, indent=2)

        # metadata.json (optional)
        if self.metadata is not None:
            with open(path / "metadata.json", "w") as f:
                json.dump(
                    self.metadata.model_dump(mode="json"), f, indent=2, default=str
                )

        logger.info(f"EnsembleModel saved to {path}")

    @classmethod
    def load(cls, path: "str | Path") -> "EnsembleModel":
        """
        Load ensemble from disk.

        Args:
            path: Directory written by save()

        Returns:
            Fitted EnsembleModel instance
        """
        path = Path(path)

        # Reconstruct instance from params
        with open(path / "params.json") as f:
            params = json.load(f)

        instance = cls(**params)

        # Restore fitted base models
        instance.base_models = joblib.load(path / "base_models.joblib")

        # Restore meta model if present
        meta_path = path / "meta_model.joblib"
        if meta_path.exists():
            instance.meta_model = joblib.load(meta_path)

        instance.is_fitted = True

        # Restore metadata if present
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            from app.ml.models.base import ModelMetadata
            with open(metadata_path) as f:
                instance.metadata = ModelMetadata(**json.load(f))

        logger.info(f"EnsembleModel loaded from {path}")
        return instance
