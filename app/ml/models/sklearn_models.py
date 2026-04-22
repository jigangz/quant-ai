from __future__ import annotations

"""
Scikit-learn Model Implementations

Provides LogisticRegression and RandomForest wrappers for both
classification and regression tasks (V4 Pivot multi-task ML).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import BaseModel


class LogisticModel(BaseModel):
    """Linear model wrapper — LogisticRegression for classification, Ridge for regression."""

    model_type = "logistic"

    def __init__(
        self,
        task: str = "classification",
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        alpha: float = 1.0,  # Ridge regularization (regression only)
        **kwargs,
    ):
        super().__init__(
            task=task, C=C, max_iter=max_iter, class_weight=class_weight,
            alpha=alpha, **kwargs,
        )
        self.task = task

        if task == "classification":
            clf = LogisticRegression(
                C=C, max_iter=max_iter, class_weight=class_weight, random_state=42,
            )
        elif task == "regression":
            clf = Ridge(alpha=alpha, max_iter=max_iter, random_state=42)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'classification' or 'regression'.")

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError(
                f"predict_proba not available for task='{self.task}'. "
                f"Use predict() for regression."
            )
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest wrapper — Classifier or Regressor based on task."""

    model_type = "random_forest"

    def __init__(
        self,
        task: str = "classification",
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        **kwargs,
    ):
        super().__init__(
            task=task,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            **kwargs,
        )
        self.task = task

        common_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        if task == "classification":
            clf = RandomForestClassifier(class_weight=class_weight, **common_kwargs)
        elif task == "regression":
            clf = RandomForestRegressor(**common_kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError(
                f"predict_proba not available for task='{self.task}'."
            )
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        clf = self.model.named_steps["clf"]
        importance = clf.feature_importances_

        return dict(
            sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        )
