"""
Scikit-learn Model Implementations

Provides LogisticRegression and RandomForest wrappers.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class LogisticModel(BaseModel):
    """Logistic Regression model with imputation and scaling."""
    
    model_type = "logistic"
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        **kwargs
    ):
        super().__init__(C=C, max_iter=max_iter, class_weight=class_weight, **kwargs)
        
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=42,
            )),
        ])
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest model with imputation."""
    
    model_type = "random_forest"
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            **kwargs
        )
        
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            )),
        ])
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        clf = self.model.named_steps["clf"]
        importance = clf.feature_importances_
        
        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))
