"""
ML Models module.

Supports 5 model types:
- logistic: Logistic Regression (always available)
- random_forest: Random Forest (always available)
- xgboost: XGBoost (optional, pip install xgboost)
- lightgbm: LightGBM (optional, pip install lightgbm)
- catboost: CatBoost (optional, pip install catboost)

All models implement unified interface:
- fit(X, y) -> self
- predict(X) -> array
- predict_proba(X) -> array
- save(path) -> None
- load(path) -> cls
"""

from .base import BaseModel, ModelMetadata
from .factory import ModelFactory, get_model, list_available_models
from .sklearn_models import LogisticModel, RandomForestModel

__all__ = [
    # Base
    "BaseModel",
    "ModelMetadata",
    # Factory
    "ModelFactory",
    "get_model",
    "list_available_models",
    # Always available
    "LogisticModel",
    "RandomForestModel",
]

# Conditionally export optional models
try:
    from .xgboost_model import XGBoostModel, XGBOOST_AVAILABLE

    if XGBOOST_AVAILABLE:
        __all__.append("XGBoostModel")
except ImportError:
    pass

try:
    from .lightgbm_model import LightGBMModel, LIGHTGBM_AVAILABLE

    if LIGHTGBM_AVAILABLE:
        __all__.append("LightGBMModel")
except ImportError:
    pass

try:
    from .catboost_model import CatBoostModel, CATBOOST_AVAILABLE

    if CATBOOST_AVAILABLE:
        __all__.append("CatBoostModel")
except ImportError:
    pass
