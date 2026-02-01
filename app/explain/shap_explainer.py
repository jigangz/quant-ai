"""
SHAP Explainer

Generates SHAP explanations for model predictions.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.ml.labels.returns import add_future_return_label
from app.ml.features.build import build_xy

logger = logging.getLogger(__name__)


class ShapExplainer:
    """
    SHAP-based model explainer.
    
    Usage:
        explainer = ShapExplainer()
        result = explainer.explain(ticker="AAPL", model_id="abc123")
    """
    
    def __init__(self, model_path: str | None = None):
        """
        Initialize explainer.
        
        Args:
            model_path: Legacy path to model (optional)
        """
        self.model_path = model_path
        self.model_cache = {}
    
    def _load_model(self, model_id: str | None = None):
        """Load model by ID or from path."""
        if model_id:
            if model_id in self.model_cache:
                return self.model_cache[model_id]
            
            from app.db.model_registry import get_model_registry
            registry = get_model_registry()
            record = registry.get_model(model_id)
            
            if not record or not record.artifact_path:
                raise ValueError(f"Model not found: {model_id}")
            
            path = Path(record.artifact_path)
            if not path.exists():
                path = Path(f"{record.artifact_path}/model.joblib")
            
            if not path.exists():
                raise ValueError(f"Model artifact not found: {path}")
            
            model = joblib.load(path)
            self.model_cache[model_id] = model
            return model
        
        elif self.model_path:
            return joblib.load(self.model_path)
        
        else:
            # Try promoted model
            from app.services.model_cache import get_model_cache
            cache = get_model_cache()
            promoted_id, model = cache.get_promoted()
            if model:
                return model
            
            raise ValueError("No model specified")
    
    def explain(
        self,
        ticker: str,
        model_id: str | None = None,
        lookback: int = 500,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Generate SHAP explanation for a ticker.
        
        Args:
            ticker: Stock ticker
            model_id: Model ID (uses promoted if not specified)
            lookback: Number of historical samples
            top_k: Number of top features to return
            
        Returns:
            Dict with SHAP values, feature values, and top features
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed")
            return {"success": False, "error": "SHAP not installed"}
        
        try:
            # Load model
            model = self._load_model(model_id)
            
            # Extract components
            if hasattr(model, "named_steps"):
                clf = model.named_steps.get("clf")
                imputer = model.named_steps.get("imputer")
            else:
                clf = model
                imputer = None
            
            # Load and prepare data
            rows = get_prices(ticker, lookback)
            if not rows:
                return {"success": False, "error": f"No data for {ticker}"}
            
            df = pd.DataFrame(rows)
            df_feat = add_technical_features(df)
            df_labeled = add_future_return_label(df_feat)
            
            X, _ = build_xy(df_labeled)
            
            if X.empty:
                return {"success": False, "error": "No features available"}
            
            # Impute if needed
            if imputer is not None:
                X_imp = imputer.transform(X)
            else:
                X_imp = X.fillna(0).values
            
            # Get latest sample for explanation
            X_latest = X_imp[-1:] if len(X_imp) > 0 else X_imp
            
            # Create SHAP explainer
            # Use background samples for context
            background_size = min(100, len(X_imp) - 1)
            background = X_imp[:background_size] if background_size > 0 else X_imp
            
            # Choose explainer based on model type
            model_type = type(clf).__name__.lower()
            
            if "linear" in model_type or "logistic" in model_type:
                explainer = shap.LinearExplainer(clf, background)
            elif "tree" in model_type or "forest" in model_type or "xgb" in model_type or "lgb" in model_type:
                explainer = shap.TreeExplainer(clf)
            else:
                # Fallback to KernelExplainer (slower)
                explainer = shap.KernelExplainer(
                    lambda x: clf.predict_proba(x)[:, 1],
                    background[:50],
                )
            
            # Get SHAP values
            shap_vals = explainer.shap_values(X_latest)
            
            # Handle different SHAP value formats
            if isinstance(shap_vals, list):
                # For multi-class, take class 1 (up)
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            
            shap_vals = shap_vals.flatten()
            
            # Build results
            feature_names = list(X.columns)
            
            # SHAP values dict
            shap_dict = {
                name: float(val)
                for name, val in zip(feature_names, shap_vals)
            }
            
            # Feature values dict (latest sample)
            feature_dict = {
                name: float(X.iloc[-1][name])
                for name in feature_names
                if name in X.columns
            }
            
            # Top features by absolute SHAP
            sorted_features = sorted(
                zip(feature_names, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_k]
            
            top_features = [
                {
                    "feature": name,
                    "shap_value": float(val),
                    "feature_value": feature_dict.get(name, 0),
                    "direction": "bullish" if val > 0 else "bearish",
                }
                for name, val in sorted_features
            ]
            
            # Mean absolute SHAP (for overall feature importance)
            all_shap = explainer.shap_values(X_imp[-100:]) if len(X_imp) > 100 else explainer.shap_values(X_imp)
            if isinstance(all_shap, list):
                all_shap = all_shap[1] if len(all_shap) > 1 else all_shap[0]
            
            mean_abs_shap = np.abs(all_shap).mean(axis=0)
            feature_importance = {
                name: float(val)
                for name, val in zip(feature_names, mean_abs_shap)
            }
            
            return {
                "success": True,
                "ticker": ticker,
                "model_id": model_id,
                "samples": len(X),
                "shap_values": shap_dict,
                "feature_values": feature_dict,
                "top_features": top_features,
                "feature_importance": feature_importance,
            }
        
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
