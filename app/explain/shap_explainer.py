import joblib
import shap
import pandas as pd
from typing import Dict

from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.ml.labels.returns import add_future_return_label
from app.ml.features.build import build_xy


class ShapExplainer:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def explain(
        self,
        ticker: str,
        lookback: int = 1000,
        top_k: int = 10,
    ) -> Dict:
        """
        Return SHAP summary for a ticker.
        """

        # === Load data ===
        rows = get_prices(ticker, lookback)
        df = pd.DataFrame(rows)

        df_feat = add_technical_features(df)
        df_labeled = add_future_return_label(df_feat)

        X, _ = build_xy(df_labeled)

        # === SHAP ===
        explainer = shap.LinearExplainer(
            self.model.named_steps["clf"],
            self.model.named_steps["imputer"].transform(X),
        )

        shap_values = explainer.shap_values(
            self.model.named_steps["imputer"].transform(X)
        )

        shap_df = pd.DataFrame(
            shap_values,
            columns=X.columns,
        )

        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

        top_features = [
            {"feature": k, "mean_abs_shap": float(v)}
            for k, v in mean_abs_shap.head(top_k).items()
        ]

        return {
            "ticker": ticker,
            "samples": len(X),
            "top_features": top_features,
        }
