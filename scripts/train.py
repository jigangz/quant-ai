

import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.ml.labels.returns import add_future_return_label
from app.ml.features.build import build_xy
from app.ml.split.time_split import time_series_split


def main():
    ticker = "AAPL"

    # === 1. Load price data ===
    rows = get_prices(ticker, 5000)
    df = pd.DataFrame(rows)

    # === 2. Feature engineering ===
    df_feat = add_technical_features(df)

    # === 3. Label generation (no leakage) ===
    df_labeled = add_future_return_label(df_feat)

    # === 4. Build X / y (no dropna on features) ===
    X, y = build_xy(df_labeled)

    # === 5. Time-series split ===
    df_xy = X.copy()
    df_xy["label"] = y.values
    df_xy["date"] = df_labeled.loc[X.index, "date"].values

    train_df, val_df, _ = time_series_split(df_xy)

    X_train = train_df.drop(columns=["label", "date"])
    y_train = train_df["label"]

    X_val = val_df.drop(columns=["label", "date"])
    y_val = val_df["label"]

    # 🔍 可选调试（跑通后可以删）
    print("Train label distribution:")
    print(y_train.value_counts())

    # === 6. Train baseline model (Pipeline + Imputer) ===
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # === 7. Evaluation ===
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "ticker": ticker,
        "model": "LogisticRegression",
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "accuracy": accuracy_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_prob) if y_val.nunique() > 1 else None,
    }

    # === 8. Save artifacts ===
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(model, artifacts_dir / "model.joblib")

    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training finished.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
