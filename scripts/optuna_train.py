import optuna
import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from app.db.prices_repo import get_prices
from app.ml.features.technical import add_technical_features
from app.ml.labels.returns import add_future_return_label
from app.ml.features.build import build_xy
from app.ml.split.time_split import time_series_split


def suggest_logistic_params(trial):
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "solver": "saga",
        "penalty": "elasticnet",
        "l1_ratio": l1_ratio,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": 2000,
        "n_jobs": -1,
    }


# =========================
# Prepare data ONCE
# =========================
rows = get_prices("AAPL", 10000)
df = pd.DataFrame(rows)

df_feat = add_technical_features(df)
df_labeled = add_future_return_label(df_feat)

X, y = build_xy(df_labeled)

df_xy = X.copy()
df_xy["label"] = y.values
df_xy["date"] = df_labeled.loc[X.index, "date"].values

train_df, val_df, _ = time_series_split(df_xy)

X_train = train_df.drop(columns=["label", "date"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["label", "date"])
y_val = val_df["label"]


def objective(trial):
    params = suggest_logistic_params(trial)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(**params)),
        ]
    )

    model.fit(X_train, y_train)

    # safety check
    if y_val.nunique() < 2:
        return 0.5

    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)

    return auc


# =========================
# Optuna entry point
# =========================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best AUC:", study.best_value)
    print("Best params:", study.best_params)

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    study.trials_dataframe().to_csv(artifacts / "optuna_trials.csv", index=False)

    with open(artifacts / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
