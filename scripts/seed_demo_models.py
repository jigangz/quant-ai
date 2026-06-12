"""Seed the model_registry with a few real trained models — TEMPORARY.

Why this exists: prod's SUPABASE_KEY is a *publishable* (anon) key, so the
backend's REST writes to model_registry are blocked by row-level security
(error 42501). Until Render's key is swapped for a secret/service_role key,
POST /train cannot register models. This script trains real models locally
and writes the registry rows via the DATABASE_URL super-user connection
(which bypasses RLS), so the Leaderboard has content.

Limitation: model *artifacts* (.pkl) live on the local disk here, not in
Supabase Storage — so prod can serve the Leaderboard (registry metadata)
but cannot run live predictions against these rows. Full demo (Dashboard
prediction + Portfolio promotion) needs the secret key + STORAGE_BACKEND
=supabase + promotion persisted to the DB. See progress.md.

Usage:  python -m scripts.seed_demo_models
Idempotent-ish: deletes any prior rows with the same model id before insert.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)
logging.disable(logging.WARNING)

from app.services.training_service import TrainingService, TrainRequest  # noqa: E402

# (label_type, model_type, feature_groups, tickers)
JOBS = [
    ("direction", "xgboost", ["ta_basic", "momentum", "volatility"], ["AAPL", "MSFT", "NVDA"]),
    ("direction", "lightgbm", ["ta_basic", "momentum"], ["TSLA", "AMZN"]),
    ("volatility", "xgboost", ["ta_basic", "volatility"], ["AAPL", "MSFT", "NVDA"]),
]

_INSERT = text(
    """INSERT INTO model_registry
    (id,name,model_type,version,status,label_type,horizon_days,tickers,
     feature_groups,feature_names,n_features,train_samples,val_samples,
     test_samples,train_date_range,metrics,extras,artifact_path,storage_backend)
    VALUES (:id,:name,:mt,1,'active',:lt,5,:tk,:fg,:fn,:nf,:tr,:vl,:ts,:dr,
            cast(:metrics as jsonb), cast(:extras as jsonb), :ap,'local')"""
)


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    svc = TrainingService()
    seeded = []

    for label_type, model_type, fgroups, tickers in JOBS:
        req = TrainRequest(
            tickers=tickers,
            feature_groups=fgroups,
            horizon_days=5,
            label_type=label_type,
            model_type=model_type,
            search_mode="none",
            save_model=True,
        )
        res = svc.train(req)
        if not getattr(res, "success", True) or not getattr(res, "model_id", None):
            print(f"  skip {label_type}/{model_type}: training produced no model")
            continue

        row = dict(
            id=res.model_id,
            name=f"{model_type}_{label_type}",
            mt=res.model_type,
            lt=label_type,
            tk=res.tickers,
            fg=res.feature_groups,
            fn=res.feature_names,
            nf=res.n_features,
            tr=res.train_samples,
            vl=res.val_samples,
            ts=res.test_samples,
            dr=list(res.train_date_range),
            metrics=json.dumps(res.metrics),
            extras=json.dumps({}),
            ap=res.model_path or "local",
        )
        with engine.begin() as c:
            c.execute(text("DELETE FROM model_registry WHERE id=:id"), {"id": res.model_id})
            c.execute(_INSERT, row)
        # Persist the artifact blob so prod (stateless) can load + predict
        from app.ml.models.base import zip_model_dir
        blob = zip_model_dir(res.model_path)
        with engine.begin() as c:
            c.execute(
                text(
                    "INSERT INTO model_artifacts (model_id, blob) VALUES (:i,:b) "
                    "ON CONFLICT (model_id) DO UPDATE SET blob = EXCLUDED.blob"
                ),
                {"i": res.model_id, "b": blob},
            )
        primary = res.metrics.get("test_auc") if label_type != "volatility" else res.metrics.get("test_qlike")
        print(f"  seeded {label_type:11} {model_type:9} {res.model_id}  primary={primary}")
        seeded.append(res.model_id)

    with engine.connect() as c:
        n = c.execute(text("SELECT count(*) FROM model_registry")).scalar()
    print(f"\nDone. {len(seeded)} models seeded; registry now has {n} row(s).")


if __name__ == "__main__":
    main()
