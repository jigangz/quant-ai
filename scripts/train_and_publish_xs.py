"""Train the cross-sectional `xs_strong` stock-ranking model and publish it — V5 Phase C.

Trains through the REAL TrainingService pipeline (not the scripts/xs_eval PoC):
reads the backfilled `prices` table (market_provider="db", survivorship-safe
universe incl. delisted names), uses the Phase-B-selected factor set
(config/xs_selected_factors.json), labels each date's top-`top_pct` forward
return as the strong group, z-scores factors per date, evaluates with Rank IC +
precision@top_pct (NOT AUC), then writes a `model_registry` row
(label_type='xs_strong') + the model blob so prod can load it by id.

Usage:
    python -m scripts.train_and_publish_xs --dry-run            # train + report, no DB write
    python -m scripts.train_and_publish_xs --limit 100          # smaller universe
    python -m scripts.train_and_publish_xs                      # train + publish to registry

The registry write hits the shared Supabase prod registry (prod reads it). Run
--dry-run first to sanity-check the IC before publishing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)
logging.disable(logging.WARNING)

from app.ml.models.base import zip_model_dir  # noqa: E402
from app.services.training_service import TrainingService, TrainRequest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SELECTED_FACTORS_PATH = ROOT / "config" / "xs_selected_factors.json"

HORIZON = 5
TOP_PCT = 0.30
MIN_CROSS_SECTION = 5
MIN_HISTORY = 400  # tickers need enough rows to carry the long-window factors

_INSERT = text(
    """INSERT INTO model_registry
    (id,name,model_type,version,status,label_type,horizon_days,tickers,
     feature_groups,feature_names,n_features,train_samples,val_samples,
     test_samples,train_date_range,metrics,extras,artifact_path,storage_backend)
    VALUES (:id,:name,:mt,1,'active',:lt,:hz,:tk,:fg,:fn,:nf,:tr,:vl,:ts,:dr,
            cast(:metrics as jsonb), cast(:extras as jsonb), :ap,'local')"""
)


def load_selected_factors() -> list[str]:
    data = json.loads(SELECTED_FACTORS_PATH.read_text())
    return list(data["selected_factors"])


def load_universe(engine, limit: int | None) -> list[str]:
    """Tickers in the prices table with enough history for the factor windows."""
    with engine.connect() as c:
        rows = c.execute(
            text(
                "SELECT ticker FROM prices GROUP BY ticker "
                "HAVING count(*) > :m ORDER BY ticker"
            ),
            {"m": MIN_HISTORY},
        ).fetchall()
    tickers = [r[0] for r in rows]
    return tickers[:limit] if limit else tickers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="train + report, no DB write")
    ap.add_argument("--limit", type=int, default=None, help="cap the universe size")
    ap.add_argument("--top-pct", type=float, default=TOP_PCT)
    args = ap.parse_args()

    engine = create_engine(os.environ["DATABASE_URL"])
    factors = load_selected_factors()
    universe = load_universe(engine, args.limit)
    if len(universe) < 20:
        print(f"Universe too small ({len(universe)}). Backfill prices first.")
        return 1

    print(
        f"Training xs_strong: {len(universe)} tickers, {len(factors)} selected factors, "
        f"top_pct={args.top_pct}, horizon={HORIZON} (reading prices DB)..."
    )

    svc = TrainingService()
    res = svc.train(
        TrainRequest(
            tickers=universe,
            market_provider="db",
            feature_names=factors,
            feature_groups=["xs_selected"],  # label only; feature_names drives columns
            horizon_days=HORIZON,
            label_type="xs_strong",
            top_pct=args.top_pct,
            model_type="xgboost",
            search_mode="none",
            save_model=True,
        )
    )
    if not res.success:
        print(f"Training failed: {res.error}")
        return 1

    m = res.metrics
    print(
        "\n=== xs_strong — test set ===\n"
        f"  rank IC          : {m.get('test_rank_ic')}   (IR {m.get('test_rank_ic_ir')})\n"
        f"  precision@top    : {m.get('test_precision_at_top')}   "
        f"(base {m.get('test_base_rate')}, lift {m.get('test_lift')}x)\n"
        f"  days evaluated   : {m.get('test_n_days')}\n"
        f"  train/val/test   : {res.train_samples}/{res.val_samples}/{res.test_samples}\n"
        f"  model_id         : {res.model_id}"
    )

    if args.dry_run:
        print("\n[dry-run] not writing to the registry.")
        return 0

    extras = {
        "xs_strong": {
            "top_pct": args.top_pct,
            "horizon_days": HORIZON,
            "normalization": "z_score_per_date",
            "min_cross_section": MIN_CROSS_SECTION,
            "n_tickers": len(universe),
            "selected_factors": factors,
        }
    }
    row = dict(
        id=res.model_id,
        name="xgboost_xs_strong",
        mt=res.model_type,
        lt="xs_strong",
        hz=HORIZON,
        tk=res.tickers,
        fg=["xs_selected"],
        fn=res.feature_names,
        nf=res.n_features,
        tr=res.train_samples,
        vl=res.val_samples,
        ts=res.test_samples,
        dr=list(res.train_date_range) if res.train_date_range else None,
        metrics=json.dumps(res.metrics),
        extras=json.dumps(extras),
        ap=res.model_path or "local",
    )
    with engine.begin() as c:
        c.execute(text("DELETE FROM model_registry WHERE id=:id"), {"id": res.model_id})
        c.execute(_INSERT, row)
    blob = zip_model_dir(res.model_path)
    with engine.begin() as c:
        c.execute(
            text(
                "INSERT INTO model_artifacts (model_id, blob) VALUES (:i,:b) "
                "ON CONFLICT (model_id) DO UPDATE SET blob = EXCLUDED.blob"
            ),
            {"i": res.model_id, "b": blob},
        )
    print(f"\nPublished xs_strong model {res.model_id} (registry row + blob).")
    print("Select it with list_models(label_type='xs_strong'). Promotion = Phase D.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
