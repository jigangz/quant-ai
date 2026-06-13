"""Cross-sectional ranking — proof-of-concept eval (V5 Phase C core).

Reads the prices table (NOT yahoo), builds a panel, labels each day's
top-30% forward-5d return as the "strong" group, normalizes factors per
date (the survivorship-safe + leak-safe way), trains XGBoost, and reports
the metrics that actually matter for stock selection: precision@30% and
rank IC — NOT the direction AUC.

Usage:
    python -m scripts.xs_eval [--tickers N] [--horizon 5] [--top 0.30]

Standalone PoC: does not touch the API / training pipeline. Validates the
method end to end before Phase C wires it into the product.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import text

from app.db.engine import engine
from app.ml.features.cross_section import cross_section_normalize
from app.ml.features.technical import add_technical_features

EXCLUDE = ["date", "ticker", "open", "high", "low", "close", "volume", "fwd_ret", "label"]


def load_panel(n_tickers: int | None) -> pd.DataFrame:
    with engine.connect() as c:
        tickers = [
            r[0]
            for r in c.execute(
                text(
                    "SELECT ticker FROM prices GROUP BY ticker "
                    "HAVING count(*) > 400 ORDER BY ticker"
                )
            )
        ]
    if n_tickers:
        tickers = tickers[:n_tickers]
    frames = []
    with engine.connect() as c:
        for t in tickers:
            rows = c.execute(
                text(
                    "SELECT ticker, date, open, high, low, close, volume "
                    "FROM prices WHERE ticker=:t ORDER BY date"
                ),
                {"t": t},
            ).mappings().all()
            if len(rows) < 400:
                continue
            df = pd.DataFrame([dict(r) for r in rows])
            df = add_technical_features(df)
            frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel, tickers


def add_xs_label(panel: pd.DataFrame, horizon: int, top: float) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    # forward horizon-day return per ticker (label uses the future — that's
    # correct for a label; features never look forward)
    panel["fwd_ret"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.shift(-horizon) / s - 1.0
    )
    panel = panel.dropna(subset=["fwd_ret"])
    # cross-sectional label: each day, top `top`% forward return = strong (1)
    panel["label"] = panel.groupby("date")["fwd_ret"].transform(
        lambda s: (s >= s.quantile(1.0 - top)).astype(int)
    )
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top", type=float, default=0.30)
    args = ap.parse_args()

    import xgboost as xgb

    print(f"Loading panel ({args.tickers} tickers)...")
    panel, tickers = load_panel(args.tickers)
    print(f"  {panel['ticker'].nunique()} tickers, {len(panel):,} rows")

    panel = add_xs_label(panel, args.horizon, args.top)
    factor_cols = [c for c in panel.columns if c not in EXCLUDE]
    panel = cross_section_normalize(panel, factor_cols).dropna(subset=factor_cols)

    # split by date: first 70% train, last 30% test (no date crosses)
    dates = np.sort(panel["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    train = panel[panel["date"] < cut]
    test = panel[panel["date"] >= cut]
    print(f"  train {len(train):,} rows / test {len(test):,} rows ({len(factor_cols)} factors)")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    model.fit(train[factor_cols], train["label"])
    test = test.copy()
    test["score"] = model.predict_proba(test[factor_cols])[:, 1]

    # --- the metrics that matter for selection ---
    precisions, rank_ics = [], []
    for _, g in test.groupby("date"):
        if len(g) < 5:
            continue
        k = max(1, int(len(g) * args.top))
        picked = g.nlargest(k, "score")
        precisions.append(picked["label"].mean())  # of our top-k, how many were truly strong
        ic, _ = spearmanr(g["score"], g["fwd_ret"])
        if ic == ic:
            rank_ics.append(ic)

    base_rate = test["label"].mean()
    print("\n=== Cross-sectional ranking — test set ===")
    print(f"  precision@{int(args.top*100)}%  : {np.mean(precisions):.3f}   (random baseline = {base_rate:.3f})")
    print(f"  lift over random : {np.mean(precisions) / base_rate:.2f}x")
    print(f"  mean rank IC     : {np.mean(rank_ics):.4f}   (IR = {np.mean(rank_ics)/ (np.std(rank_ics)+1e-9):.2f})")
    print(f"  days evaluated   : {len(precisions)}")
    print("\n(precision@30% > base rate = the model ranks strong stocks above")
    print(" random. rank IC > 0 = predicted order tracks realized order.)")
    return 0


if __name__ == "__main__":
    main()
