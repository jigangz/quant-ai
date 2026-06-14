"""Reproducible local micro-benchmarks for resume numbers (single machine).

Measures the latencies that back the UX/performance claims:
  - single prediction inference
  - full-universe batch scoring (the ranking compute)
  - per-name feature engineering
  - end-to-end ranking pipeline over the whole universe (synthetic prices,
    so it times compute, not DB network)

Run:  python -m scripts.bench_resume
"""

from __future__ import annotations

import logging
import statistics
import time

logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
from pathlib import Path

from app.core.settings import settings
from app.ml.features.technical import add_technical_features
from app.ml.models.xgboost_model import XGBoostModel


def bench(fn, n, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000.0)
    return statistics.median(ts), statistics.mean(ts), float(np.percentile(ts, 95))


def _synth_ohlcv(n, seed):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-01", periods=n),
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


def main():
    sp = Path(settings.STORAGE_LOCAL_PATH)
    # full-universe production model (527 tickers = ...plus524...)
    mids = sorted(p.name for p in sp.glob("xgboost_A_AAL_AAPL_plus524*") if p.is_dir())
    model = XGBoostModel.load(sp / mids[-1])
    feats = list(model.metadata.feature_names)
    nf = len(feats)
    tickers = list(model.metadata.tickers)
    N = len(tickers)
    print(f"model={mids[-1]}  {nf} factors  {N} tickers\n")

    # 1. single prediction inference
    X1 = pd.DataFrame(np.random.normal(size=(1, nf)), columns=feats)
    m, a, p95 = bench(lambda: model.predict_proba(X1), 3000)
    print(f"1. single prediction (predict_proba)      p50={m:.3f}ms  mean={a:.3f}ms  p95={p95:.3f}ms")

    # 2. full-universe batch scoring (N names at once)
    XN = pd.DataFrame(np.random.normal(size=(N, nf)), columns=feats)
    m, a, p95 = bench(lambda: model.predict_proba(XN), 500)
    print(f"2. score all {N} names (batch inference)   p50={m:.2f}ms   mean={a:.2f}ms")

    # 3. feature engineering per name (~500 bars -> technical indicators)
    df = _synth_ohlcv(500, 1)
    n_ind = len([c for c in add_technical_features(df).columns
                 if c not in ("date", "open", "high", "low", "close", "volume")])
    m, a, p95 = bench(lambda: add_technical_features(df), 300)
    print(f"3. feature build (500 bars, {n_ind} indicators) p50={m:.2f}ms")

    # 4. end-to-end ranking over the whole universe (synthetic prices = pure compute)
    from app.services.ranking_service import RankingService
    prices = {t: _synth_ohlcv(320, abs(hash(t)) % 10_000) for t in tickers}
    svc = RankingService()
    # warmup once (it's heavy), then time a few
    svc.rank(model=model, universe=tickers, prices=prices, top_n=20)
    ts = []
    for _ in range(5):
        s = time.perf_counter()
        svc.rank(model=model, universe=tickers, prices=prices, top_n=20)
        ts.append((time.perf_counter() - s) * 1000.0)
    print(f"4. rank ENTIRE {N}-name universe end-to-end  p50={statistics.median(ts):.0f}ms "
          f"(feature build + per-date normalize + score + sort, {N} names)")


if __name__ == "__main__":
    main()
