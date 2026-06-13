"""V5 Phase E — Top-N cross-sectional ranking portfolio backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _SignalModel:
    """Fake model: predict_proba[:,1] is monotonic in the 'signal' factor."""

    def predict_proba(self, X):
        s = (X["signal"] if "signal" in X.columns else X.iloc[:, 0]).to_numpy(dtype=float)
        rng = s.max() - s.min()
        p = (s - s.min()) / rng if rng > 0 else np.full(len(s), 0.5)
        return np.column_stack([1.0 - p, p])


# ===================================================================
# E1 — chaining + selection + cost math
# ===================================================================

def test_backtest_chaining_math_stable_selection():
    """Hand-computed 2-period case pins chaining, selection, and turnover=0."""
    from app.backtest.xs_portfolio import backtest_xs_portfolio

    # 4 names, 2 dates; signal ranks A>B>C>D (so top-2 = A,B both dates -> no churn)
    rows = []
    fr = {
        "2025-01-01": {"A": 0.10, "B": 0.05, "C": -0.02, "D": -0.05},
        "2025-01-02": {"A": 0.04, "B": 0.02, "C": 0.00, "D": -0.06},
    }
    sig = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    for d, m in fr.items():
        for t, r in m.items():
            rows.append({"date": d, "ticker": t, "signal": sig[t], "future_return": r})
    panel = pd.DataFrame(rows)

    out = backtest_xs_portfolio(
        panel, _SignalModel(), factor_cols=["signal"], top_pct=0.5, H=1, cost_bps=10
    )
    assert out["success"]
    assert out["n_rebalances"] == 2
    # long top-2 = A,B: period rets 0.075 then 0.03 -> (1.075)(1.03)-1 = 10.725%
    assert out["gross"]["metrics"]["total_return"] == pytest.approx(10.73, abs=0.05)
    # benchmark = mean of all 4: 0.02 then 0.0 -> 2.0%
    assert out["benchmark"]["metrics"]["total_return"] == pytest.approx(2.0, abs=0.02)
    # selection never changed -> no turnover -> net == gross
    assert out["avg_turnover"] == pytest.approx(0.0, abs=1e-9)
    assert out["net"]["metrics"]["total_return"] == pytest.approx(
        out["gross"]["metrics"]["total_return"], abs=1e-6
    )


def test_costs_reduce_net_when_turnover():
    """When the basket churns, net return is below gross."""
    from app.backtest.xs_portfolio import backtest_xs_portfolio

    # date1 ranks A,B on top; date2 flips to C,D -> full turnover
    rows = [
        {"date": "2025-01-01", "ticker": "A", "signal": 9, "future_return": 0.05},
        {"date": "2025-01-01", "ticker": "B", "signal": 8, "future_return": 0.04},
        {"date": "2025-01-01", "ticker": "C", "signal": 1, "future_return": 0.01},
        {"date": "2025-01-01", "ticker": "D", "signal": 0, "future_return": 0.00},
        {"date": "2025-01-02", "ticker": "A", "signal": 0, "future_return": 0.00},
        {"date": "2025-01-02", "ticker": "B", "signal": 1, "future_return": 0.01},
        {"date": "2025-01-02", "ticker": "C", "signal": 8, "future_return": 0.05},
        {"date": "2025-01-02", "ticker": "D", "signal": 9, "future_return": 0.06},
    ]
    out = backtest_xs_portfolio(
        pd.DataFrame(rows), _SignalModel(), factor_cols=["signal"],
        top_pct=0.5, H=1, cost_bps=50,
    )
    assert out["avg_turnover"] > 0.0
    assert out["net"]["metrics"]["total_return"] < out["gross"]["metrics"]["total_return"]


def _planted_panel(n_tickers=12, n_dates=8, seed=0):
    """Factor 'signal' predicts future_return (planted positive correlation)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        sig = rng.permutation(np.linspace(-1.0, 1.0, n_tickers))
        for t, s in zip(tickers, sig):
            rows.append({
                "date": d, "ticker": t, "signal": float(s),
                "future_return": 0.02 * s + rng.normal(0, 0.001),
            })
    return pd.DataFrame(rows)


def test_planted_signal_long_only_beats_benchmark_gross():
    from app.backtest.xs_portfolio import backtest_xs_portfolio

    out = backtest_xs_portfolio(
        _planted_panel(), _SignalModel(), factor_cols=["signal"], top_pct=0.25, H=1, cost_bps=10
    )
    assert out["success"]
    assert (
        out["gross"]["metrics"]["total_return"]
        > out["benchmark"]["metrics"]["total_return"]
    )


def test_long_short_positive_with_signal():
    from app.backtest.xs_portfolio import backtest_xs_portfolio

    out = backtest_xs_portfolio(
        _planted_panel(), _SignalModel(), factor_cols=["signal"],
        top_pct=0.25, H=1, cost_bps=10, long_short=True,
    )
    assert out["success"]
    # long strong - short weak, with planted signal -> positive gross
    assert out["gross"]["metrics"]["total_return"] > 0.0


# ===================================================================
# E2 — OOS panel builder (reuse Phase C/D feature pipeline)
# ===================================================================

_FACTORS = ["rsi_14", "returns_5d", "bb_position", "volume_ratio"]


class _Meta:
    feature_names = _FACTORS
    train_end_date = "2024-04-01"   # earlier; must be IGNORED in favor of test_start
    test_start_date = "2024-06-01"  # the honest OOS boundary


class _ModelWithMeta:
    metadata = _Meta()

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def _synth_prices(tickers, n=260):
    out = {}
    for t in tickers:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        dates = pd.bdate_range("2024-01-01", periods=n)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        out[t] = pd.DataFrame({
            "ticker": t, "date": dates, "open": close,
            "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        })
    return out


def test_build_backtest_panel_is_oos_only():
    from app.backtest.xs_portfolio import build_backtest_panel

    prices = _synth_prices(["AA", "BB", "CC", "DD", "EE", "FF"])
    panel, factor_cols = build_backtest_panel(_ModelWithMeta(), prices, H=5)

    assert factor_cols == _FACTORS
    assert len(panel) > 0
    # OOS cutoff prefers test_start_date over the (earlier) train_end_date
    assert (panel["date"] > pd.Timestamp("2024-06-01")).all()
    # carries factors + the realized H-day forward return
    for c in _FACTORS + ["future_return", "ticker", "date"]:
        assert c in panel.columns
    assert panel["future_return"].notna().all()
    # per-date normalized: each date's factor mean ~ 0
    for col in _FACTORS:
        means = panel.groupby("date")[col].mean().abs()
        assert (means < 1e-6).all()


def test_build_backtest_panel_since_override():
    from app.backtest.xs_portfolio import build_backtest_panel

    prices = _synth_prices(["AA", "BB", "CC", "DD", "EE", "FF"])
    panel, _ = build_backtest_panel(_ModelWithMeta(), prices, H=5, since="2024-09-01")
    # explicit `since` overrides the metadata boundary
    assert (panel["date"] > pd.Timestamp("2024-09-01")).all()
