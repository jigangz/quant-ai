from __future__ import annotations

"""Cross-sectional Top-N ranking portfolio backtest — V5 Phase E.

Turns the xs_strong ranking signal into an honest portfolio backtest, distinct
from the per-ticker timing engine in app/backtest/engine.py:

  Every `H` days (= the label horizon, so rebalances use non-overlapping H-day
  returns) score the cross-section, equal-weight the top `top_pct` names (and,
  for long_short, short the bottom), hold to the next rebalance, and chain the
  realized H-day returns into an equity curve. Transaction costs are charged on
  turnover (names that churn between rebalances). Benchmark = equal-weight of the
  WHOLE universe each period — the fair cross-sectional benchmark that isolates
  selection skill. Metrics reuse calculate_strategy_metrics.

The panel passed in is expected to already carry per-date-normalized factor
columns + a `future_return` (the H-day forward return) per (date, ticker) — see
build_backtest_panel. The backtest itself does not re-normalize.
"""

import numpy as np
import pandas as pd

from app.backtest.metrics import calculate_strategy_metrics


def _rebalance_dates(dates, H: int) -> list:
    """Every H-th unique date (non-overlapping H-day holding periods)."""
    uniq = sorted(pd.unique(dates))
    return uniq[::H]


def build_backtest_panel(model, prices: dict, H: int = 5, since=None):
    """Build the out-of-sample, per-date-normalized panel for the backtest.

    Reuses the Phase C/D feature pipeline: per ticker add_technical_features +
    H-day forward return, concat, keep only dates AFTER the OOS cutoff, then
    per-date normalize the model's factors (the same cross_section_normalize used
    at train and serve) and drop rows NaN in any factor.

    OOS cutoff precedence: explicit `since` > model.metadata.test_start_date (the
    honest val/test boundary — after fit AND tuning) > train_end_date (legacy;
    includes the val split, which only leaks if the model was hyperparameter-tuned).

    Returns:
        (panel, factor_cols) where panel has [date, ticker, *factor_cols,
        future_return]. Empty panel if no data / no OOS dates.
    """
    from app.ml.features.technical import add_technical_features
    from app.ml.features.cross_section import cross_section_normalize
    from app.ml.labels.xs_strong import add_xs_forward_return
    from app.ml.dataset.schemas import LabelConfig

    meta = getattr(model, "metadata", None)
    factor_cols = list(getattr(meta, "feature_names", []) or [])
    cutoff = (
        since
        or getattr(meta, "test_start_date", None)
        or getattr(meta, "train_end_date", None)
    )
    cfg = LabelConfig(label_type="xs_strong", horizon_days=H)

    frames = []
    for ticker, df in prices.items():
        if df is None or len(df) < 60:
            continue
        f = add_technical_features(df)
        f = add_xs_forward_return(f, cfg)  # adds future_price/future_return (+ provisional label)
        f = f.dropna(subset=["future_return"]).copy()
        f["ticker"] = ticker
        frames.append(f)

    if not frames:
        return pd.DataFrame(), factor_cols

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    if cutoff is not None:
        panel = panel[panel["date"] > pd.Timestamp(cutoff)]
    if panel.empty:
        return panel, factor_cols

    panel = cross_section_normalize(panel, factor_cols)
    panel = panel.dropna(subset=factor_cols).reset_index(drop=True)
    return panel, factor_cols


def backtest_xs_portfolio(
    panel: pd.DataFrame,
    model,
    *,
    factor_cols: list[str],
    top_pct: float = 0.10,
    H: int = 5,
    cost_bps: int = 10,
    long_short: bool = False,
) -> dict:
    """Backtest the Top-N (and optional long-short) ranking portfolio.

    Args:
        panel: long frame [date, ticker, *factor_cols, future_return]; factors
            already per-date normalized, future_return = H-day forward return.
        model: anything with predict_proba(X)[:,1] = strength score.
        factor_cols: the model's feature columns (order matters).
        top_pct: fraction of names in each leg.
        H: rebalance interval in trading days (matches the label horizon).
        cost_bps: per-side transaction cost in basis points (charged on turnover).
        long_short: also short the bottom `top_pct` (market-neutral).

    Returns:
        dict with success, n_rebalances, avg_turnover, and gross/net/benchmark
        each carrying {metrics, equity}.
    """
    rebals = _rebalance_dates(panel["date"], H)
    if len(rebals) < 2:
        return {"success": False, "error": "Not enough rebalance dates.",
                "n_rebalances": len(rebals)}

    long_rets, bench_rets, turnovers = [], [], []
    prev_long: set = set()
    prev_short: set = set()

    for t in rebals:
        g = panel[panel["date"] == t]
        if len(g) < 2:
            continue
        scores = model.predict_proba(g[factor_cols])[:, 1]
        gg = g.assign(_score=scores)
        k = max(1, int(round(len(gg) * top_pct)))
        ranked = gg.sort_values("_score", ascending=False)

        long_names = ranked.head(k)
        long_ret = float(long_names["future_return"].mean())
        cur_long = set(long_names["ticker"])
        # one-sided turnover = fraction of the long leg that is new this rebalance
        turn = len(cur_long - prev_long) / k if prev_long else 0.0

        if long_short:
            short_names = ranked.tail(k)
            short_ret = float(short_names["future_return"].mean())
            period_ret = long_ret - short_ret
            cur_short = set(short_names["ticker"])
            turn += len(cur_short - prev_short) / k if prev_short else 0.0
            prev_short = cur_short
        else:
            period_ret = long_ret

        long_rets.append(period_ret)
        bench_rets.append(float(gg["future_return"].mean()))
        turnovers.append(turn)
        prev_long = cur_long

    if len(long_rets) < 2:
        return {"success": False, "error": "Not enough scored rebalances.",
                "n_rebalances": len(long_rets)}

    cost_rate = cost_bps / 1e4
    gross = pd.Series(long_rets)
    # round-trip cost: sell the leaving fraction + buy the entering fraction
    net = gross - 2.0 * pd.Series(turnovers) * cost_rate
    bench = pd.Series(bench_rets)
    ppy = 252.0 / H

    def _curve(s: pd.Series) -> list:
        return [float(x) for x in (1.0 + s).cumprod()]

    return {
        "success": True,
        "n_rebalances": len(long_rets),
        "H": H,
        "top_pct": top_pct,
        "long_short": long_short,
        "cost_bps": cost_bps,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "gross": {"metrics": calculate_strategy_metrics(gross, bench, periods_per_year=ppy),
                  "equity": _curve(gross)},
        "net": {"metrics": calculate_strategy_metrics(net, bench, periods_per_year=ppy),
                "equity": _curve(net)},
        "benchmark": {"metrics": calculate_strategy_metrics(bench, periods_per_year=ppy),
                      "equity": _curve(bench)},
    }
