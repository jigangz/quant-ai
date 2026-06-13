from __future__ import annotations

"""Cross-sectional ranking service — V5 Phase D.

Serves the xs_strong model as a live "today's Top-N strong stocks" board: load
the model, build each universe name's latest features, normalize them as ONE
cross-section (the same per-date z-score used at train time — no stored scaler,
see app/ml/features/cross_section.py), score, and rank.

The score is predict_proba[:,1] = the model's probability that a name lands in
the next-horizon top-`top_pct` by return — a relative RANK signal, not a price
target.
"""

import logging
from typing import Optional

import pandas as pd

from app.db.model_registry import get_model_registry
from app.db.prices_repo import get_prices_batch
from app.ml.features.cross_section import cross_section_normalize
from app.ml.features.technical import add_technical_features
from app.services.predict_service import get_model

logger = logging.getLogger(__name__)

SCORE_SEMANTICS = (
    "Strength score (0-1): the model's cross-sectional probability that this "
    "name is in the next-5d top-30% by return. A relative RANK signal across "
    "the universe — NOT a price target or expected return."
)

# Minimum price rows needed to build the (long-window) factors meaningfully.
MIN_HISTORY = 60


def _list_models(**kwargs):
    """Indirection over the registry list (monkeypatchable in tests)."""
    return get_model_registry().list_models(**kwargs)


def default_xs_model_id() -> Optional[str]:
    """The default ranking model = the most recent active xs_strong model.

    (Label-scoped promotion — choosing among several xs models — is deferred;
    one xs model exists today, so 'latest active' is the default.)
    """
    rows = _list_models(label_type="xs_strong", limit=1)
    return rows[0].id if rows else None


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg, "rankings": []}


class RankingService:
    """Scores and ranks the universe with the cross-sectional xs_strong model."""

    def rank(
        self,
        model_id: Optional[str] = None,
        top_n: int = 20,
        universe: Optional[list[str]] = None,
        prices: Optional[dict] = None,
        model=None,
    ) -> dict:
        # 1. Resolve the model (explicit object > model_id > latest xs model).
        if model is None:
            mid = model_id or default_xs_model_id()
            if not mid:
                return _fail("No xs_strong model available. Publish one first.")
            model = get_model(mid)
            model_id = mid
            if model is None:
                return _fail(f"Model {mid} could not be loaded.")
        meta = getattr(model, "metadata", None)
        feature_names = list(getattr(meta, "feature_names", []) or [])
        if not feature_names:
            return _fail("Model has no feature_names metadata.")

        universe = universe or list(getattr(meta, "tickers", []) or [])
        if not universe:
            return _fail("Empty universe.")

        # 2. Fetch prices (batch in prod; injected in tests).
        if prices is None:
            prices = get_prices_batch(universe)

        # 3. One latest-features row per ticker that has enough history.
        #    Skip names whose latest row is NaN in any model factor — long-window
        #    factors (e.g. mom_12_1 needs 252 rows) are NaN for thin-history names,
        #    and training dropna'd exactly those rows. Excluding them here keeps the
        #    serve universe consistent with the train contract (instead of letting
        #    the pipeline's imputer silently fill them with the train median).
        records, as_of = [], None
        for t in universe:
            df = prices.get(t)
            if df is None or len(df) < MIN_HISTORY:
                continue
            last = add_technical_features(df).iloc[-1]
            vals = {f: last.get(f) for f in feature_names}
            if any(pd.isna(v) for v in vals.values()):
                continue
            records.append({"ticker": t, **vals})
            d = last.get("date")
            if d is not None:
                ts = pd.Timestamp(d)
                as_of = ts if as_of is None else max(as_of, ts)

        if not records:
            return _fail("No tickers had enough data to score.")

        # 4. Normalize as ONE cross-section (single synthetic date), then score.
        panel = pd.DataFrame(records)
        panel["date"] = "today"
        panel = cross_section_normalize(panel, feature_names)
        scores = model.predict_proba(panel[feature_names])[:, 1]
        panel = (
            panel.assign(score=scores)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        n = len(panel)
        rankings = [
            {
                "rank": i + 1,
                "ticker": row["ticker"],
                "score": round(float(row["score"]), 4),
                "percentile": round(100.0 * (1.0 - i / n), 2) if n > 1 else 100.0,
            }
            for i, row in panel.head(top_n).iterrows()
        ]

        return {
            "success": True,
            "model_id": model_id,
            "as_of": as_of.date().isoformat() if as_of is not None else None,
            "universe_size": len(universe),
            "scored": n,
            "top_n": top_n,
            "rankings": rankings,
            "score_semantics": SCORE_SEMANTICS,
        }


def backtest_ranking(
    model_id: Optional[str] = None,
    top_pct: float = 0.10,
    cost_bps: int = 10,
    long_short: bool = False,
    *,
    model=None,
    universe: Optional[list[str]] = None,
    prices: Optional[dict] = None,
) -> dict:
    """Out-of-sample Top-N portfolio backtest of the ranking model (V5 Phase E).

    Loads the xs model, builds the held-out (test-split) panel, runs the
    cross-sectional backtest, and returns compact equity curves + metrics
    (NET of costs) for the strategy and the equal-weight-universe benchmark —
    shaped for the frontend Backtest card.
    """
    from app.backtest.xs_portfolio import backtest_xs_portfolio, build_backtest_panel

    H = 5
    if model is None:
        mid = model_id or default_xs_model_id()
        if not mid:
            return _fail("No xs_strong model available. Publish one first.")
        model = get_model(mid)
        model_id = mid
        if model is None:
            return _fail(f"Model {mid} could not be loaded.")

    universe = universe or list(getattr(getattr(model, "metadata", None), "tickers", []) or [])
    if not universe:
        return _fail("Empty universe.")
    if prices is None:
        prices = get_prices_batch(universe)

    panel, factor_cols = build_backtest_panel(model, prices, H=H)
    if panel is None or panel.empty:
        return _fail("No out-of-sample data to backtest.")

    bt = backtest_xs_portfolio(
        panel, model, factor_cols=factor_cols,
        top_pct=top_pct, H=H, cost_bps=cost_bps, long_short=long_short,
    )
    if not bt.get("success"):
        return _fail(bt.get("error", "Backtest produced no rebalances."))

    def _slim(cell: dict) -> dict:
        m = cell["metrics"]
        return {
            "sharpe": m.get("sharpe"), "cagr": m.get("cagr"),
            "max_drawdown": m.get("max_drawdown"), "total_return": m.get("total_return"),
            "equity": cell["equity"],
        }

    dates = bt["dates"]
    return {
        "success": True,
        "model_id": model_id,
        "oos_start": dates[0] if dates else None,
        "oos_end": dates[-1] if dates else None,
        "n_rebalances": bt["n_rebalances"],
        "top_pct": top_pct,
        "cost_bps": cost_bps,
        "long_short": long_short,
        "avg_turnover": round(bt["avg_turnover"], 4),
        "dates": dates,
        "strategy": _slim(bt["net"]),       # net of transaction costs (the honest one)
        "benchmark": _slim(bt["benchmark"]),
        "caveat": (
            f"Out-of-sample (held-out test split, ~{bt['n_rebalances']} rebalances) net of "
            f"{cost_bps}bps/side vs an equal-weight-universe benchmark. Short, single-regime "
            "window — a directional read, not a robust Sharpe estimate."
        ),
    }
