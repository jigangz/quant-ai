from __future__ import annotations

"""
Backtest API

POST /backtest - Run backtest on a trained model
"""

import logging
import threading
from datetime import date

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.backtest.engine import BacktestEngine, BacktestRequest, BacktestResult
from app.services.ranking_service import backtest_ranking

logger = logging.getLogger(__name__)
router = APIRouter()

# Daily cache for the (expensive) ranking backtest — keyed by params + today.
_BT_CACHE: dict = {}
_BT_LOCK = threading.Lock()


# ===================================
# POST /backtest
# ===================================
@router.post("/backtest", response_model=BacktestResult)
def run_backtest(request: BacktestRequest):
    """
    Run a backtest on a trained model.

    Args:
        model_id: ID of the model to backtest
        tickers: Override model's tickers (optional)
        start_date: Backtest start date (optional)
        end_date: Backtest end date (optional)
        signal_threshold: Probability threshold for long signal (default 0.55)
        transaction_cost_bps: Transaction cost in basis points (default 10)
        position_size: Position size as fraction of capital (default 1.0)

    Returns:
        BacktestResult with strategy metrics, classification metrics, and report
    """
    engine = BacktestEngine()
    result = engine.run(request)

    if not result.success:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Backtest failed",
                "message": result.error,
                "model_id": request.model_id,
            },
        )

    return result


# ===================================
# POST /backtest/report
# ===================================
@router.get("/backtest/ranking")
def backtest_ranking_api(
    top_pct: float = Query(0.10, gt=0, le=0.5, description="fraction of names per leg"),
    cost_bps: int = Query(10, ge=0, le=100, description="per-side transaction cost (bps)"),
    long_short: bool = Query(False, description="also short the bottom (market-neutral)"),
):
    """Out-of-sample Top-N ranking portfolio backtest (V5 Phase E).

    Net of costs vs an equal-weight-universe benchmark over the model's held-out
    test split. Cached per (params, day). Returns equity curves + metrics for the
    Backtest card on the ranking page.
    """
    today = date.today().isoformat()
    key = (round(top_pct, 3), cost_bps, long_short, today)
    res = _BT_CACHE.get(key)
    if res is None or not res.get("success"):
        with _BT_LOCK:
            res = _BT_CACHE.get(key)
            if res is None or not res.get("success"):
                res = backtest_ranking(top_pct=top_pct, cost_bps=cost_bps, long_short=long_short)
                if res.get("success"):
                    for k in [k for k in _BT_CACHE if k[3] != today]:
                        _BT_CACHE.pop(k, None)
                    _BT_CACHE[key] = res
    return res


@router.post("/backtest/report", response_class=PlainTextResponse)
def get_backtest_report(request: BacktestRequest):
    """
    Run backtest and return markdown report only.

    Useful for quick review or piping to markdown viewer.
    """
    engine = BacktestEngine()
    result = engine.run(request)

    if not result.success:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Backtest failed",
                "message": result.error,
            },
        )

    return result.report_markdown
