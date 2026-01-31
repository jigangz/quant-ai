"""
Backtest API

POST /backtest - Run backtest on a trained model
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.backtest.engine import BacktestEngine, BacktestRequest, BacktestResult

logger = logging.getLogger(__name__)
router = APIRouter()


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
