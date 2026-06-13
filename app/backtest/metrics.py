from __future__ import annotations

"""
Backtest Metrics

Classification metrics:
- AUC, F1, Precision, Recall

Strategy metrics:
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities for class 1

    Returns:
        Dict with accuracy, precision, recall, f1, auc
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    # AUC only if both classes present and probabilities provided
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["auc"] = None

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression metrics for V4 Pivot multi-task ML.

    Used for label_type in {"return", "volatility", "meta_label" (when scored as R)}.

    Args:
        y_true: True target values (continuous).
        y_pred: Predicted target values.

    Returns:
        Dict with MAE, RMSE, MAPE, QLIKE (if y_true is positive), and R^2.
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Drop rows where either is NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "mape": None, "qlike": None, "r2": None}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE — ignore rows where |y_true| < epsilon to avoid div-by-zero blowup
    eps = 1e-8
    safe_mask = np.abs(y_true) > eps
    if safe_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_true[safe_mask] - y_pred[safe_mask]) / y_true[safe_mask]))
        )
    else:
        mape = None

    # QLIKE — volatility-specific loss (Patton 2011);
    # only defined when y_true > 0 AND y_pred > 0 (both positive vol estimates).
    pos_mask = (y_true > eps) & (y_pred > eps)
    if pos_mask.sum() > 0:
        ratio = y_true[pos_mask] / y_pred[pos_mask]
        qlike = float(np.mean(ratio - np.log(ratio) - 1))
    else:
        qlike = None

    # R^2 — only meaningful with variance in y_true
    if np.var(y_true) > eps:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = None

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 6) if mape is not None else None,
        "qlike": round(qlike, 6) if qlike is not None else None,
        "r2": round(r2, 4) if r2 is not None else None,
    }


def calculate_xs_metrics(
    scores: np.ndarray,
    future_return: np.ndarray,
    dates: np.ndarray,
    top_pct: float = 0.30,
    min_names: int = 5,
) -> dict[str, float | None]:
    """Cross-sectional ranking metrics (V5 Phase C) — the numbers that matter
    for stock SELECTION, not global AUC.

    For each date with at least ``min_names`` names:
      - precision@top_pct: of our top-k highest-score names, the fraction that
        were truly in the date's top-k by realized forward return.
      - rank IC: Spearman correlation of predicted score vs realized
        future_return across that date's cross-section.
    Returns the mean over evaluated dates (+ IR, base rate, lift). Mirrors
    scripts/xs_eval.py:112-129. All-thin input → n_days=0 and None metrics.

    Args:
        scores: model score per row (higher = predicted stronger).
        future_return: realized forward return per row.
        dates: the cross-section key per row (same length as scores).
        top_pct: strong-group fraction (matches the label definition).
        min_names: skip dates with fewer names (too thin to rank).
    """
    from scipy.stats import spearmanr

    df = pd.DataFrame(
        {"date": np.asarray(dates), "score": np.asarray(scores, dtype=float),
         "fwd": np.asarray(future_return, dtype=float)}
    )

    precisions: list[float] = []
    rank_ics: list[float] = []
    base_rates: list[float] = []

    for _, g in df.groupby("date"):
        if len(g) < min_names:
            continue
        k = max(1, int(len(g) * top_pct))
        cutoff = g["fwd"].quantile(1.0 - top_pct)
        true_strong = (g["fwd"] >= cutoff).astype(int)
        picked = g.nlargest(k, "score").index
        precisions.append(float(true_strong.loc[picked].mean()))
        base_rates.append(float(true_strong.mean()))
        # Rank IC undefined when either side is constant (e.g. a date where the
        # model emits identical scores) — skip rather than warn.
        if g["score"].nunique() > 1 and g["fwd"].nunique() > 1:
            ic, _ = spearmanr(g["score"], g["fwd"])
            if ic == ic:  # not NaN
                rank_ics.append(float(ic))

    n_days = len(precisions)
    if n_days == 0:
        return {
            "precision_at_top": None,
            "rank_ic": None,
            "rank_ic_ir": None,
            "base_rate": None,
            "lift": None,
            "n_days": 0,
        }

    mean_precision = float(np.mean(precisions))
    mean_base = float(np.mean(base_rates))
    mean_ic = float(np.mean(rank_ics)) if rank_ics else None
    ir = (
        float(np.mean(rank_ics) / (np.std(rank_ics) + 1e-9))
        if len(rank_ics) > 1
        else None
    )
    return {
        "precision_at_top": round(mean_precision, 4),
        "rank_ic": round(mean_ic, 4) if mean_ic is not None else None,
        "rank_ic_ir": round(ir, 4) if ir is not None else None,
        "base_rate": round(mean_base, 4),
        "lift": round(mean_precision / mean_base, 4) if mean_base > 0 else None,
        "n_days": n_days,
    }


def calculate_strategy_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Calculate strategy performance metrics.

    Args:
        returns: Daily returns series
        benchmark_returns: Benchmark returns for comparison
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Trading periods per year (default 252)

    Returns:
        Dict with CAGR, Sharpe, MaxDD, etc.
    """
    if len(returns) == 0:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "n_trades": 0,
        }

    # Total return
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # CAGR
    n_years = len(returns) / periods_per_year
    if n_years > 0 and cumulative.iloc[-1] > 0:
        cagr = (cumulative.iloc[-1] ** (1 / n_years)) - 1
    else:
        cagr = 0.0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio
    if volatility > 0:
        excess_return = returns.mean() * periods_per_year - risk_free_rate
        sharpe = excess_return / volatility
    else:
        sharpe = 0.0

    # Max Drawdown
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    # Win Rate
    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    metrics = {
        "total_return": round(total_return * 100, 2),  # percentage
        "cagr": round(cagr * 100, 2),  # percentage
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown * 100, 2),  # percentage
        "volatility": round(volatility * 100, 2),  # percentage
        "win_rate": round(win_rate * 100, 2),  # percentage
        "profit_factor": (
            round(profit_factor, 2) if profit_factor != float("inf") else None
        ),
        "n_trades": int(total_trades),
    }

    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        bench_total = bench_cumulative.iloc[-1] - 1

        n_years_bench = len(benchmark_returns) / periods_per_year
        if n_years_bench > 0 and bench_cumulative.iloc[-1] > 0:
            bench_cagr = (bench_cumulative.iloc[-1] ** (1 / n_years_bench)) - 1
        else:
            bench_cagr = 0.0

        bench_volatility = benchmark_returns.std() * np.sqrt(periods_per_year)
        if bench_volatility > 0:
            bench_sharpe = (
                benchmark_returns.mean() * periods_per_year - risk_free_rate
            ) / bench_volatility
        else:
            bench_sharpe = 0.0

        bench_rolling_max = bench_cumulative.expanding().max()
        bench_drawdown = (bench_cumulative - bench_rolling_max) / bench_rolling_max
        bench_max_dd = abs(bench_drawdown.min())

        metrics["benchmark"] = {
            "total_return": round(bench_total * 100, 2),
            "cagr": round(bench_cagr * 100, 2),
            "sharpe": round(bench_sharpe, 2),
            "max_drawdown": round(bench_max_dd * 100, 2),
            "volatility": round(bench_volatility * 100, 2),
        }

        # Alpha & Beta
        if len(returns) == len(benchmark_returns) and bench_volatility > 0:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            beta = covariance / (benchmark_returns.std() ** 2)
            excess = returns.mean() - beta * benchmark_returns.mean()
            alpha = excess * periods_per_year
            metrics["alpha"] = round(alpha * 100, 2)
            metrics["beta"] = round(beta, 2)

    return metrics


def generate_report_markdown(
    strategy_metrics: dict,
    classification_metrics: dict,
    config: dict,
) -> str:
    """
    Generate a markdown report from backtest results.

    Args:
        strategy_metrics: Strategy performance metrics
        classification_metrics: Classification metrics
        config: Backtest configuration

    Returns:
        Markdown formatted report
    """
    lines = [
        "# Backtest Report",
        "",
        f"**Model:** {config.get('model_id', 'N/A')}",
        f"**Ticker(s):** {', '.join(config.get('tickers', []))}",
        f"**Period:** {config.get('start_date', 'N/A')} → "
        f"{config.get('end_date', 'N/A')}",
        f"**Transaction Cost:** {config.get('transaction_cost_bps', 10)} bps",
        "",
        "---",
        "",
        "## Strategy Performance",
        "",
        "| Metric | Strategy | Benchmark (B&H) |",
        "|--------|----------|-----------------|",
    ]

    bench = strategy_metrics.get("benchmark", {})
    metrics_rows = [
        ("Total Return", "total_return", "%"),
        ("CAGR", "cagr", "%"),
        ("Sharpe Ratio", "sharpe", ""),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Volatility", "volatility", "%"),
    ]

    for label, key, suffix in metrics_rows:
        strat_val = strategy_metrics.get(key, "N/A")
        bench_val = bench.get(key, "N/A") if bench else "N/A"
        strat_str = f"{strat_val}{suffix}" if strat_val != "N/A" else "N/A"
        bench_str = f"{bench_val}{suffix}" if bench_val != "N/A" else "N/A"
        lines.append(f"| {label} | {strat_str} | {bench_str} |")

    lines.extend([
        "",
        f"**Win Rate:** {strategy_metrics.get('win_rate', 'N/A')}%",
        f"**Profit Factor:** {strategy_metrics.get('profit_factor', 'N/A')}",
        f"**Total Trades:** {strategy_metrics.get('n_trades', 'N/A')}",
    ])

    if "alpha" in strategy_metrics:
        lines.append(f"**Alpha:** {strategy_metrics['alpha']}%")
        lines.append(f"**Beta:** {strategy_metrics['beta']}")

    lines.extend([
        "",
        "---",
        "",
        "## Classification Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])

    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        val = classification_metrics.get(key, "N/A")
        lines.append(f"| {key.upper()} | {val} |")

    lines.extend([
        "",
        "---",
        "",
        "*Generated by Quant-AI Backtest Engine*",
    ])

    return "\n".join(lines)


def calculate_meta_label_metrics(
    y_true,
    y_proba,
    realized_r,
    trade_threshold: float = 0.5,
) -> dict:
    """Meta-label specific metrics.

    Args:
        y_true: binary array (1 if primary was correct).
        y_proba: predicted probabilities from meta-model.
        realized_r: realized R multiples for each event (in trade's favor frame).
        trade_threshold: probability cutoff above which the meta-model recommends trading.

    Returns:
        dict with keys:
          - auc: ROC AUC (if both classes present; else 0.0)
          - precision_at_threshold: fraction of "trade" events that were correct
          - hit_rate_when_trade: identical to precision_at_threshold (alias for readability)
          - expected_R_when_trade: mean realized_R over traded events
          - trade_count: number of events where proba >= trade_threshold
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    realized_r = np.asarray(realized_r)

    trade_mask = y_proba >= trade_threshold
    trade_count = int(trade_mask.sum())

    if trade_count == 0:
        precision = 0.0
        expected_r = 0.0
    else:
        correct_and_trade = (y_true == 1) & trade_mask
        precision = float(correct_and_trade.sum() / trade_count)
        expected_r = float(realized_r[trade_mask].mean())

    if len(np.unique(y_true)) < 2:
        auc = 0.0
    else:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            auc = 0.0

    return {
        "auc": auc,
        "precision_at_threshold": precision,
        "hit_rate_when_trade": precision,  # alias
        "expected_R_when_trade": expected_r,
        "trade_count": trade_count,
    }
