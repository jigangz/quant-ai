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
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
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
            alpha = (returns.mean() - beta * benchmark_returns.mean()) * periods_per_year
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
        f"**Period:** {config.get('start_date', 'N/A')} → {config.get('end_date', 'N/A')}",
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
