"""
V4 Pivot · Phase 2 · Volatility Backend Benchmark (D12).

Trains 5 model families (Logistic/Ridge, RandomForest, XGBoost, LightGBM, CatBoost)
against two targets — direction (baseline) and volatility — on 3 tickers
(AAPL, MSFT, GOOGL) using yfinance daily data.

Writes a Markdown comparison report to `docs/benchmarks/v4_volatility_benchmark.md`.

Does NOT touch the Model Registry (Supabase migration may still be pending).
Pure benchmark run — results are printed + persisted to the report file.

Usage:
    cd C:\\Users\\zjg09\\projects\\quant-ai
    .venv\\Scripts\\python -m scripts.v4_volatility_benchmark
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure test-friendly env so we don't accidentally touch Supabase.
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./benchmark_quant.db")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("STORAGE_LOCAL_PATH", "./benchmark_artifacts")
os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("BROKER_BACKEND", "memory")
os.environ.setdefault("QUEUE_BACKEND", "memory")
os.environ.setdefault("NOTIFY_BACKEND", "memory")
os.environ.setdefault("FUNCTIONS_BACKEND", "local")
os.environ.setdefault("REDIS_URL", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("v4-benchmark")

# Eager-import these at module scope to avoid circular-import headaches when
# different targets are exercised in sequence. All V4 Pivot modules loaded here.
import pandas as pd
import yfinance as yf

from app.ml.features.technical import add_technical_features
from app.ml.features.registry import feature_registry
from app.ml.labels.registry import add_labels
from app.ml.dataset.schemas import (
    DatasetConfig,
    DatasetOutput,
    DatasetResult,
    LabelConfig,
    SplitConfig,
    TickerDataset,
)
from app.ml.models import get_model
from app.backtest.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


TICKERS = ["AAPL", "MSFT", "GOOGL"]
MODELS = ["logistic", "random_forest", "xgboost", "lightgbm", "catboost"]
TARGETS = ["direction", "volatility"]
HORIZON = 5


def fetch_prices(ticker: str, start: date, end: date):
    """Fetch OHLCV from yfinance and return a DataFrame with expected schema."""
    logger.info(f"Fetching {ticker} ({start} -> {end})")
    hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    if hist.empty:
        raise ValueError(f"No data for {ticker}")

    df = hist.reset_index()
    # yfinance returns columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def build_dataset_from_df(raw_df, label_type: str, tickers: list[str]):
    """Bypass the market provider and feed a pre-fetched DataFrame through feature+label pipeline."""
    label_config = LabelConfig(label_type=label_type, horizon_days=HORIZON)
    split_config = SplitConfig(train_ratio=0.7, val_ratio=0.15)
    dataset_config = DatasetConfig(
        tickers=tickers,
        feature_groups=["ta_basic"],
        label_config=label_config,
        split_config=split_config,
    )

    all_dfs = []
    ticker_stats = []
    for ticker in tickers:
        df_ticker = raw_df[raw_df["ticker"] == ticker].copy()
        df_ticker = df_ticker.sort_values("date").reset_index(drop=True)
        df_ticker = add_technical_features(df_ticker)
        df_ticker = add_labels(df_ticker, label_config)

        # Drop rows without labels (tail of series for regression; no-op for direction int labels)
        df_ticker = df_ticker.dropna(subset=["label"])

        # Drop rows with NaN features
        feature_cols = feature_registry.get_feature_names(dataset_config.feature_groups)
        existing = [c for c in feature_cols if c in df_ticker.columns]
        df_ticker = df_ticker.dropna(subset=existing)

        if len(df_ticker) < 100:
            logger.warning(f"{ticker}: only {len(df_ticker)} rows after clean; skipping")
            continue

        all_dfs.append(df_ticker)
        ticker_stats.append(
            TickerDataset(
                ticker=ticker,
                n_samples=len(df_ticker),
                n_features=len(existing),
                date_range=(
                    df_ticker["date"].min().isoformat(),
                    df_ticker["date"].max().isoformat(),
                ),
                label_distribution={},
            )
        )

    combined = pd.concat(all_dfs, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    # Time-series split by date
    unique_dates = combined["date"].unique()
    train_end_idx = int(len(unique_dates) * split_config.train_ratio)
    val_end_idx = int(len(unique_dates) * (split_config.train_ratio + split_config.val_ratio))
    train_end_date = unique_dates[train_end_idx - 1]
    val_end_date = unique_dates[val_end_idx - 1]

    train_df = combined[combined["date"] <= train_end_date]
    val_df = combined[(combined["date"] > train_end_date) & (combined["date"] <= val_end_date)]
    test_df = combined[combined["date"] > val_end_date]

    feature_cols = [c for c in existing if c in combined.columns]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    meta = DatasetResult(
        config=dataset_config,
        tickers_processed=tickers,
        total_samples=len(combined),
        n_features=len(feature_cols),
        feature_names=feature_cols,
        ticker_stats=ticker_stats,
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=len(test_df),
        train_date_range=(train_df["date"].min().isoformat(), train_df["date"].max().isoformat()),
        val_date_range=(val_df["date"].min().isoformat(), val_df["date"].max().isoformat()),
        test_date_range=(test_df["date"].min().isoformat(), test_df["date"].max().isoformat()),
    )

    return DatasetOutput(X_train, y_train, X_val, y_val, X_test, y_test, meta)


def train_and_evaluate(
    model_type: str, task: str, dataset, time_budget_sec: int = 60
) -> dict[str, Any]:
    """Train one model, evaluate on val + test, return metrics dict."""
    logger.info(f"  [{task}] {model_type} - training...")
    start = time.time()
    model = get_model(model_type, task=task)
    model.fit(dataset.X_train, dataset.y_train)
    train_time = time.time() - start

    # Evaluate on val + test
    row = {"model": model_type, "train_time_s": round(train_time, 2)}
    for split_name, X, y in [("val", dataset.X_val, dataset.y_val), ("test", dataset.X_test, dataset.y_test)]:
        if len(X) == 0:
            continue
        # Drop NaN labels
        import pandas as pd
        mask = pd.Series(y).notna().values
        X_c, y_c = X.loc[mask], pd.Series(y).loc[mask].values
        if len(y_c) == 0:
            continue

        y_pred = model.predict(X_c)
        if task == "classification":
            y_prob = model.predict_proba(X_c)[:, 1]
            m = calculate_classification_metrics(y_c, y_pred, y_prob)
        else:
            m = calculate_regression_metrics(y_c, y_pred)

        for k, v in m.items():
            if v is not None:
                row[f"{split_name}_{k}"] = v

    return row


def main() -> int:
    from app.db.engine import engine  # noqa: delayed for env to be set first
    from sqlalchemy import text

    # Ensure local sqlite has prices table (the conftest pattern)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume INTEGER,
                    UNIQUE(ticker, date)
                )
                """
            )
        )

    end = date.today()
    start = end - timedelta(days=730)

    logger.info(f"Fetching {len(TICKERS)} tickers from yfinance...")
    raw_frames = []
    for ticker in TICKERS:
        try:
            raw_frames.append(fetch_prices(ticker, start, end))
        except Exception as e:
            logger.error(f"{ticker}: {e}")

    import pandas as pd

    raw_df = pd.concat(raw_frames, ignore_index=True) if raw_frames else None
    if raw_df is None or raw_df.empty:
        logger.error("No data fetched. Aborting.")
        return 1
    logger.info(f"Total rows: {len(raw_df)} across {raw_df['ticker'].nunique()} tickers")

    results: dict[str, list[dict]] = {"direction": [], "volatility": []}

    for target in TARGETS:
        task = "classification" if target == "direction" else "regression"
        logger.info(f"\n===== Target: {target} (task={task}) =====")

        try:
            dataset = build_dataset_from_df(raw_df, target, TICKERS)
            logger.info(
                f"Dataset: train={dataset.metadata.train_samples}, "
                f"val={dataset.metadata.val_samples}, "
                f"test={dataset.metadata.test_samples}, "
                f"features={dataset.metadata.n_features}"
            )
        except Exception as e:
            logger.error(f"Failed to build dataset for {target}: {e}")
            continue

        for model_type in MODELS:
            try:
                row = train_and_evaluate(model_type, task, dataset)
                results[target].append(row)
                logger.info(f"    -> {row}")
            except Exception as e:
                logger.error(f"    {model_type} failed: {e}")
                results[target].append({"model": model_type, "error": str(e)})

    # ==========================================================================
    # Write Markdown report
    # ==========================================================================
    report_path = Path("docs/benchmarks/v4_volatility_benchmark.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# V4 Pivot · Phase 2 · Volatility Backend Benchmark (D12)")
    lines.append("")
    lines.append(f"**Run date**: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append(f"**Tickers**: {', '.join(TICKERS)}")
    lines.append(f"**Data window**: {start} to {end} ({(end - start).days} days)")
    lines.append(f"**Horizon**: {HORIZON} days")
    lines.append("**Features**: ta_basic (OHLC-derived technical indicators)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Compare 5 model families on two targets: **direction** (baseline classification) "
        "vs **volatility** (V4 Phase 2 regression). Goal is to empirically confirm the "
        "V4 Pivot thesis: volatility forecasting is more tractable than direction prediction "
        "(López de Prado · Cochrane · 50-year GARCH literature)."
    )
    lines.append("")

    # Direction table
    lines.append("## Direction (classification baseline)")
    lines.append("")
    direction_rows = results["direction"]
    if direction_rows:
        header_keys = sorted(
            {k for r in direction_rows for k in r.keys() if k != "model"},
            key=lambda k: (k != "train_time_s", k),
        )
        lines.append("| model | " + " | ".join(header_keys) + " |")
        lines.append("|---" * (len(header_keys) + 1) + "|")
        for r in direction_rows:
            cells = [str(r.get("model", ""))] + [str(r.get(k, "—")) for k in header_keys]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Volatility table
    lines.append("## Volatility (regression V4 Phase 2)")
    lines.append("")
    vol_rows = results["volatility"]
    if vol_rows:
        header_keys = sorted(
            {k for r in vol_rows for k in r.keys() if k != "model"},
            key=lambda k: (k != "train_time_s", k),
        )
        lines.append("| model | " + " | ".join(header_keys) + " |")
        lines.append("|---" * (len(header_keys) + 1) + "|")
        for r in vol_rows:
            cells = [str(r.get("model", ""))] + [str(r.get(k, "—")) for k in header_keys]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- **QLIKE** (volatility only) is the Patton (2011) loss for vol forecasts; "
        "lower is better and it strictly penalizes under-forecasts."
    )
    lines.append(
        "- **R²** measures variance explained; for vol the sign and magnitude relative "
        "to a naive mean baseline matter more than the absolute value."
    )
    lines.append(
        "- **Direction AUC ≈ 0.50-0.58** is typical and consistent with near-martingale "
        "price series (López de Prado, *Advances in Financial Machine Learning*, Ch. 2)."
    )
    lines.append(
        "- If the volatility regression models show R² noticeably above 0 on the test split "
        "(while direction AUC hovers around coin-flip), that quantitatively confirms the "
        "V4 Pivot thesis: **the platform rightly trades direction glamour for a target with "
        "real predictability**."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Benchmark runs with default hyperparameters. Hyperparameter tuning (Optuna) "
        "is available via `POST /train?search_mode=optuna` for production runs."
    )
    lines.append(
        "- Ensemble model excluded from this benchmark (V4 P1 limitation: ensemble "
        "regression not yet implemented — will be revisited in post-P1 hardening)."
    )
    lines.append(
        "- Data source: yfinance daily bars; exact rows vary with market calendar."
    )
    lines.append(
        f"- Rerun: `python -m scripts.v4_volatility_benchmark`"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to {report_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
