from __future__ import annotations

"""
News Sentiment Features

Adds news sentiment features to OHLCV data by joining with
news data from the database.
"""

import logging

import pandas as pd

from app.db.news_repo import get_news_by_date_range

logger = logging.getLogger(__name__)


def add_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add news sentiment features to an OHLCV dataframe.

    Assumes:
        - df contains columns: ticker, date, close (at minimum)
        - df is time-series data for ONE ticker
        - NO future data leakage (uses only past data)

    Features added:
        - news_count: number of articles on that day
        - sentiment_score: average sentiment on that day
        - sentiment_positive_ratio: fraction of positive articles
        - sentiment_negative_ratio: fraction of negative articles
        - sentiment_ma_3: 3-day moving average of sentiment
        - sentiment_ma_7: 7-day moving average of sentiment
        - sentiment_momentum: sentiment change (today - 3d MA)

    Returns:
        DataFrame with new feature columns. Missing values filled with 0.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return df

    # Get date range and ticker
    ticker = df["ticker"].iloc[0] if "ticker" in df.columns else None
    if not ticker:
        logger.warning("No ticker column found, skipping news features")
        _fill_empty(df)
        return df

    # Convert dates for querying
    dates = pd.to_datetime(df["date"])
    start_date = (dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")

    # Fetch news from DB
    news_rows = get_news_by_date_range(ticker, start_date, end_date)

    if not news_rows:
        logger.info(f"No news data found for {ticker}, filling with zeros")
        _fill_empty(df)
        return df

    # Aggregate news by date
    news_df = pd.DataFrame(news_rows)
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")

    daily_agg = (
        news_df.groupby("date")
        .agg(
            news_count=("headline", "count"),
            sentiment_score=("sentiment_score", "mean"),
            sentiment_positive_ratio=(
                "sentiment_score",
                lambda x: (x > 0.1).sum() / max(len(x), 1),
            ),
            sentiment_negative_ratio=(
                "sentiment_score",
                lambda x: (x < -0.1).sum() / max(len(x), 1),
            ),
        )
        .reset_index()
    )

    # Merge with main df
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.merge(daily_agg, left_on="date_str", right_on="date", how="left", suffixes=("", "_news"))

    # Drop merge artifacts
    if "date_news" in df.columns:
        df = df.drop(columns=["date_news"])
    df = df.drop(columns=["date_str"], errors="ignore")

    # Moving averages
    df["sentiment_ma_3"] = df["sentiment_score"].rolling(window=3, min_periods=1).mean()
    df["sentiment_ma_7"] = df["sentiment_score"].rolling(window=7, min_periods=1).mean()

    # Momentum: current sentiment - 3d MA
    df["sentiment_momentum"] = df["sentiment_score"] - df["sentiment_ma_3"]

    # Fill missing values with 0
    news_cols = [
        "news_count",
        "sentiment_score",
        "sentiment_positive_ratio",
        "sentiment_negative_ratio",
        "sentiment_ma_3",
        "sentiment_ma_7",
        "sentiment_momentum",
    ]
    for col in news_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    return df


def _fill_empty(df: pd.DataFrame) -> None:
    """Fill all news feature columns with zeros."""
    for col in [
        "news_count",
        "sentiment_score",
        "sentiment_positive_ratio",
        "sentiment_negative_ratio",
        "sentiment_ma_3",
        "sentiment_ma_7",
        "sentiment_momentum",
    ]:
        df[col] = 0.0


# Feature group definitions for FeatureRegistry
NEWS_FEATURE_NAMES = [
    "news_count",
    "sentiment_score",
    "sentiment_positive_ratio",
    "sentiment_negative_ratio",
    "sentiment_ma_3",
    "sentiment_ma_7",
    "sentiment_momentum",
]
