"""
Technical Indicators

Adds technical analysis features to OHLCV data.
All features use only past data (no future leakage).
"""

import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV dataframe.

    Assumes:
    - df contains columns: date, open, high, low, close, volume
    - df is time-series data for ONE ticker
    - NO future data leakage (uses only past data)

    Returns:
    - DataFrame with new feature columns
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ===== Moving Averages =====
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ma_50"] = df["close"].rolling(window=50).mean()

    # ===== Exponential Moving Averages =====
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ===== MACD =====
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ===== Volatility =====
    df["volatility_5"] = df["close"].rolling(window=5).std()
    df["volatility_20"] = df["close"].rolling(window=20).std()

    # ===== Bollinger Bands =====
    df["bb_middle"] = df["ma_20"]
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    # ===== RSI (Relative Strength Index) =====
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ===== Stochastic Oscillator =====
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ===== Average True Range (ATR) =====
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()

    # ===== Volume Features =====
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    # ===== Price Features =====
    df["returns_1d"] = df["close"].pct_change(1)
    df["returns_5d"] = df["close"].pct_change(5)
    df["returns_20d"] = df["close"].pct_change(20)

    # ===== Price Position =====
    df["price_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["price_vs_ma50"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    # Clean up intermediate columns
    df = df.drop(columns=["bb_std"], errors="ignore")

    return df


# Feature group definitions for DatasetBuilder
FEATURE_GROUPS = {
    "ta_basic": [
        "ma_5",
        "ma_10",
        "ma_20",
        "ema_12",
        "ema_26",
    ],
    "ta_advanced": [
        "macd",
        "macd_signal",
        "macd_hist",
        "stoch_k",
        "stoch_d",
    ],
    "momentum": [
        "rsi_14",
        "returns_1d",
        "returns_5d",
        "returns_20d",
    ],
    "volatility": [
        "volatility_5",
        "volatility_20",
        "atr_14",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_position",
    ],
    "volume": [
        "volume_ma_20",
        "volume_ratio",
    ],
    "price_position": [
        "price_vs_ma20",
        "price_vs_ma50",
    ],
    # Legacy group for backward compatibility
    "technical": [
        "ma_5",
        "ma_20",
        "ema_12",
        "ema_26",
        "macd",
        "rsi_14",
        "volatility_20",
    ],
}

# All available features
ALL_FEATURES = list(set(feat for group in FEATURE_GROUPS.values() for feat in group))
