import pandas as pd


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
    df["ma_20"] = df["close"].rolling(window=20).mean()

    # ===== Exponential Moving Averages =====
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ===== MACD =====
    df["macd"] = df["ema_12"] - df["ema_26"]

    # ===== Volatility =====
    df["volatility_20"] = df["close"].rolling(window=20).std()

    # ===== RSI (14) =====
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df
