

import pandas as pd
from typing import Tuple, List

FEATURE_COLUMNS: List[str] = [
    "ma_5",
    "ma_20",
    "ema_12",
    "ema_26",
    "macd",
    "rsi_14",
    "volatility_20",
]


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and label vector y.

    Rules:
    - label must exist
    - feature NaN is allowed (handled by imputer later)
    """
    df = df.copy()

    # 只移除 label 不存在的行（未来不可用）
    df = df.dropna(subset=["label"])

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    return X, y
