import pandas as pd
from typing import Tuple, List, Dict, Iterable

# =========================
# Feature group definition
# =========================

FEATURE_GROUPS: Dict[str, List[str]] = {
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

DEFAULT_FEATURE_GROUPS: Iterable[str] = ("technical",)


def build_xy(
    df: pd.DataFrame,
    feature_groups: Iterable[str] = DEFAULT_FEATURE_GROUPS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and label vector y.

    Rules:
    - 'label' must exist
    - Feature NaN is allowed (handled later by imputer)
    - Feature selection is controlled by feature_groups
    """

    df = df.copy()

    # 只移除 label 不存在的行（未来不可用）
    df = df.dropna(subset=["label"])

    # 收集 feature columns（按 group）
    feature_columns: List[str] = []
    for group in feature_groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group}")
        feature_columns.extend(FEATURE_GROUPS[group])

    X = df[feature_columns]
    y = df["label"]

    return X, y
