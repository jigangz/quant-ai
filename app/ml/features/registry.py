"""
Feature Registry

Centralized registry for feature groups and feature computation.
Allows dynamic registration and selection of features without if/else.
"""

import logging
from typing import Protocol
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureComputer(Protocol):
    """Protocol for feature computation functions."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features and return DataFrame with new columns."""
        ...


@dataclass
class FeatureGroup:
    """A group of related features."""

    name: str
    description: str
    feature_names: list[str]
    compute_fn: FeatureComputer
    dependencies: list[str] = field(default_factory=list)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute this feature group."""
        return self.compute_fn(df)


class FeatureRegistry:
    """
    Central registry for all feature groups.

    Usage:
        registry = FeatureRegistry()
        registry.register(FeatureGroup(...))

        # Get specific groups
        df = registry.compute(df, groups=["ta_basic", "momentum"])

        # Get all available groups
        registry.list_groups()
    """

    _instance = None
    _groups: dict[str, FeatureGroup] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._groups = {}
        return cls._instance

    def register(self, group: FeatureGroup) -> None:
        """Register a feature group."""
        if group.name in self._groups:
            logger.warning(f"Overwriting feature group: {group.name}")
        self._groups[group.name] = group
        logger.debug(f"Registered feature group: {group.name}")

    def get(self, name: str) -> FeatureGroup | None:
        """Get a feature group by name."""
        return self._groups.get(name)

    def list_groups(self) -> list[str]:
        """List all registered group names."""
        return list(self._groups.keys())

    def get_group_info(self) -> list[dict]:
        """Get info about all registered groups."""
        return [
            {
                "name": g.name,
                "description": g.description,
                "features": g.feature_names,
                "dependencies": g.dependencies,
            }
            for g in self._groups.values()
        ]

    def get_feature_names(self, groups: list[str]) -> list[str]:
        """Get all feature names for the specified groups."""
        features = []
        seen = set()

        for group_name in groups:
            group = self._groups.get(group_name)
            if group:
                for feat in group.feature_names:
                    if feat not in seen:
                        features.append(feat)
                        seen.add(feat)
            else:
                logger.warning(f"Unknown feature group: {group_name}")

        return features

    def compute(
        self,
        df: pd.DataFrame,
        groups: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute features for the specified groups.

        Args:
            df: Input DataFrame with OHLCV data
            groups: List of group names to compute (None = all)

        Returns:
            DataFrame with all computed features
        """
        if groups is None:
            groups = self.list_groups()

        # Resolve dependencies
        resolved_groups = self._resolve_dependencies(groups)

        # Compute each group
        for group_name in resolved_groups:
            group = self._groups.get(group_name)
            if group:
                logger.debug(f"Computing feature group: {group_name}")
                df = group.compute(df)

        return df

    def _resolve_dependencies(self, groups: list[str]) -> list[str]:
        """Resolve dependencies and return groups in correct order."""
        resolved = []
        visited = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            group = self._groups.get(name)
            if group:
                for dep in group.dependencies:
                    visit(dep)
                if name not in resolved:
                    resolved.append(name)

        for group in groups:
            visit(group)

        return resolved

    def clear(self) -> None:
        """Clear all registered groups (mainly for testing)."""
        self._groups.clear()


# Global registry instance
feature_registry = FeatureRegistry()


# ===================================
# Feature Computation Functions
# ===================================


def compute_ta_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic technical indicators."""
    df = df.copy()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    return df


def compute_ta_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced technical indicators (requires ta_basic)."""
    df = df.copy()

    # Ensure EMAs exist
    if "ema_12" not in df.columns:
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    if "ema_26" not in df.columns:
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    return df


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum indicators."""
    df = df.copy()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Returns
    df["returns_1d"] = df["close"].pct_change(1)
    df["returns_5d"] = df["close"].pct_change(5)
    df["returns_20d"] = df["close"].pct_change(20)

    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility indicators."""
    df = df.copy()

    # Standard deviation
    df["volatility_5"] = df["close"].rolling(window=5).std()
    df["volatility_20"] = df["close"].rolling(window=20).std()

    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    import pandas as pd

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()

    # Bollinger Bands
    ma_20 = df["close"].rolling(window=20).mean()
    std_20 = df["close"].rolling(window=20).std()
    df["bb_upper"] = ma_20 + 2 * std_20
    df["bb_lower"] = ma_20 - 2 * std_20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / ma_20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    return df


def compute_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume indicators."""
    df = df.copy()
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
    return df


def compute_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price position indicators (requires ta_basic)."""
    df = df.copy()

    # Ensure MAs exist
    if "ma_20" not in df.columns:
        df["ma_20"] = df["close"].rolling(window=20).mean()
    if "ma_50" not in df.columns:
        df["ma_50"] = df["close"].rolling(window=50).mean()

    df["price_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["price_vs_ma50"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    return df


# ===================================
# Register Default Feature Groups
# ===================================


def register_default_groups():
    """Register all default feature groups."""

    feature_registry.register(
        FeatureGroup(
            name="ta_basic",
            description="Basic technical indicators (MA, EMA)",
            feature_names=["ma_5", "ma_10", "ma_20", "ema_12", "ema_26"],
            compute_fn=compute_ta_basic,
        )
    )

    feature_registry.register(
        FeatureGroup(
            name="ta_advanced",
            description="Advanced technical indicators (MACD, Stochastic)",
            feature_names=["macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d"],
            compute_fn=compute_ta_advanced,
            dependencies=["ta_basic"],
        )
    )

    feature_registry.register(
        FeatureGroup(
            name="momentum",
            description="Momentum indicators (RSI, Returns)",
            feature_names=["rsi_14", "returns_1d", "returns_5d", "returns_20d"],
            compute_fn=compute_momentum,
        )
    )

    feature_registry.register(
        FeatureGroup(
            name="volatility",
            description="Volatility indicators (ATR, Bollinger Bands)",
            feature_names=[
                "volatility_5",
                "volatility_20",
                "atr_14",
                "bb_upper",
                "bb_lower",
                "bb_width",
                "bb_position",
            ],
            compute_fn=compute_volatility,
        )
    )

    feature_registry.register(
        FeatureGroup(
            name="volume",
            description="Volume indicators",
            feature_names=["volume_ma_20", "volume_ratio"],
            compute_fn=compute_volume,
        )
    )

    feature_registry.register(
        FeatureGroup(
            name="price_position",
            description="Price position relative to moving averages",
            feature_names=["price_vs_ma20", "price_vs_ma50"],
            compute_fn=compute_price_position,
            dependencies=["ta_basic"],
        )
    )


# Auto-register on import
register_default_groups()
