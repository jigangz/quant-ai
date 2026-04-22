"""
Tests for app/ml/labels/ module — V4 Pivot Phase 2 (2026-04-22, Day 2).

Scope:
- LabelConfig schema extension (Literal accepts volatility + meta_label)
- app.ml.labels.registry dispatcher (direction + return work via registry)
- Backward compat: DatasetBuilder._add_labels output unchanged for direction/return
- Volatility + meta_label raise NotImplementedError (implementation in Day 3+ / P3)
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from app.ml.dataset.schemas import LabelConfig
from app.ml.labels.registry import add_labels, LABEL_GENERATORS


# ==========================================================================
# LabelConfig schema tests (V4 extension)
# ==========================================================================


class TestLabelConfigSchema:
    """LabelConfig.label_type Literal accepts all 4 V4 target types."""

    def test_default_label_type_is_direction(self):
        """Backward compat: default remains 'direction'."""
        cfg = LabelConfig()
        assert cfg.label_type == "direction"

    def test_direction_label_type_accepted(self):
        cfg = LabelConfig(label_type="direction")
        assert cfg.label_type == "direction"

    def test_return_label_type_accepted(self):
        cfg = LabelConfig(label_type="return")
        assert cfg.label_type == "return"

    def test_volatility_label_type_accepted(self):
        """V4 Phase 2: volatility target."""
        cfg = LabelConfig(label_type="volatility")
        assert cfg.label_type == "volatility"

    def test_meta_label_label_type_accepted(self):
        """V4 Phase 3: meta-labeling target."""
        cfg = LabelConfig(label_type="meta_label")
        assert cfg.label_type == "meta_label"

    def test_invalid_label_type_rejected(self):
        """Typos / unknown types must raise ValidationError."""
        with pytest.raises(ValidationError):
            LabelConfig(label_type="invalid_xyz")

    def test_horizon_days_default(self):
        cfg = LabelConfig()
        assert cfg.horizon_days == 5

    def test_threshold_default(self):
        cfg = LabelConfig()
        assert cfg.threshold == 0.0


# ==========================================================================
# Registry dispatcher tests
# ==========================================================================


def _make_monotonic_rising_df(n: int = 20) -> pd.DataFrame:
    """Synthetic OHLC with close strictly rising — direction all 1."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "ticker": ["AAPL"] * n,
            "close": [100.0 + i for i in range(n)],
        }
    )


def _make_monotonic_falling_df(n: int = 20) -> pd.DataFrame:
    """Synthetic OHLC with close strictly falling — direction all 0."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "ticker": ["AAPL"] * n,
            "close": [100.0 - i for i in range(n)],
        }
    )


class TestRegistry:
    """LABEL_GENERATORS dict covers all 4 types; add_labels dispatches correctly."""

    def test_registry_has_all_four_types(self):
        """Registry must expose all 4 V4 types."""
        assert "direction" in LABEL_GENERATORS
        assert "return" in LABEL_GENERATORS
        assert "volatility" in LABEL_GENERATORS
        assert "meta_label" in LABEL_GENERATORS

    def test_direction_rising_produces_all_ones_for_valid_rows(self):
        """Rows with valid future → label = 1 for strictly rising close."""
        df = _make_monotonic_rising_df(n=20)
        cfg = LabelConfig(label_type="direction", horizon_days=5, threshold=0.0)
        result = add_labels(df, cfg)

        # Filter rows that have a valid future (exclude tail with NaN future_return)
        valid = result[result["future_return"].notna()]
        assert len(valid) == 15  # 20 - 5 horizon
        assert (valid["label"] == 1).all()

    def test_direction_falling_produces_all_zeros_for_valid_rows(self):
        """Rows with valid future → label = 0 for strictly falling close."""
        df = _make_monotonic_falling_df(n=20)
        cfg = LabelConfig(label_type="direction", horizon_days=5, threshold=0.0)
        result = add_labels(df, cfg)

        valid = result[result["future_return"].notna()]
        assert (valid["label"] == 0).all()

    def test_return_produces_float_labels_with_nan_tail(self):
        """label_type=return → labels are raw future returns (float);
        tail rows without future data have NaN label (dropped by caller)."""
        df = _make_monotonic_rising_df(n=20)
        cfg = LabelConfig(label_type="return", horizon_days=5)
        result = add_labels(df, cfg)

        # Return-type label preserves NaN in tail
        labeled = result.dropna(subset=["label"])
        assert len(labeled) == 15
        assert labeled["label"].dtype in (float, "float64", "float32")
        assert (labeled["label"] > 0).all()

    def test_direction_preserves_zero_label_for_nan_future(self):
        """Backward-compat quirk: direction astype(int) makes NaN future → 0 label
        (not NaN). Behavior preserved through registry refactor."""
        n, horizon = 20, 5
        df = _make_monotonic_rising_df(n=n)
        cfg = LabelConfig(label_type="direction", horizon_days=horizon)
        result = add_labels(df, cfg)

        # All rows kept (registry does not drop)
        assert len(result) == n
        # Tail `horizon` rows have NaN future_return...
        tail = result[result["future_return"].isna()]
        assert len(tail) == horizon
        # ...but their label is 0 (NaN > threshold → False → 0 via astype(int))
        assert (tail["label"] == 0).all()

    def test_meta_label_raises_not_implemented(self):
        """V4 Phase 3 implementation target — Day 2 placeholder."""
        df = _make_monotonic_rising_df()
        cfg = LabelConfig(label_type="meta_label", horizon_days=5)
        with pytest.raises(NotImplementedError, match="meta_label"):
            add_labels(df, cfg)


# ==========================================================================
# Volatility label generator (V4 Phase 2)
# ==========================================================================


class TestVolatility:
    """add_volatility_label computes forward-looking realized vol."""

    def test_constant_close_produces_zero_vol(self):
        """Constant price → zero daily returns → zero realized vol."""
        n = 20
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "ticker": ["AAPL"] * n,
                "close": [100.0] * n,
            }
        )
        cfg = LabelConfig(label_type="volatility", horizon_days=5)
        result = add_labels(df, cfg)

        valid = result.dropna(subset=["label"])
        assert len(valid) > 0
        assert (valid["label"] == 0).all()

    def test_rising_close_produces_positive_vol(self):
        """Strictly rising close → non-zero returns → positive vol."""
        df = _make_monotonic_rising_df(n=20)
        cfg = LabelConfig(label_type="volatility", horizon_days=5)
        result = add_labels(df, cfg)

        valid = result.dropna(subset=["label"])
        assert len(valid) > 0
        assert (valid["label"] > 0).all()

    def test_nan_tail_for_insufficient_future(self):
        """Last `horizon` rows lack future data → NaN label."""
        n, horizon = 20, 5
        df = _make_monotonic_rising_df(n=n)
        cfg = LabelConfig(label_type="volatility", horizon_days=horizon)
        result = add_labels(df, cfg)

        # Last `horizon` rows get NaN (not enough forward returns)
        tail = result.iloc[-horizon:]
        assert tail["label"].isna().all()

    def test_annualize_multiplies_by_sqrt_252(self):
        """volatility_annualize=True multiplies raw vol by sqrt(252)."""
        # Synthetic: alternating ±1% return gives constant vol
        prices = [100.0]
        for i in range(25):
            prices.append(prices[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
                "ticker": ["AAPL"] * len(prices),
                "close": prices,
            }
        )

        cfg_ann = LabelConfig(
            label_type="volatility", horizon_days=5, volatility_annualize=True
        )
        cfg_raw = LabelConfig(
            label_type="volatility", horizon_days=5, volatility_annualize=False
        )

        ann = add_labels(df, cfg_ann).dropna(subset=["label"])
        raw = add_labels(df, cfg_raw).dropna(subset=["label"])

        import math

        ratio = ann["label"].iloc[0] / raw["label"].iloc[0]
        assert abs(ratio - math.sqrt(252)) < 1e-6

    def test_vol_magnitude_reasonable_for_toy_data(self):
        """1% daily alternating swings → annualized vol ~ 1% * sqrt(252) ≈ 15.9%."""
        import math

        prices = [100.0]
        for i in range(30):
            prices.append(prices[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
                "ticker": ["AAPL"] * len(prices),
                "close": prices,
            }
        )
        cfg = LabelConfig(
            label_type="volatility", horizon_days=10, volatility_annualize=True
        )
        result = add_labels(df, cfg).dropna(subset=["label"])

        # Daily vol ~ 0.01 (1% log returns); annualized ~ 0.01 * sqrt(252) ≈ 0.159
        median_vol = result["label"].median()
        assert 0.10 < median_vol < 0.25, (
            f"Annualized vol outside sane range for 1% daily swings: {median_vol}"
        )


# ==========================================================================
# DatasetBuilder backward compat
# ==========================================================================


class TestBuilderBackwardCompat:
    """DatasetBuilder._add_labels after refactor must produce same output for direction/return."""

    def test_builder_add_labels_direction_backward_compat(self):
        """DatasetBuilder._add_labels(label_type=direction) after registry refactor
        produces byte-identical output to pre-refactor (int labels, 0 for NaN future)."""
        from app.ml.dataset.builder import DatasetBuilder
        from app.ml.dataset.schemas import DatasetConfig

        df = _make_monotonic_rising_df(n=20)
        config = DatasetConfig(
            tickers=["AAPL"],
            label_config=LabelConfig(label_type="direction", horizon_days=5),
        )
        builder = DatasetBuilder(config)
        result = builder._add_labels(df)

        # All rows retained; 15 with valid future all have label=1
        assert len(result) == 20
        valid = result[result["future_return"].notna()]
        assert len(valid) == 15
        assert (valid["label"] == 1).all()

    def test_builder_add_labels_return_backward_compat(self):
        """label_type=return → labels == future_return (float, NaN in tail)."""
        from app.ml.dataset.builder import DatasetBuilder
        from app.ml.dataset.schemas import DatasetConfig

        df = _make_monotonic_rising_df(n=20)
        config = DatasetConfig(
            tickers=["AAPL"],
            label_config=LabelConfig(label_type="return", horizon_days=5),
        )
        builder = DatasetBuilder(config)
        result = builder._add_labels(df)

        labeled = result.dropna(subset=["label"])
        assert len(labeled) == 15
        assert (labeled["label"] == labeled["future_return"]).all()
