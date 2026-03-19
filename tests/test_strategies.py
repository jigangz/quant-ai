"""
Tests for Strategy System

Tests for:
- BaseStrategy abstract class
- Built-in strategy templates (MA Cross, RSI, Bollinger, Sentiment)
- StrategyRegistry
- Strategy API endpoints
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.strategies import (
    get_registry,
    StrategyRegistry,
    MovingAverageCrossover,
    RSIStrategy,
    BollingerBreakout,
    SentimentDriven,
)


# ===================================
# Fixtures
# ===================================

@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV price data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    
    # Generate realistic price movement
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    open_price = close + np.random.randn(100) * 0.5
    volume = np.random.randint(1000000, 10000000, 100)
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    
    return df


@pytest.fixture
def sample_sentiment_data(sample_price_data):
    """Add sentiment scores to price data."""
    df = sample_price_data.copy()
    np.random.seed(42)
    df["sentiment_score"] = np.random.uniform(-0.8, 0.8, len(df))
    return df


@pytest.fixture
def fresh_registry():
    """Create a fresh registry for testing."""
    return StrategyRegistry()


# ===================================
# Test BaseStrategy
# ===================================

def test_base_strategy_metadata():
    """Test that strategies have required metadata."""
    strategy = MovingAverageCrossover()
    metadata = strategy.get_metadata()
    
    assert metadata.name == "ma_crossover"
    assert metadata.description
    assert metadata.version
    assert metadata.parameters_schema
    assert "close" in metadata.required_columns


def test_base_strategy_validation_missing_column(sample_price_data):
    """Test data validation with missing columns."""
    strategy = MovingAverageCrossover()
    df_missing = sample_price_data.drop(columns=["close"])
    
    missing = strategy.validate_data(df_missing)
    assert "close" in missing


def test_base_strategy_validation_empty_df():
    """Test validation with empty DataFrame."""
    strategy = MovingAverageCrossover()
    empty_df = pd.DataFrame()
    
    missing = strategy.validate_data(empty_df)
    assert len(missing) > 0


# ===================================
# Test Moving Average Crossover
# ===================================

def test_ma_crossover_init_params():
    """Test MA crossover initialization with parameters."""
    params = {
        "fast_period": 5,
        "slow_period": 20,
        "ma_type": "ema"
    }
    strategy = MovingAverageCrossover(**params)
    
    assert strategy.params.fast_period == 5
    assert strategy.params.slow_period == 20
    assert strategy.params.ma_type == "ema"


def test_ma_crossover_invalid_periods():
    """Test that fast >= slow raises error."""
    with pytest.raises(ValueError, match="fast_period.*must be less than.*slow_period"):
        MovingAverageCrossover(fast_period=50, slow_period=20)


def test_ma_crossover_generates_signals(sample_price_data):
    """Test that MA crossover generates valid signals."""
    strategy = MovingAverageCrossover(
        fast_period=5,
        slow_period=20,
        ma_type="sma"
    )
    
    signals = strategy.generate_signals(sample_price_data)
    
    assert len(signals) == len(sample_price_data)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert signals.dtype == int


def test_ma_crossover_signal_modes(sample_price_data):
    """Test cross-only vs persistent signal modes."""
    # Cross-only mode
    strategy_cross = MovingAverageCrossover(
        fast_period=5,
        slow_period=20,
        signal_on_cross_only=True
    )
    signals_cross = strategy_cross.generate_signals(sample_price_data)
    
    # Persistent mode
    strategy_persist = MovingAverageCrossover(
        fast_period=5,
        slow_period=20,
        signal_on_cross_only=False
    )
    signals_persist = strategy_persist.generate_signals(sample_price_data)
    
    # Persistent should have more non-zero signals
    assert (signals_persist != 0).sum() >= (signals_cross != 0).sum()


# ===================================
# Test RSI Strategy
# ===================================

def test_rsi_strategy_init():
    """Test RSI strategy initialization."""
    strategy = RSIStrategy(
        rsi_period=14,
        overbought=70,
        oversold=30
    )
    
    assert strategy.params.rsi_period == 14
    assert strategy.params.overbought == 70
    assert strategy.params.oversold == 30


def test_rsi_invalid_thresholds():
    """Test that oversold >= overbought raises error."""
    with pytest.raises(ValueError, match="oversold must be less than overbought"):
        RSIStrategy(overbought=30, oversold=70)


def test_rsi_generates_signals(sample_price_data):
    """Test RSI generates valid signals."""
    strategy = RSIStrategy(
        rsi_period=14,
        overbought=70,
        oversold=30,
        exit_on_neutral=True
    )
    
    signals = strategy.generate_signals(sample_price_data)
    
    assert len(signals) == len(sample_price_data)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_rsi_extreme_conditions():
    """Test RSI with extreme overbought/oversold values."""
    # Create trending data
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    close = 100 + np.arange(50) * 2  # Strong uptrend
    df = pd.DataFrame({"close": close}, index=dates)
    
    strategy = RSIStrategy(rsi_period=14, overbought=65, oversold=35)
    signals = strategy.generate_signals(df)
    
    # Should generate some short signals in strong uptrend
    assert (signals == -1).any() or (signals == 0).all()


# ===================================
# Test Bollinger Breakout
# ===================================

def test_bollinger_init():
    """Test Bollinger Bands initialization."""
    strategy = BollingerBreakout(
        bb_period=20,
        bb_std=2.0,
        strategy_mode="breakout"
    )
    
    assert strategy.params.bb_period == 20
    assert strategy.params.bb_std == 2.0


def test_bollinger_generates_signals(sample_price_data):
    """Test Bollinger Bands generates signals."""
    strategy = BollingerBreakout(
        bb_period=20,
        bb_std=2.0,
        strategy_mode="breakout",
        use_close_cross=True
    )
    
    signals = strategy.generate_signals(sample_price_data)
    
    assert len(signals) == len(sample_price_data)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_bollinger_breakout_vs_mean_reversion(sample_price_data):
    """Test that breakout and mean-reversion modes give opposite signals."""
    strategy_breakout = BollingerBreakout(
        bb_period=20,
        strategy_mode="breakout",
        use_close_cross=True
    )
    
    strategy_reversion = BollingerBreakout(
        bb_period=20,
        strategy_mode="mean_reversion",
        use_close_cross=True
    )
    
    signals_breakout = strategy_breakout.generate_signals(sample_price_data)
    signals_reversion = strategy_reversion.generate_signals(sample_price_data)
    
    # Where one signals long, other should signal short (or neutral)
    non_zero_idx = (signals_breakout != 0) & (signals_reversion != 0)
    if non_zero_idx.any():
        assert (signals_breakout[non_zero_idx] == -signals_reversion[non_zero_idx]).all()


def test_bollinger_high_low_mode(sample_price_data):
    """Test Bollinger with high/low instead of close."""
    strategy = BollingerBreakout(
        bb_period=20,
        use_close_cross=False  # Use high/low
    )
    
    # Should require high and low columns
    assert "high" in strategy.required_columns
    assert "low" in strategy.required_columns
    
    signals = strategy.generate_signals(sample_price_data)
    assert len(signals) == len(sample_price_data)


# ===================================
# Test Sentiment Driven
# ===================================

def test_sentiment_init():
    """Test sentiment strategy initialization."""
    strategy = SentimentDriven(
        positive_threshold=0.5,
        negative_threshold=-0.5,
        sentiment_column="sentiment_score"
    )
    
    assert strategy.params.positive_threshold == 0.5
    assert strategy.params.negative_threshold == -0.5


def test_sentiment_generates_signals(sample_sentiment_data):
    """Test sentiment strategy generates signals."""
    strategy = SentimentDriven(
        positive_threshold=0.3,
        negative_threshold=-0.3,
        smoothing_period=1
    )
    
    signals = strategy.generate_signals(sample_sentiment_data)
    
    assert len(signals) == len(sample_sentiment_data)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_sentiment_smoothing(sample_sentiment_data):
    """Test sentiment smoothing reduces signal noise."""
    strategy_no_smooth = SentimentDriven(
        positive_threshold=0.3,
        negative_threshold=-0.3,
        smoothing_period=1
    )
    
    strategy_smooth = SentimentDriven(
        positive_threshold=0.3,
        negative_threshold=-0.3,
        smoothing_period=5
    )
    
    signals_no_smooth = strategy_no_smooth.generate_signals(sample_sentiment_data)
    signals_smooth = strategy_smooth.generate_signals(sample_sentiment_data)
    
    # Count signal changes
    changes_no_smooth = (signals_no_smooth.diff() != 0).sum()
    changes_smooth = (signals_smooth.diff() != 0).sum()
    
    # Smoothing should reduce changes
    assert changes_smooth <= changes_no_smooth


def test_sentiment_confirmation(sample_sentiment_data):
    """Test sentiment confirmation requirement."""
    strategy = SentimentDriven(
        positive_threshold=0.2,
        negative_threshold=-0.2,
        require_confirmation=True,
        confirmation_periods=3
    )
    
    signals = strategy.generate_signals(sample_sentiment_data)
    
    # With confirmation, should have fewer signals
    assert len(signals) == len(sample_sentiment_data)


def test_sentiment_missing_column(sample_price_data):
    """Test sentiment strategy with missing sentiment column."""
    strategy = SentimentDriven()
    
    # sample_price_data doesn't have sentiment_score
    missing = strategy.validate_data(sample_price_data)
    assert "sentiment_score" in missing


# ===================================
# Test StrategyRegistry
# ===================================

def test_registry_builtin_strategies(fresh_registry):
    """Test that registry comes with built-in strategies."""
    assert len(fresh_registry) == 4
    assert "ma_crossover" in fresh_registry
    assert "rsi_strategy" in fresh_registry
    assert "bollinger_breakout" in fresh_registry
    assert "sentiment_driven" in fresh_registry


def test_registry_get_strategy(fresh_registry):
    """Test getting strategy from registry."""
    StrategyClass = fresh_registry.get("ma_crossover")
    assert StrategyClass is MovingAverageCrossover
    
    # Not found
    assert fresh_registry.get("nonexistent") is None


def test_registry_get_or_raise(fresh_registry):
    """Test get_or_raise method."""
    StrategyClass = fresh_registry.get_or_raise("ma_crossover")
    assert StrategyClass is MovingAverageCrossover
    
    with pytest.raises(KeyError, match="not found"):
        fresh_registry.get_or_raise("nonexistent")


def test_registry_list_strategies(fresh_registry):
    """Test listing strategies."""
    strategies = fresh_registry.list_strategies()
    assert len(strategies) == 4
    assert "ma_crossover" in strategies


def test_registry_list_metadata(fresh_registry):
    """Test listing metadata."""
    metadata_list = fresh_registry.list_metadata()
    assert len(metadata_list) == 4
    
    for metadata in metadata_list:
        assert metadata.name
        assert metadata.description
        assert metadata.version


def test_registry_get_metadata(fresh_registry):
    """Test getting metadata for specific strategy."""
    metadata = fresh_registry.get_metadata("ma_crossover")
    assert metadata.name == "ma_crossover"
    assert metadata.parameters_schema
    
    # Not found
    assert fresh_registry.get_metadata("nonexistent") is None


def test_registry_create_instance(fresh_registry):
    """Test creating strategy instance via registry."""
    strategy = fresh_registry.create_instance(
        "ma_crossover",
        parameters={"fast_period": 5, "slow_period": 20}
    )
    
    assert isinstance(strategy, MovingAverageCrossover)
    assert strategy.params.fast_period == 5


def test_registry_create_instance_no_params(fresh_registry):
    """Test creating instance without parameters."""
    strategy = fresh_registry.create_instance("rsi_strategy")
    assert isinstance(strategy, RSIStrategy)


def test_registry_create_instance_not_found(fresh_registry):
    """Test creating instance of non-existent strategy."""
    with pytest.raises(KeyError):
        fresh_registry.create_instance("nonexistent")


# ===================================
# Test Global Registry
# ===================================

def test_global_registry_singleton():
    """Test that get_registry returns same instance."""
    registry1 = get_registry()
    registry2 = get_registry()
    assert registry1 is registry2


# ===================================
# Integration Tests
# ===================================

def test_strategy_end_to_end(sample_price_data):
    """Test complete workflow: registry -> strategy -> signals."""
    registry = get_registry()
    
    # Get strategy
    StrategyClass = registry.get("ma_crossover")
    
    # Create instance
    strategy = StrategyClass(fast_period=5, slow_period=20)
    
    # Validate data
    missing = strategy.validate_data(sample_price_data)
    assert len(missing) == 0
    
    # Generate signals
    signals = strategy.generate_signals(sample_price_data)
    
    assert len(signals) == len(sample_price_data)
    assert signals.dtype == int


def test_all_strategies_work(sample_price_data, sample_sentiment_data):
    """Test that all built-in strategies can generate signals."""
    registry = get_registry()
    
    for strategy_name in registry.list_strategies():
        StrategyClass = registry.get(strategy_name)
        strategy = StrategyClass()
        
        # Use appropriate data
        if strategy_name == "sentiment_driven":
            df = sample_sentiment_data
        else:
            df = sample_price_data
        
        # Should not raise
        signals = strategy.generate_signals(df)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1, 0, 1})
