"""
Tests for the strategy system and API.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from app.strategies import (
    get_registry,
    StrategyRegistry,
    BaseStrategy,
    BaseParameters,
    MovingAverageCrossover,
    RSIStrategy,
    BollingerBreakout,
    SentimentDriven,
)


# ===================================
# Registry Tests
# ===================================


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""
    
    def test_get_registry_singleton(self):
        """get_registry returns the same instance."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
    
    def test_builtin_strategies_registered(self):
        """All builtin strategies are registered by default."""
        registry = StrategyRegistry()
        
        assert "ma_crossover" in registry
        assert "rsi_strategy" in registry
        assert "bollinger_breakout" in registry
        assert "sentiment_driven" in registry
        assert len(registry) == 4
    
    def test_list_strategies(self):
        """list_strategies returns all strategy names."""
        registry = StrategyRegistry()
        names = registry.list_strategies()
        
        assert isinstance(names, list)
        assert len(names) == 4
        assert "ma_crossover" in names
    
    def test_get_strategy(self):
        """get returns strategy class or None."""
        registry = StrategyRegistry()
        
        cls = registry.get("ma_crossover")
        assert cls is MovingAverageCrossover
        
        assert registry.get("nonexistent") is None
    
    def test_get_or_raise(self):
        """get_or_raise raises KeyError for unknown strategy."""
        registry = StrategyRegistry()
        
        with pytest.raises(KeyError):
            registry.get_or_raise("nonexistent")
    
    def test_get_metadata(self):
        """get_metadata returns StrategyMetadata."""
        registry = StrategyRegistry()
        
        meta = registry.get_metadata("ma_crossover")
        assert meta is not None
        assert meta.name == "ma_crossover"
        assert meta.version == "1.0.0"
        assert "parameters_schema" in meta.dict()
    
    def test_create_instance(self):
        """create_instance returns strategy with parameters."""
        registry = StrategyRegistry()
        
        # Default parameters
        strategy = registry.create_instance("ma_crossover")
        assert strategy.params.fast_period == 10
        assert strategy.params.slow_period == 50
        
        # Custom parameters
        strategy = registry.create_instance(
            "ma_crossover",
            {"fast_period": 5, "slow_period": 20}
        )
        assert strategy.params.fast_period == 5
        assert strategy.params.slow_period == 20
    
    def test_register_custom_strategy(self):
        """Can register a custom strategy."""
        registry = StrategyRegistry()
        
        class CustomStrategy(BaseStrategy):
            name = "custom_test"
            description = "Test strategy"
            version = "1.0.0"
            
            def generate_signals(self, df):
                return pd.Series(0, index=df.index)
        
        registry.register(CustomStrategy)
        assert "custom_test" in registry
        assert registry.get("custom_test") is CustomStrategy
    
    def test_register_duplicate_raises(self):
        """Cannot register duplicate strategy name."""
        registry = StrategyRegistry()
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MovingAverageCrossover)
    
    def test_unregister(self):
        """unregister removes a strategy."""
        registry = StrategyRegistry()
        
        assert registry.unregister("ma_crossover") is True
        assert "ma_crossover" not in registry
        assert registry.unregister("ma_crossover") is False


# ===================================
# Strategy Tests
# ===================================


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame with upward trend."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    
    # Simulate trending price data
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    close = base_price + trend + noise
    
    df = pd.DataFrame({
        "date": dates,
        "open": close - np.random.rand(100),
        "high": close + np.abs(np.random.randn(100)),
        "low": close - np.abs(np.random.randn(100)),
        "close": close,
        "volume": np.random.randint(1000000, 10000000, 100),
    })
    
    return df


@pytest.fixture
def sample_df_with_sentiment(sample_ohlcv_df):
    """Add sentiment column to sample DataFrame."""
    df = sample_ohlcv_df.copy()
    np.random.seed(42)
    df["sentiment_score"] = np.random.uniform(-1, 1, len(df))
    return df


class TestMovingAverageCrossover:
    """Tests for MA Crossover strategy."""
    
    def test_default_parameters(self):
        """Default parameters are valid."""
        strategy = MovingAverageCrossover()
        assert strategy.params.fast_period == 10
        assert strategy.params.slow_period == 50
        assert strategy.params.ma_type == "sma"
    
    def test_custom_parameters(self):
        """Can create with custom parameters."""
        strategy = MovingAverageCrossover(
            fast_period=5,
            slow_period=20,
            ma_type="ema"
        )
        assert strategy.params.fast_period == 5
        assert strategy.params.slow_period == 20
        assert strategy.params.ma_type == "ema"
    
    def test_fast_must_be_less_than_slow(self):
        """fast_period must be less than slow_period."""
        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_period=50, slow_period=20)
    
    def test_generate_signals(self, sample_ohlcv_df):
        """generate_signals returns correct Series."""
        strategy = MovingAverageCrossover(fast_period=5, slow_period=20)
        signals = strategy.generate_signals(sample_ohlcv_df)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_df)
        assert set(signals.unique()).issubset({-1, 0, 1})
    
    def test_validate_data_missing_column(self):
        """validate_data detects missing columns."""
        strategy = MovingAverageCrossover()
        df = pd.DataFrame({"other": [1, 2, 3]})
        
        missing = strategy.validate_data(df)
        assert "close" in missing
    
    def test_get_metadata(self):
        """get_metadata returns correct info."""
        strategy = MovingAverageCrossover()
        meta = strategy.get_metadata()
        
        assert meta.name == "ma_crossover"
        assert "close" in meta.required_columns


class TestRSIStrategy:
    """Tests for RSI strategy."""
    
    def test_default_parameters(self):
        """Default parameters are valid."""
        strategy = RSIStrategy()
        assert strategy.params.rsi_period == 14
        assert strategy.params.overbought == 70
        assert strategy.params.oversold == 30
    
    def test_generate_signals(self, sample_ohlcv_df):
        """generate_signals returns correct Series."""
        strategy = RSIStrategy()
        signals = strategy.generate_signals(sample_ohlcv_df)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_df)


class TestBollingerBreakout:
    """Tests for Bollinger Breakout strategy."""
    
    def test_default_parameters(self):
        """Default parameters are valid."""
        strategy = BollingerBreakout()
        assert strategy.params.bb_period == 20
        assert strategy.params.bb_std == 2.0
    
    def test_generate_signals(self, sample_ohlcv_df):
        """generate_signals returns correct Series."""
        strategy = BollingerBreakout()
        signals = strategy.generate_signals(sample_ohlcv_df)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_df)


class TestSentimentDriven:
    """Tests for Sentiment Driven strategy."""
    
    def test_default_parameters(self):
        """Default parameters are valid."""
        strategy = SentimentDriven()
        assert strategy.params.positive_threshold == 0.3
    
    def test_generate_signals_with_sentiment(self, sample_df_with_sentiment):
        """generate_signals works with sentiment data."""
        strategy = SentimentDriven()
        signals = strategy.generate_signals(sample_df_with_sentiment)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_df_with_sentiment)
    
    def test_generate_signals_missing_sentiment(self, sample_ohlcv_df):
        """Raises error without sentiment column."""
        strategy = SentimentDriven()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.generate_signals(sample_ohlcv_df)


# ===================================
# API Tests
# ===================================


class TestStrategiesAPI:
    """Tests for strategies API endpoints."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_list_strategies(self, client):
        """GET /api/strategies returns all strategies."""
        response = client.get("/api/strategies")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) >= 4
        
        names = [s["name"] for s in data]
        assert "ma_crossover" in names
        assert "rsi_strategy" in names
    
    def test_get_strategy_detail(self, client):
        """GET /api/strategies/{name} returns strategy details."""
        response = client.get("/api/strategies/ma_crossover")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "ma_crossover"
        assert "description" in data
        assert "version" in data
        assert "parameters_schema" in data
        assert "required_columns" in data
    
    def test_get_strategy_not_found(self, client):
        """GET /api/strategies/{name} returns 404 for unknown strategy."""
        response = client.get("/api/strategies/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_signals_no_data(self, client):
        """POST /api/strategies/{name}/signals returns 404 without data."""
        response = client.post(
            "/api/strategies/ma_crossover/signals",
            json={"ticker": "UNKNOWN_TICKER_XYZ"}
        )
        
        # Should return 404 because no price data
        assert response.status_code == 404
    
    def test_signals_invalid_strategy(self, client):
        """POST /api/strategies/{name}/signals returns 404 for unknown strategy."""
        response = client.post(
            "/api/strategies/nonexistent/signals",
            json={"ticker": "AAPL"}
        )
        
        assert response.status_code == 404
    
    def test_backtest_invalid_strategy(self, client):
        """POST /api/strategies/{name}/backtest returns 404 for unknown strategy."""
        response = client.post(
            "/api/strategies/nonexistent/backtest",
            json={"ticker": "AAPL"}
        )
        
        assert response.status_code == 404
