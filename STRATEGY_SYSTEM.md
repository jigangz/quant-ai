# Strategy System Documentation

## Overview

The Strategy System provides a flexible framework for creating, managing, and backtesting trading strategies in the Quant AI platform.

## Architecture

### Core Components

1. **BaseStrategy** (`app/strategies/base.py`)
   - Abstract base class for all trading strategies
   - Defines the interface: `generate_signals(df) -> pd.Series`
   - Parameter validation via Pydantic models
   - Data validation via `validate_data(df)`

2. **StrategyRegistry** (`app/strategies/__init__.py`)
   - Global registry for managing strategies
   - Pre-loaded with 4 built-in strategy templates
   - Methods: `get()`, `list_strategies()`, `create_instance()`

3. **Strategy Templates** (`app/strategies/templates/`)
   - 4 ready-to-use strategies
   - All use native pandas/numpy (no external TA library dependency)

4. **REST API** (`app/api/strategies.py`)
   - Full CRUD endpoints for strategies
   - Signal generation and backtesting

## Built-in Strategies

### 1. Moving Average Crossover (`ma_crossover`)

**Description:** Classic trend-following strategy using fast/slow MA crossover

**Parameters:**
- `fast_period` (int, default=10): Fast MA period
- `slow_period` (int, default=50): Slow MA period  
- `ma_type` ("sma" | "ema", default="sma"): MA type
- `signal_on_cross_only` (bool, default=False): Signal only on crossover bars

**Signals:**
- Long (1): Fast MA crosses above Slow MA
- Short (-1): Fast MA crosses below Slow MA
- Hold (0): No crossover (if cross_only=True) or between trends

**Required Columns:** `close`

---

### 2. RSI Strategy (`rsi_strategy`)

**Description:** Mean-reversion strategy using RSI overbought/oversold levels

**Parameters:**
- `rsi_period` (int, default=14): RSI calculation period
- `overbought` (float, default=70): Overbought threshold
- `oversold` (float, default=30): Oversold threshold
- `exit_on_neutral` (bool, default=False): Exit when RSI returns to neutral zone

**Signals:**
- Long (1): RSI < oversold (expect bounce)
- Short (-1): RSI > overbought (expect pullback)
- Hold (0): RSI in neutral zone

**Required Columns:** `close`

---

### 3. Bollinger Bands Breakout (`bollinger_breakout`)

**Description:** Volatility-based breakout or mean-reversion strategy

**Parameters:**
- `bb_period` (int, default=20): Bollinger Bands period
- `bb_std` (float, default=2.0): Standard deviation multiplier
- `strategy_mode` ("breakout" | "mean_reversion", default="breakout"): Strategy mode
- `use_close_cross` (bool, default=True): Use close vs high/low for signals

**Signals (Breakout Mode):**
- Long (1): Price breaks above upper band
- Short (-1): Price breaks below lower band
- Hold (0): Price within bands

**Signals (Mean-Reversion Mode):**
- Short (-1): Price breaks above upper band (fade the extreme)
- Long (1): Price breaks below lower band (fade the extreme)
- Hold (0): Price within bands

**Required Columns:** `close` (+ `high`, `low` if use_close_cross=False)

---

### 4. Sentiment Driven (`sentiment_driven`)

**Description:** News/social sentiment-based strategy

**Parameters:**
- `positive_threshold` (float, default=0.3): Bullish sentiment threshold
- `negative_threshold` (float, default=-0.3): Bearish sentiment threshold
- `sentiment_column` (str, default="sentiment_score"): Column name for sentiment
- `smoothing_period` (int, default=1): Rolling window for smoothing (1=no smoothing)
- `require_confirmation` (bool, default=False): Require consecutive extreme readings
- `confirmation_periods` (int, default=2): Number of periods for confirmation

**Signals:**
- Long (1): Sentiment > positive_threshold
- Short (-1): Sentiment < negative_threshold
- Hold (0): Sentiment in neutral zone

**Required Columns:** `sentiment_score` (or configured column name)

**Note:** Sentiment scores should be in range [-1, 1] where -1=very negative, 0=neutral, +1=very positive

## API Endpoints

### GET `/api/strategies`

List all available strategies with metadata.

**Response:**
```json
{
  "strategies": [
    {
      "name": "ma_crossover",
      "description": "...",
      "version": "1.0.0",
      "parameters_schema": {...},
      "required_columns": ["close"]
    }
  ],
  "total": 4
}
```

---

### GET `/api/strategies/{name}`

Get details for a specific strategy.

**Response:**
```json
{
  "name": "ma_crossover",
  "description": "...",
  "version": "1.0.0",
  "parameters_schema": {
    "properties": {
      "fast_period": {"type": "integer", "default": 10, ...},
      "slow_period": {"type": "integer", "default": 50, ...}
    }
  },
  "required_columns": ["close"]
}
```

---

### POST `/api/strategies/{name}/signals`

Generate trading signals using the strategy.

**Request:**
```json
{
  "ticker": "AAPL",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "parameters": {
    "fast_period": 5,
    "slow_period": 20
  },
  "add_features": true
}
```

**Response:**
```json
{
  "success": true,
  "strategy": "ma_crossover",
  "ticker": "AAPL",
  "parameters": {...},
  "signals": [0, 0, 1, 1, 1, -1, ...],
  "dates": ["2023-01-01", "2023-01-02", ...],
  "summary": {
    "long": 45,
    "short": 38,
    "neutral": 170
  }
}
```

---

### POST `/api/strategies/{name}/backtest`

Run a backtest using the strategy.

**Request:**
```json
{
  "ticker": "AAPL",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "parameters": {
    "fast_period": 5,
    "slow_period": 20
  },
  "initial_capital": 10000,
  "position_size": 1.0,
  "transaction_cost_bps": 10
}
```

**Response:**
```json
{
  "success": true,
  "strategy": "ma_crossover",
  "ticker": "AAPL",
  "parameters": {...},
  "total_return": 15.23,
  "sharpe_ratio": 1.45,
  "max_drawdown": 8.50,
  "win_rate": 55.2,
  "n_trades": 42,
  "n_long": 21,
  "n_short": 21,
  "equity_curve": [...],
  "benchmark_curve": [...]
}
```

## Usage Examples

### Python API

```python
from app.strategies import get_registry
import pandas as pd

# Get the registry
registry = get_registry()

# List all strategies
strategies = registry.list_strategies()
# ['ma_crossover', 'rsi_strategy', 'bollinger_breakout', 'sentiment_driven']

# Get a strategy
StrategyClass = registry.get('ma_crossover')

# Create instance with custom parameters
strategy = StrategyClass(
    fast_period=5,
    slow_period=20,
    ma_type='ema'
)

# Or use registry.create_instance
strategy = registry.create_instance('ma_crossover', {
    'fast_period': 5,
    'slow_period': 20
})

# Validate data
df = pd.DataFrame({'close': [100, 101, 102, 103]})
missing = strategy.validate_data(df)
if missing:
    print(f"Missing columns: {missing}")

# Generate signals
signals = strategy.generate_signals(df)
# Returns pd.Series with values: -1 (short), 0 (hold), 1 (long)
```

### Creating Custom Strategies

```python
from app.strategies.base import BaseStrategy, BaseParameters
from pydantic import Field
import pandas as pd

class MyStrategyParameters(BaseParameters):
    threshold: float = Field(default=0.5, ge=0, le=1)

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    description = "My custom strategy"
    version = "1.0.0"
    required_columns = ["close", "volume"]
    Parameters = MyStrategyParameters
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        
        # Your logic here
        volume_ma = df["volume"].rolling(20).mean()
        signals[df["volume"] > volume_ma * self.params.threshold] = 1
        
        return signals

# Register your strategy
from app.strategies import get_registry
registry = get_registry()
registry.register(MyStrategy)
```

## Testing

Run tests with pytest:

```bash
pytest tests/test_strategies.py -v
```

Test coverage:
- ✅ BaseStrategy validation and metadata
- ✅ All 4 strategy templates
- ✅ StrategyRegistry operations
- ✅ Signal generation for all strategies
- ✅ Parameter validation
- ✅ Edge cases (empty data, missing columns, invalid params)

## Technical Details

### Signal Format

All strategies return a `pd.Series` with integer values:
- `1`: Long signal (buy/bullish)
- `0`: Neutral/hold (no position or between signals)
- `-1`: Short signal (sell/bearish)

The series index should match the input DataFrame index (typically datetime).

### Data Requirements

- All strategies require at minimum a `close` price column
- Some strategies require additional columns (high, low, sentiment_score, etc.)
- Use `strategy.validate_data(df)` to check before generating signals
- Set `add_features=True` in API requests to auto-add technical indicators

### Python Compatibility

- Python 3.9+ compatible
- Uses `Optional[]` not `X | None`
- Uses `List[]` not `list[]`
- No external TA library dependency (uses pandas/numpy)

## Integration with Existing Systems

The Strategy System integrates with:

1. **Backtest Engine** (`app/backtest/engine.py`)
   - Generate signals from strategies
   - Feed signals to existing backtest simulation

2. **ML Features** (`app/ml/features/technical.py`)
   - Strategies can use existing technical indicators
   - Or calculate their own

3. **Price Data** (`app/db/prices_repo.py`)
   - Fetch historical data via `get_prices()`
   - Auto-add features with `add_technical_features()`

## Future Enhancements

Potential improvements:
- [ ] Strategy optimizer (grid search for best parameters)
- [ ] Multi-strategy portfolio backtesting
- [ ] Strategy performance dashboard
- [ ] Save/load strategy configurations
- [ ] Strategy marketplace (import/export)
- [ ] ML-based strategy meta-learning
- [ ] Live trading signal generation
