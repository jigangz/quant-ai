# Paper Trading System

Complete paper trading simulation for the Quant AI platform.

## Architecture

### Components

1. **Models** (`app/trading/models.py`)
   - Pydantic models for type safety and validation
   - Order lifecycle: `OrderCreate` → `Order` → `Trade`
   - Position tracking with `Position` model
   - Portfolio state with `Portfolio` model
   - Historical tracking with `PortfolioSnapshot`

2. **Matching Engine** (`app/trading/engine.py`)
   - Local order matching with immediate execution
   - Market orders: fill at current price
   - Limit orders: fill when price reaches limit
   - Order validation against portfolio state
   - Thread-safe operations

3. **Portfolio Manager** (`app/trading/portfolio.py`)
   - Cash balance tracking
   - Position management with cost averaging
   - Real-time P&L calculation (realized + unrealized)
   - Portfolio history snapshots
   - Thread-safe with locking

4. **WebSocket Streaming** (`app/trading/websocket.py`)
   - Real-time price updates
   - Mock price generator with random walk
   - Connection management
   - Symbol-based subscriptions

5. **REST API** (`app/api/trading.py`)
   - Order placement and management
   - Portfolio queries
   - Trade history
   - WebSocket endpoint

## API Reference

### Order Management

#### Place Order
```http
POST /api/trading/orders
Content-Type: application/json

{
  "symbol": "AAPL",
  "side": "buy",
  "order_type": "market",
  "quantity": 10,
  "limit_price": null
}
```

Response:
```json
{
  "success": true,
  "order": {
    "id": "uuid",
    "symbol": "AAPL",
    "side": "buy",
    "order_type": "market",
    "quantity": 10,
    "filled_quantity": 10,
    "status": "filled",
    "fill_price": 175.50,
    "created_at": "2024-01-15T10:30:00Z",
    "filled_at": "2024-01-15T10:30:01Z"
  },
  "message": "Order filled at $175.50",
  "trade": {
    "id": "uuid",
    "order_id": "uuid",
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 10,
    "price": 175.50,
    "total_value": 1755.00,
    "timestamp": "2024-01-15T10:30:01Z"
  }
}
```

#### List Orders
```http
GET /api/trading/orders?status=filled&symbol=AAPL&limit=100
```

#### Get Order
```http
GET /api/trading/orders/{order_id}
```

#### Cancel Order
```http
DELETE /api/trading/orders/{order_id}
```

### Portfolio Management

#### Get Portfolio
```http
GET /api/trading/portfolio
```

Response:
```json
{
  "cash": 95000.00,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 10,
      "avg_cost": 175.50,
      "current_price": 180.25,
      "market_value": 1802.50,
      "unrealized_pnl": 47.50,
      "unrealized_pnl_pct": 2.71
    }
  ],
  "total_market_value": 1802.50,
  "total_equity": 96802.50,
  "total_unrealized_pnl": 47.50,
  "total_unrealized_pnl_pct": 2.71,
  "buying_power": 95000.00
}
```

#### Get Portfolio History
```http
GET /api/trading/portfolio/history?limit=100
```

#### Reset Portfolio
```http
POST /api/trading/portfolio/reset?starting_cash=100000
```

#### Get Trades
```http
GET /api/trading/trades?limit=100
```

### WebSocket Price Streaming

Connect to `ws://localhost:8000/api/trading/ws/prices`

**Subscribe to symbols:**
```json
{
  "action": "subscribe",
  "symbols": ["AAPL", "GOOGL", "MSFT"]
}
```

**Receive price updates:**
```json
{
  "type": "price_update",
  "data": {
    "AAPL": {
      "symbol": "AAPL",
      "price": 180.25,
      "change": 1.50,
      "change_pct": 0.84,
      "volume": 1250000,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }
}
```

**Unsubscribe:**
```json
{
  "action": "unsubscribe",
  "symbols": ["AAPL"]
}
```

## Order Types

### Market Orders
- Execute immediately at current market price
- No limit price required
- Guaranteed fill (if funds/position available)

```python
OrderCreate(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
)
```

### Limit Orders
- Execute only when price reaches or exceeds limit
- Buy limit: fills when market price ≤ limit price
- Sell limit: fills when market price ≥ limit price
- May remain pending if price never reached

```python
OrderCreate(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=10,
    limit_price=175.00
)
```

## Order Lifecycle

1. **PENDING** - Order submitted, waiting to fill
   - Limit orders awaiting price
   - Market orders awaiting validation

2. **FILLED** - Order executed completely
   - `filled_quantity == quantity`
   - `fill_price` set to execution price
   - Trade record created

3. **PARTIALLY_FILLED** - Order partially executed
   - Not implemented in MVP (orders fill completely or not at all)

4. **CANCELLED** - Order cancelled by user
   - Only pending orders can be cancelled
   - Cancellation is immediate

5. **REJECTED** - Order rejected by system
   - Insufficient funds for buy
   - Insufficient position for sell
   - Invalid parameters

## P&L Calculation

### Unrealized P&L
Calculated based on current market prices:
```
unrealized_pnl = (current_price - avg_cost) * quantity
unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100
```

### Realized P&L
Calculated when positions are closed:
```
# Long position
realized_pnl = (sell_price - avg_cost) * quantity_sold

# Short position  
realized_pnl = (avg_cost - buy_price) * quantity_covered
```

### Cost Averaging
When adding to existing position:
```
new_avg_cost = (old_qty * old_avg + new_qty * new_price) / (old_qty + new_qty)
```

## Position Management

### Opening Position
```python
# Buy 10 AAPL @ $175
buy_order = OrderCreate(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
)
```

### Adding to Position
```python
# Buy 5 more AAPL @ $180
# Average cost becomes: (10*175 + 5*180) / 15 = $176.67
```

### Closing Position
```python
# Sell all 15 AAPL @ $185
sell_order = OrderCreate(
    symbol="AAPL",
    side=OrderSide.SELL,
    order_type=OrderType.MARKET,
    quantity=15
)
# Realized P&L: (185 - 176.67) * 15 = $125
```

### Partial Close
```python
# Sell 10 AAPL @ $185
# Realized P&L: (185 - 176.67) * 10 = $83.33
# Remaining: 5 AAPL @ $176.67 avg cost
```

## Mock Price Provider

For demo/testing, prices are simulated using:

1. **Base Prices** - Common symbols have predefined starting prices
2. **Random Walk** - Prices move randomly with ~0.1-0.3% volatility per tick
3. **Mean Reversion** - Tendency to revert toward base price
4. **Volume Simulation** - Realistic volume numbers

### Updating Prices Manually
```python
from app.trading.engine import get_matching_engine

engine = get_matching_engine()
engine.price_provider.set_price("AAPL", 180.00)

# Bulk update
engine.price_provider.update_prices({
    "AAPL": 180.00,
    "GOOGL": 145.00,
    "MSFT": 380.00
})
```

## Testing

Run tests with pytest:
```bash
pytest tests/test_trading.py -v
```

Test coverage:
- ✓ Portfolio initialization and reset
- ✓ Market order execution (buy/sell)
- ✓ Limit order execution and pending
- ✓ Order rejection (insufficient funds/position)
- ✓ Order cancellation
- ✓ Position tracking and averaging
- ✓ Realized/unrealized P&L calculation
- ✓ Cash balance tracking
- ✓ Portfolio history snapshots
- ✓ Price generation
- ✓ Edge cases and validation

## Integration with Market Data

The paper trading system can be integrated with real market data:

```python
from app.providers.market.yahoo import YahooMarketProvider
from app.trading.engine import PriceProvider

class LivePriceProvider(PriceProvider):
    def __init__(self):
        self.market = YahooMarketProvider()
        self._cache = {}
    
    def get_price(self, symbol: str) -> Optional[float]:
        # Fetch latest price from Yahoo Finance
        df = self.market.fetch(symbol, period="1d", interval="1m")
        if df.empty:
            return self._cache.get(symbol)
        
        price = df.iloc[-1]["close"]
        self._cache[symbol] = price
        return price

# Use in engine
engine = MatchingEngine(price_provider=LivePriceProvider())
```

## Thread Safety

All components are thread-safe:
- `MatchingEngine` uses `threading.Lock`
- `PortfolioManager` uses `threading.Lock`
- `ConnectionManager` uses `Set` operations

This allows:
- Concurrent order submission
- Parallel price updates
- Multiple WebSocket connections
- Background limit order checking

## In-Memory Storage

Current implementation uses in-memory storage:
- Orders: `Dict[UUID, Order]` in `MatchingEngine`
- Positions: `Dict[str, PositionState]` in `PortfolioManager`
- Trades: `List[Trade]` in `PortfolioManager`
- History: `List[PortfolioSnapshot]` in `PortfolioManager`

**Limitations:**
- Data lost on server restart
- No persistence
- Limited scalability

**Future Enhancement:**
- Persist to database (PostgreSQL/Redis)
- Separate order service
- Event sourcing for full audit trail

## Example Usage

### Simple Buy and Sell
```python
from app.trading import OrderCreate, OrderSide, OrderType
from app.trading.engine import get_matching_engine

engine = get_matching_engine()

# Buy 10 AAPL
order, trade = engine.submit_order(OrderCreate(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10
))

print(f"Bought {order.filled_quantity} shares at ${order.fill_price}")

# Sell 5 AAPL
order, trade = engine.submit_order(OrderCreate(
    symbol="AAPL",
    side=OrderSide.SELL,
    order_type=OrderType.MARKET,
    quantity=5
))

print(f"Sold {order.filled_quantity} shares at ${order.fill_price}")
```

### Limit Order with Price Watch
```python
# Place limit order
order, trade = engine.submit_order(OrderCreate(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=10,
    limit_price=170.00
))

if order.status == OrderStatus.PENDING:
    print("Order pending, waiting for price <= $170")
    
    # Later, when price updates...
    engine.price_provider.set_price("AAPL", 169.00)
    filled = engine.check_limit_orders()
    
    if filled:
        print(f"Order filled at ${filled[0][0].fill_price}")
```

### Portfolio Monitoring
```python
from app.trading.portfolio import get_portfolio_manager

portfolio = get_portfolio_manager()

# Get current state
state = portfolio.get_portfolio({"AAPL": 180.00})

print(f"Cash: ${state.cash:,.2f}")
print(f"Total Equity: ${state.total_equity:,.2f}")
print(f"Unrealized P&L: ${state.total_unrealized_pnl:,.2f} ({state.total_unrealized_pnl_pct:.2f}%)")

for pos in state.positions:
    print(f"{pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f} "
          f"(current: ${pos.current_price:.2f}, P&L: ${pos.unrealized_pnl:.2f})")
```

## Next Steps

Potential enhancements:
1. **Database Persistence** - PostgreSQL for order/trade history
2. **Advanced Orders** - Stop loss, trailing stop, brackets
3. **Short Selling** - Full short position support
4. **Margin** - Leverage and margin requirements
5. **Fees** - Commission and slippage simulation
6. **Risk Management** - Position limits, max loss rules
7. **Portfolio Analytics** - Sharpe ratio, drawdown, etc.
8. **Live Data Integration** - Real-time market data feeds
9. **Backtesting Bridge** - Connect to existing backtest engine
10. **Multi-Currency** - Support for forex pairs

## Questions or Issues?

See main README or open an issue on GitHub.
