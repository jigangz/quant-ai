"""
Tests for Paper Trading System

Tests matching engine, portfolio management, and API endpoints.
"""

import pytest
from datetime import datetime
from uuid import UUID

from app.trading.models import (
    OrderCreate,
    OrderSide,
    OrderType,
    OrderStatus,
)
from app.trading.engine import MatchingEngine, MockPriceProvider
from app.trading.portfolio import PortfolioManager
from app.trading.websocket import PriceGenerator


# ===================================
# Fixtures
# ===================================

@pytest.fixture
def price_provider():
    """Create a mock price provider."""
    return MockPriceProvider({"AAPL": 150.0, "GOOGL": 100.0})


@pytest.fixture
def portfolio():
    """Create a fresh portfolio manager."""
    return PortfolioManager(starting_cash=100000.0)


@pytest.fixture
def engine(portfolio, price_provider):
    """Create a matching engine."""
    return MatchingEngine(portfolio=portfolio, price_provider=price_provider)


# ===================================
# Portfolio Tests
# ===================================

def test_portfolio_initial_state(portfolio):
    """Test initial portfolio state."""
    state = portfolio.get_portfolio({})
    
    assert state.cash == 100000.0
    assert state.total_equity == 100000.0
    assert state.total_market_value == 0.0
    assert len(state.positions) == 0


def test_portfolio_reset(portfolio):
    """Test portfolio reset."""
    # Make some changes
    portfolio._cash = 50000.0
    
    # Reset
    state = portfolio.reset(starting_cash=150000.0)
    
    assert state.cash == 150000.0
    assert state.total_equity == 150000.0


def test_portfolio_buying_power(portfolio):
    """Test buying power checks."""
    assert portfolio.check_buying_power(50000.0) is True
    assert portfolio.check_buying_power(150000.0) is False


# ===================================
# Matching Engine Tests
# ===================================

def test_market_order_buy(engine):
    """Test market buy order execution."""
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
    )
    
    order, trade = engine.submit_order(order_request)
    
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 10
    assert order.fill_price == 150.0
    assert trade is not None
    assert trade.symbol == "AAPL"
    assert trade.quantity == 10
    assert trade.price == 150.0


def test_market_order_sell(engine, portfolio):
    """Test market sell order execution."""
    # First buy some shares
    buy_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
    )
    engine.submit_order(buy_request)
    
    # Now sell
    sell_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=5,
    )
    
    order, trade = engine.submit_order(sell_request)
    
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 5
    assert trade.quantity == 5


def test_insufficient_funds(engine):
    """Test order rejection due to insufficient funds."""
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000,  # Would cost $150,000
    )
    
    order, trade = engine.submit_order(order_request)
    
    assert order.status == OrderStatus.REJECTED
    assert trade is None
    assert "Insufficient funds" in order.reject_reason


def test_insufficient_position(engine):
    """Test sell rejection due to insufficient position."""
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=10,
    )
    
    order, trade = engine.submit_order(order_request)
    
    assert order.status == OrderStatus.REJECTED
    assert trade is None
    assert "Insufficient position" in order.reject_reason


def test_limit_order_immediate_fill(engine):
    """Test limit order that fills immediately."""
    # Buy limit at $160 when price is $150 - should fill
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        limit_price=160.0,
    )
    
    order, trade = engine.submit_order(order_request)
    
    assert order.status == OrderStatus.FILLED
    assert order.fill_price == 150.0  # Fills at market, not limit


def test_limit_order_pending(engine):
    """Test limit order that remains pending."""
    # Buy limit at $140 when price is $150 - should pend
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        limit_price=140.0,
    )
    
    order, trade = engine.submit_order(order_request)
    
    assert order.status == OrderStatus.PENDING
    assert trade is None
    assert order.limit_price == 140.0


def test_limit_order_fill_on_price_change(engine, price_provider):
    """Test limit order filling when price changes."""
    # Place buy limit at $140
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        limit_price=140.0,
    )
    
    order, trade = engine.submit_order(order_request)
    assert order.status == OrderStatus.PENDING
    
    # Update price to $135
    price_provider.set_price("AAPL", 135.0)
    
    # Check limit orders
    filled = engine.check_limit_orders()
    
    assert len(filled) == 1
    filled_order, filled_trade = filled[0]
    assert filled_order.id == order.id
    assert filled_order.status == OrderStatus.FILLED
    assert filled_order.fill_price == 135.0


def test_cancel_order(engine):
    """Test order cancellation."""
    # Place pending limit order
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10,
        limit_price=140.0,
    )
    
    order, _ = engine.submit_order(order_request)
    assert order.status == OrderStatus.PENDING
    
    # Cancel it
    cancelled = engine.cancel_order(order.id)
    
    assert cancelled is not None
    assert cancelled.status == OrderStatus.CANCELLED
    assert cancelled.id == order.id


def test_cannot_cancel_filled_order(engine):
    """Test that filled orders cannot be cancelled."""
    # Place and fill market order
    order_request = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
    )
    
    order, _ = engine.submit_order(order_request)
    assert order.status == OrderStatus.FILLED
    
    # Try to cancel
    cancelled = engine.cancel_order(order.id)
    
    assert cancelled is None


def test_get_orders_filtering(engine):
    """Test order list filtering."""
    # Create mix of orders
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=5
    ))
    
    engine.submit_order(OrderCreate(
        symbol="GOOGL", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=10, limit_price=90.0
    ))
    
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=5, limit_price=140.0
    ))
    
    # Filter by status
    filled = engine.get_orders(status=OrderStatus.FILLED)
    assert len(filled) == 1
    assert filled[0].symbol == "AAPL"
    
    pending = engine.get_orders(status=OrderStatus.PENDING)
    assert len(pending) == 2
    
    # Filter by symbol
    aapl_orders = engine.get_orders(symbol="AAPL")
    assert len(aapl_orders) == 2
    assert all(o.symbol == "AAPL" for o in aapl_orders)


# ===================================
# Portfolio Integration Tests
# ===================================

def test_position_tracking(engine, portfolio):
    """Test that positions are tracked correctly."""
    # Buy 10 AAPL @ $150
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    state = portfolio.get_portfolio({"AAPL": 150.0})
    
    assert len(state.positions) == 1
    pos = state.positions[0]
    assert pos.symbol == "AAPL"
    assert pos.quantity == 10
    assert pos.avg_cost == 150.0
    assert pos.current_price == 150.0
    assert pos.unrealized_pnl == 0.0


def test_position_averaging(engine, portfolio):
    """Test position cost averaging."""
    # Buy 10 @ $150
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    # Buy 10 more @ $160
    engine.price_provider.set_price("AAPL", 160.0)
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    state = portfolio.get_portfolio({"AAPL": 160.0})
    pos = state.positions[0]
    
    assert pos.quantity == 20
    assert pos.avg_cost == 155.0  # (10*150 + 10*160) / 20


def test_realized_pnl(engine, portfolio):
    """Test realized P&L calculation."""
    # Buy 10 @ $150
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    # Sell 5 @ $160
    engine.price_provider.set_price("AAPL", 160.0)
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=5
    ))
    
    realized_pnl = portfolio.get_total_realized_pnl()
    
    # Sold 5 @ $160 that cost $150 = $50 profit
    assert realized_pnl == 50.0


def test_unrealized_pnl(engine, portfolio):
    """Test unrealized P&L calculation."""
    # Buy 10 @ $150
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    # Price increases to $160
    state = portfolio.get_portfolio({"AAPL": 160.0})
    
    assert state.total_unrealized_pnl == 100.0  # 10 * ($160 - $150)
    assert state.total_unrealized_pnl_pct == pytest.approx(6.67, rel=0.01)


def test_cash_tracking(engine, portfolio):
    """Test cash balance tracking."""
    initial_cash = portfolio.cash
    
    # Buy 10 AAPL @ $150 = $1500
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    assert portfolio.cash == initial_cash - 1500.0
    
    # Sell 5 @ $160 = $800
    engine.price_provider.set_price("AAPL", 160.0)
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=5
    ))
    
    assert portfolio.cash == initial_cash - 1500.0 + 800.0


def test_portfolio_history(engine, portfolio):
    """Test portfolio history tracking."""
    initial_history = portfolio.get_history()
    assert len(initial_history) == 1  # Initial snapshot
    
    # Make a trade
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    # Record snapshot
    snapshot = portfolio.record_snapshot({"AAPL": 155.0})
    
    history = portfolio.get_history()
    assert len(history) == 2
    assert history[-1].timestamp == snapshot.timestamp


# ===================================
# Price Generator Tests
# ===================================

def test_price_generator():
    """Test mock price generation."""
    gen = PriceGenerator()
    
    # Get initial price
    initial = gen.get_price("AAPL")
    assert initial > 0
    
    # Generate tick
    updates = gen.tick(["AAPL"])
    
    assert "AAPL" in updates
    update = updates["AAPL"]
    assert update.price > 0
    assert update.symbol == "AAPL"


def test_price_generator_multiple_symbols():
    """Test price generation for multiple symbols."""
    gen = PriceGenerator()
    
    symbols = ["AAPL", "GOOGL", "MSFT"]
    updates = gen.tick(symbols)
    
    assert len(updates) == 3
    for symbol in symbols:
        assert symbol in updates
        assert updates[symbol].price > 0


# ===================================
# Edge Cases
# ===================================

def test_zero_quantity_rejected():
    """Test that zero quantity orders are rejected."""
    with pytest.raises(Exception):  # Pydantic validation error
        OrderCreate(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
        )


def test_negative_quantity_rejected():
    """Test that negative quantity orders are rejected."""
    with pytest.raises(Exception):  # Pydantic validation error
        OrderCreate(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=-10,
        )


def test_limit_order_without_price_rejected(engine):
    """Test that limit orders without price are rejected."""
    with pytest.raises(ValueError, match="Limit price required"):
        engine.submit_order(OrderCreate(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=None,
        ))


def test_multiple_positions(engine, portfolio):
    """Test managing multiple positions."""
    # Buy different symbols
    engine.submit_order(OrderCreate(
        symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=10
    ))
    
    engine.submit_order(OrderCreate(
        symbol="GOOGL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=20
    ))
    
    state = portfolio.get_portfolio({"AAPL": 150.0, "GOOGL": 100.0})
    
    assert len(state.positions) == 2
    symbols = {p.symbol for p in state.positions}
    assert symbols == {"AAPL", "GOOGL"}
