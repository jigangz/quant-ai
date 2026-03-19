"""
Paper Trading Module

Provides simulated trading functionality with:
- Local matching engine for order execution
- Portfolio management with position tracking
- Real-time price streaming via WebSocket
"""

from app.trading.models import (
    OrderSide,
    OrderType,
    OrderStatus,
    Order,
    OrderCreate,
    Position,
    Portfolio,
    Trade,
    PortfolioSnapshot,
)
from app.trading.engine import MatchingEngine
from app.trading.portfolio import PortfolioManager

__all__ = [
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Order",
    "OrderCreate",
    "Position",
    "Portfolio",
    "Trade",
    "PortfolioSnapshot",
    "MatchingEngine",
    "PortfolioManager",
]
