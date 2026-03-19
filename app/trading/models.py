"""
Trading Models

Pydantic models for orders, positions, portfolios, and trades.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side: buy or sell."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type: market or limit."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderCreate(BaseModel):
    """Request model for creating a new order."""
    symbol: str = Field(..., description="Ticker symbol", min_length=1, max_length=10)
    side: OrderSide = Field(..., description="Buy or sell")
    order_type: OrderType = Field(..., description="Market or limit")
    quantity: float = Field(..., gt=0, description="Number of shares")
    limit_price: Optional[float] = Field(
        None, gt=0, description="Limit price (required for limit orders)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "market",
                "quantity": 10,
                "limit_price": None,
            }
        }


class Order(BaseModel):
    """Order model with full lifecycle information."""
    id: UUID = Field(default_factory=uuid4, description="Unique order ID")
    symbol: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="Buy or sell")
    order_type: OrderType = Field(..., description="Market or limit")
    quantity: float = Field(..., gt=0, description="Original quantity")
    filled_quantity: float = Field(default=0.0, ge=0, description="Filled quantity")
    limit_price: Optional[float] = Field(None, description="Limit price")
    fill_price: Optional[float] = Field(None, description="Average fill price")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = Field(None, description="When order was filled")
    reject_reason: Optional[str] = Field(None, description="Reason for rejection")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "market",
                "quantity": 10,
                "filled_quantity": 10,
                "limit_price": None,
                "fill_price": 175.50,
                "status": "filled",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:01Z",
                "filled_at": "2024-01-15T10:30:01Z",
            }
        }


class Trade(BaseModel):
    """Record of an executed trade (fill)."""
    id: UUID = Field(default_factory=uuid4, description="Unique trade ID")
    order_id: UUID = Field(..., description="Associated order ID")
    symbol: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="Buy or sell")
    quantity: float = Field(..., gt=0, description="Executed quantity")
    price: float = Field(..., gt=0, description="Execution price")
    total_value: float = Field(..., description="Total trade value")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "456e7890-e89b-12d3-a456-426614174001",
                "order_id": "123e4567-e89b-12d3-a456-426614174000",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "price": 175.50,
                "total_value": 1755.00,
                "timestamp": "2024-01-15T10:30:01Z",
            }
        }


class Position(BaseModel):
    """Current position in a single security."""
    symbol: str = Field(..., description="Ticker symbol")
    quantity: float = Field(..., description="Number of shares (negative for short)")
    avg_cost: float = Field(..., ge=0, description="Average cost basis per share")
    current_price: float = Field(..., ge=0, description="Current market price")
    market_value: float = Field(..., description="Current market value")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    unrealized_pnl_pct: float = Field(..., description="Unrealized P&L percentage")
    
    @classmethod
    def calculate(
        cls, symbol: str, quantity: float, avg_cost: float, current_price: float
    ) -> "Position":
        """Calculate position metrics from basic inputs."""
        market_value = quantity * current_price
        cost_basis = quantity * avg_cost
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0.0
        
        return cls(
            symbol=symbol,
            quantity=quantity,
            avg_cost=avg_cost,
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )


class Portfolio(BaseModel):
    """Full portfolio state including cash and positions."""
    cash: float = Field(..., ge=0, description="Available cash balance")
    positions: List[Position] = Field(default_factory=list, description="Current positions")
    total_market_value: float = Field(..., description="Total value of all positions")
    total_equity: float = Field(..., description="Cash + positions value")
    total_unrealized_pnl: float = Field(..., description="Total unrealized P&L")
    total_unrealized_pnl_pct: float = Field(..., description="Total unrealized P&L percentage")
    buying_power: float = Field(..., description="Available buying power")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cash": 95000.00,
                "positions": [
                    {
                        "symbol": "AAPL",
                        "quantity": 28,
                        "avg_cost": 175.50,
                        "current_price": 180.25,
                        "market_value": 5047.00,
                        "unrealized_pnl": 133.00,
                        "unrealized_pnl_pct": 2.71,
                    }
                ],
                "total_market_value": 5047.00,
                "total_equity": 100047.00,
                "total_unrealized_pnl": 133.00,
                "total_unrealized_pnl_pct": 2.71,
                "buying_power": 95000.00,
            }
        }


class PortfolioSnapshot(BaseModel):
    """Point-in-time snapshot of portfolio value for history tracking."""
    timestamp: datetime = Field(..., description="Snapshot timestamp")
    cash: float = Field(..., description="Cash balance")
    market_value: float = Field(..., description="Total positions value")
    total_equity: float = Field(..., description="Total portfolio value")
    realized_pnl: float = Field(..., description="Cumulative realized P&L")
    unrealized_pnl: float = Field(..., description="Unrealized P&L at snapshot")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T16:00:00Z",
                "cash": 95000.00,
                "market_value": 5047.00,
                "total_equity": 100047.00,
                "realized_pnl": 0.00,
                "unrealized_pnl": 133.00,
            }
        }


class PriceUpdate(BaseModel):
    """Real-time price update for WebSocket streaming."""
    symbol: str = Field(..., description="Ticker symbol")
    price: float = Field(..., gt=0, description="Current price")
    change: float = Field(..., description="Price change from previous")
    change_pct: float = Field(..., description="Percentage change")
    volume: int = Field(default=0, ge=0, description="Volume")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "price": 180.25,
                "change": 1.50,
                "change_pct": 0.84,
                "volume": 1250000,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class OrderResponse(BaseModel):
    """Response wrapper for order operations."""
    success: bool = Field(..., description="Whether operation succeeded")
    order: Optional[Order] = Field(None, description="The order object")
    message: str = Field(..., description="Status message")
    trade: Optional[Trade] = Field(None, description="Trade record if filled")


class PortfolioHistoryResponse(BaseModel):
    """Response for portfolio history endpoint."""
    snapshots: List[PortfolioSnapshot] = Field(..., description="Historical snapshots")
    period_start: datetime = Field(..., description="History period start")
    period_end: datetime = Field(..., description="History period end")
    total_return: float = Field(..., description="Total return over period")
    total_return_pct: float = Field(..., description="Total return percentage")
