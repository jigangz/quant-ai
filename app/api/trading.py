"""
Paper Trading API

REST endpoints and WebSocket for paper trading simulation.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

from app.trading import (
    OrderCreate,
    Order,
    OrderStatus,
    Portfolio,
    PortfolioSnapshot,
    Trade,
)
from app.trading.engine import get_matching_engine, reset_matching_engine
from app.trading.portfolio import get_portfolio_manager, reset_portfolio_manager
from app.trading.websocket import get_connection_manager, get_price_streamer
from app.trading.models import OrderResponse, PortfolioHistoryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading")


# ===================================
# Order Endpoints
# ===================================

@router.post("/orders", response_model=OrderResponse, status_code=201)
async def place_order(order_request: OrderCreate):
    """
    Place a new order (market or limit).
    
    Market orders execute immediately at current price.
    Limit orders execute when price reaches limit.
    """
    try:
        engine = get_matching_engine()
        order, trade = engine.submit_order(order_request)
        
        if order.status == OrderStatus.REJECTED:
            return OrderResponse(
                success=False,
                order=order,
                message=order.reject_reason or "Order rejected",
                trade=None,
            )
        
        if order.status == OrderStatus.FILLED:
            return OrderResponse(
                success=True,
                order=order,
                message=f"Order filled at ${order.fill_price:.2f}",
                trade=trade,
            )
        
        # Pending
        return OrderResponse(
            success=True,
            order=order,
            message="Order placed (pending)",
            trade=None,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error placing order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to place order")


@router.get("/orders", response_model=List[Order])
async def list_orders(
    status: Optional[OrderStatus] = Query(None, description="Filter by status"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, ge=1, le=1000, description="Max orders to return"),
):
    """List orders with optional filtering."""
    try:
        engine = get_matching_engine()
        orders = engine.get_orders(status=status, symbol=symbol, limit=limit)
        return orders
    except Exception as e:
        logger.error(f"Error listing orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list orders")


@router.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: UUID):
    """Get a specific order by ID."""
    try:
        engine = get_matching_engine()
        order = engine.get_order(order_id)
        
        if order is None:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get order")


@router.delete("/orders/{order_id}", response_model=Order)
async def cancel_order(order_id: UUID):
    """Cancel a pending order."""
    try:
        engine = get_matching_engine()
        order = engine.cancel_order(order_id)
        
        if order is None:
            raise HTTPException(
                status_code=404,
                detail="Order not found or cannot be cancelled"
            )
        
        return order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel order")


# ===================================
# Portfolio Endpoints
# ===================================

@router.get("/portfolio", response_model=Portfolio)
async def get_portfolio():
    """
    Get current portfolio state.
    
    Returns cash, positions, and calculated metrics.
    """
    try:
        portfolio_mgr = get_portfolio_manager()
        engine = get_matching_engine()
        
        # Get current prices from price provider
        price_provider = engine.price_provider
        
        # Collect all symbols from positions
        current_prices = {}
        
        # Get positions to know which symbols we need prices for
        # We'll do a quick snapshot with empty prices first to get symbols
        temp_portfolio = portfolio_mgr.get_portfolio({})
        for position in temp_portfolio.positions:
            price = price_provider.get_price(position.symbol)
            if price:
                current_prices[position.symbol] = price
        
        # Get full portfolio with current prices
        portfolio = portfolio_mgr.get_portfolio(current_prices)
        return portfolio
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get portfolio")


@router.get("/portfolio/history", response_model=PortfolioHistoryResponse)
async def get_portfolio_history(
    limit: int = Query(100, ge=1, le=1000, description="Max snapshots to return"),
):
    """
    Get portfolio value history.
    
    Returns historical snapshots for tracking P&L over time.
    """
    try:
        portfolio_mgr = get_portfolio_manager()
        snapshots = portfolio_mgr.get_history(limit=limit)
        
        if not snapshots:
            return PortfolioHistoryResponse(
                snapshots=[],
                period_start=None,
                period_end=None,
                total_return=0.0,
                total_return_pct=0.0,
            )
        
        first = snapshots[0]
        last = snapshots[-1]
        
        # Calculate total return
        initial_equity = first.total_equity
        final_equity = last.total_equity
        total_return = final_equity - initial_equity
        total_return_pct = (total_return / initial_equity * 100) if initial_equity > 0 else 0.0
        
        return PortfolioHistoryResponse(
            snapshots=snapshots,
            period_start=first.timestamp,
            period_end=last.timestamp,
            total_return=total_return,
            total_return_pct=total_return_pct,
        )
        
    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get portfolio history")


@router.post("/portfolio/reset")
async def reset_portfolio(
    starting_cash: float = Query(100000.0, gt=0, description="Starting cash balance"),
):
    """
    Reset portfolio to initial state.
    
    Clears all positions and orders, resets cash to starting amount.
    """
    try:
        # Reset portfolio
        portfolio_mgr = reset_portfolio_manager(starting_cash)
        
        # Reset matching engine
        reset_matching_engine()
        
        logger.info(f"Portfolio reset to ${starting_cash:,.2f}")
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Portfolio reset to ${starting_cash:,.2f}",
                "cash": starting_cash,
            }
        )
        
    except Exception as e:
        logger.error(f"Error resetting portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset portfolio")


@router.get("/trades", response_model=List[Trade])
async def get_trades(
    limit: int = Query(100, ge=1, le=1000, description="Max trades to return"),
):
    """Get recent trades (fills)."""
    try:
        portfolio_mgr = get_portfolio_manager()
        trades = portfolio_mgr.trades
        
        # Return most recent trades
        return trades[-limit:][::-1]  # Reverse to get newest first
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get trades")


# ===================================
# WebSocket Endpoint
# ===================================

@router.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price streaming.
    
    Protocol:
    - Client sends: {"action": "subscribe", "symbols": ["AAPL", "GOOGL"]}
    - Client sends: {"action": "unsubscribe", "symbols": ["AAPL"]}
    - Server sends: {"type": "price_update", "data": {symbol: PriceUpdate}}
    """
    manager = get_connection_manager()
    streamer = get_price_streamer()
    
    # Start streamer if not running
    if not streamer._running:
        await streamer.start()
    
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await manager.send_personal(websocket, {
            "type": "connected",
            "message": "Connected to price stream",
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "subscribe":
                symbols = data.get("symbols", [])
                if symbols:
                    manager.subscribe(websocket, symbols)
                    await manager.send_personal(websocket, {
                        "type": "subscribed",
                        "symbols": symbols,
                    })
            
            elif action == "unsubscribe":
                symbols = data.get("symbols", [])
                if symbols:
                    manager.unsubscribe(websocket, symbols)
                    await manager.send_personal(websocket, {
                        "type": "unsubscribed",
                        "symbols": symbols,
                    })
            
            elif action == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
            
            else:
                await manager.send_personal(websocket, {
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)
