"""
Agents API - Simplified AI Agents for Analysis

POST /agents/technical - Technical analysis with predict + SHAP
POST /agents/summary - Portfolio summary agent

Outputs structured JSON that can be used as RAG evidence.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ===================================
# Request / Response Schemas
# ===================================
class TechnicalAnalysisRequest(BaseModel):
    """Request for technical analysis agent."""
    
    ticker: str
    model_id: str | None = None  # Uses promoted if not specified
    include_shap: bool = True
    include_features: bool = True
    top_features: int = Field(default=5, ge=1, le=20)


class FeatureContribution(BaseModel):
    """A feature's contribution to the prediction."""
    
    name: str
    value: float
    contribution: float  # SHAP value
    direction: Literal["bullish", "bearish", "neutral"]
    description: str


class TechnicalSignal(BaseModel):
    """A technical signal derived from features."""
    
    indicator: str
    signal: Literal["bullish", "bearish", "neutral"]
    strength: Literal["strong", "moderate", "weak"]
    description: str


class TechnicalAnalysisResponse(BaseModel):
    """Response from technical analysis agent."""
    
    # Status
    success: bool
    error: str | None = None
    
    # Ticker info
    ticker: str
    model_id: str | None = None
    timestamp: str
    
    # Prediction
    prediction: Literal["up", "down"] | None = None
    probability: float | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    
    # Analysis
    summary: str = ""
    signals: list[TechnicalSignal] = []
    top_features: list[FeatureContribution] = []
    
    # Raw data (for RAG indexing)
    raw_features: dict[str, float] = {}
    shap_values: dict[str, float] = {}
    
    # Metadata
    evidence_type: str = "technical_analysis"
    can_index: bool = True


class PortfolioSummaryRequest(BaseModel):
    """Request for portfolio summary agent."""
    
    tickers: list[str] = Field(min_length=1, max_length=20)
    model_id: str | None = None


class TickerAnalysis(BaseModel):
    """Analysis for a single ticker in portfolio."""
    
    ticker: str
    prediction: Literal["up", "down"]
    probability: float
    signal: Literal["bullish", "bearish", "neutral"]
    top_driver: str


class PortfolioSummaryResponse(BaseModel):
    """Response from portfolio summary agent."""
    
    success: bool
    error: str | None = None
    
    # Summary
    overall_signal: Literal["bullish", "bearish", "mixed"]
    bullish_count: int
    bearish_count: int
    
    # Per-ticker
    analyses: list[TickerAnalysis] = []
    
    # Narrative
    summary: str = ""
    
    evidence_type: str = "portfolio_summary"


# ===================================
# POST /agents/technical
# ===================================
@router.post("/agents/technical", response_model=TechnicalAnalysisResponse)
def technical_analysis(request: TechnicalAnalysisRequest):
    """
    Run technical analysis agent on a ticker.
    
    Combines:
    - Model prediction (up/down probability)
    - SHAP feature importance
    - Technical indicator interpretation
    
    Returns structured JSON suitable for:
    - UI display
    - RAG indexing
    - Further agent reasoning
    """
    from datetime import datetime
    from app.services.predict_service import PredictionService
    from app.services.model_cache import get_model_cache
    
    try:
        # Get model
        cache = get_model_cache()
        if request.model_id:
            model_id = request.model_id
        else:
            model_id = cache.get_promoted_id()
        
        if not model_id:
            return TechnicalAnalysisResponse(
                success=False,
                error="No model available. Train and promote a model first.",
                ticker=request.ticker,
                timestamp=datetime.utcnow().isoformat(),
            )
        
        # Get prediction
        pred_service = PredictionService()
        pred_result = pred_service.predict(
            ticker=request.ticker,
            model_id=model_id,
        )
        
        if not pred_result.get("success"):
            return TechnicalAnalysisResponse(
                success=False,
                error=pred_result.get("error", "Prediction failed"),
                ticker=request.ticker,
                model_id=model_id,
                timestamp=datetime.utcnow().isoformat(),
            )
        
        # Extract prediction
        prob_up = pred_result.get("probability", {}).get("up", 0.5)
        prediction = "up" if pred_result.get("prediction") == 1 else "down"
        confidence = _get_confidence_level(prob_up)
        
        # Get SHAP explanation if requested
        shap_values = {}
        top_features = []
        signals = []
        raw_features = {}
        
        if request.include_shap:
            shap_result = _get_shap_explanation(request.ticker, model_id)
            if shap_result:
                shap_values = shap_result.get("shap_values", {})
                raw_features = shap_result.get("feature_values", {})
                
                # Build top features
                top_features = _build_top_features(
                    shap_values, 
                    raw_features, 
                    request.top_features
                )
                
                # Build signals
                signals = _build_technical_signals(raw_features)
        
        # Generate summary
        summary = _generate_summary(
            ticker=request.ticker,
            prediction=prediction,
            prob_up=prob_up,
            confidence=confidence,
            top_features=top_features,
            signals=signals,
        )
        
        return TechnicalAnalysisResponse(
            success=True,
            ticker=request.ticker,
            model_id=model_id,
            timestamp=datetime.utcnow().isoformat(),
            prediction=prediction,
            probability=round(prob_up, 4),
            confidence=confidence,
            summary=summary,
            signals=signals,
            top_features=top_features,
            raw_features=raw_features,
            shap_values=shap_values,
        )
    
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}", exc_info=True)
        return TechnicalAnalysisResponse(
            success=False,
            error=str(e),
            ticker=request.ticker,
            timestamp=datetime.utcnow().isoformat(),
        )


# ===================================
# POST /agents/summary
# ===================================
@router.post("/agents/summary", response_model=PortfolioSummaryResponse)
def portfolio_summary(request: PortfolioSummaryRequest):
    """
    Run portfolio summary agent on multiple tickers.
    
    Aggregates predictions across tickers and provides
    an overall market view.
    """
    from app.services.predict_service import PredictionService
    from app.services.model_cache import get_model_cache
    
    try:
        cache = get_model_cache()
        model_id = request.model_id or cache.get_promoted_id()
        
        if not model_id:
            return PortfolioSummaryResponse(
                success=False,
                error="No model available",
            )
        
        pred_service = PredictionService()
        analyses = []
        bullish = 0
        bearish = 0
        
        for ticker in request.tickers:
            result = pred_service.predict(ticker=ticker, model_id=model_id)
            
            if result.get("success"):
                prob_up = result.get("probability", {}).get("up", 0.5)
                pred = "up" if result.get("prediction") == 1 else "down"
                
                if prob_up > 0.55:
                    signal = "bullish"
                    bullish += 1
                elif prob_up < 0.45:
                    signal = "bearish"
                    bearish += 1
                else:
                    signal = "neutral"
                
                analyses.append(TickerAnalysis(
                    ticker=ticker,
                    prediction=pred,
                    probability=round(prob_up, 4),
                    signal=signal,
                    top_driver="momentum",  # Simplified
                ))
        
        # Overall signal
        if bullish > bearish * 1.5:
            overall = "bullish"
        elif bearish > bullish * 1.5:
            overall = "bearish"
        else:
            overall = "mixed"
        
        # Summary narrative
        summary = _generate_portfolio_summary(
            analyses=analyses,
            overall=overall,
            bullish=bullish,
            bearish=bearish,
        )
        
        return PortfolioSummaryResponse(
            success=True,
            overall_signal=overall,
            bullish_count=bullish,
            bearish_count=bearish,
            analyses=analyses,
            summary=summary,
        )
    
    except Exception as e:
        logger.error(f"Portfolio summary failed: {e}", exc_info=True)
        return PortfolioSummaryResponse(
            success=False,
            error=str(e),
        )


# ===================================
# Helper Functions
# ===================================
def _get_confidence_level(prob: float) -> Literal["high", "medium", "low"]:
    """Convert probability to confidence level."""
    deviation = abs(prob - 0.5)
    if deviation > 0.2:
        return "high"
    elif deviation > 0.1:
        return "medium"
    return "low"


def _get_shap_explanation(ticker: str, model_id: str) -> dict | None:
    """Get SHAP explanation for a ticker."""
    try:
        from app.explain.shap_explainer import ShapExplainer
        
        explainer = ShapExplainer()
        result = explainer.explain(ticker=ticker, model_id=model_id)
        
        if result.get("success"):
            return {
                "shap_values": result.get("shap_values", {}),
                "feature_values": result.get("feature_values", {}),
            }
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
    
    return None


def _build_top_features(
    shap_values: dict,
    feature_values: dict,
    top_n: int,
) -> list[FeatureContribution]:
    """Build top feature contributions."""
    if not shap_values:
        return []
    
    # Sort by absolute SHAP value
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]
    
    result = []
    for name, shap_val in sorted_features:
        value = feature_values.get(name, 0)
        
        if shap_val > 0.01:
            direction = "bullish"
        elif shap_val < -0.01:
            direction = "bearish"
        else:
            direction = "neutral"
        
        desc = _get_feature_description(name, value, direction)
        
        result.append(FeatureContribution(
            name=name,
            value=round(value, 4),
            contribution=round(shap_val, 4),
            direction=direction,
            description=desc,
        ))
    
    return result


def _get_feature_description(name: str, value: float, direction: str) -> str:
    """Get human-readable description for a feature."""
    name_lower = name.lower()
    
    if "rsi" in name_lower:
        if value > 70:
            return "RSI overbought (>70), potential reversal"
        elif value < 30:
            return "RSI oversold (<30), potential bounce"
        return f"RSI at {value:.1f}, neutral momentum"
    
    elif "macd" in name_lower:
        if value > 0:
            return "MACD positive, bullish momentum"
        return "MACD negative, bearish momentum"
    
    elif "sma" in name_lower or "ema" in name_lower:
        return f"Moving average signal: {direction}"
    
    elif "volatility" in name_lower or "atr" in name_lower:
        if value > 0.03:
            return "High volatility environment"
        return "Low volatility environment"
    
    elif "volume" in name_lower:
        return "Volume indicator signal"
    
    elif "bb" in name_lower or "bollinger" in name_lower:
        return "Bollinger band signal"
    
    return f"{name}: {direction} signal"


def _build_technical_signals(features: dict) -> list[TechnicalSignal]:
    """Build technical signals from feature values."""
    signals = []
    
    # RSI
    for key, val in features.items():
        if "rsi" in key.lower():
            if val > 70:
                signals.append(TechnicalSignal(
                    indicator="RSI",
                    signal="bearish",
                    strength="strong",
                    description=f"RSI at {val:.1f} indicates overbought conditions",
                ))
            elif val < 30:
                signals.append(TechnicalSignal(
                    indicator="RSI",
                    signal="bullish",
                    strength="strong",
                    description=f"RSI at {val:.1f} indicates oversold conditions",
                ))
            break
    
    # MACD
    for key, val in features.items():
        if "macd" in key.lower() and "signal" not in key.lower():
            if val > 0:
                signals.append(TechnicalSignal(
                    indicator="MACD",
                    signal="bullish",
                    strength="moderate",
                    description="MACD above zero line",
                ))
            else:
                signals.append(TechnicalSignal(
                    indicator="MACD",
                    signal="bearish",
                    strength="moderate",
                    description="MACD below zero line",
                ))
            break
    
    return signals


def _generate_summary(
    ticker: str,
    prediction: str,
    prob_up: float,
    confidence: str,
    top_features: list,
    signals: list,
) -> str:
    """Generate natural language summary."""
    direction = "bullish" if prediction == "up" else "bearish"
    prob_pct = prob_up * 100 if prediction == "up" else (1 - prob_up) * 100
    
    summary = f"{ticker} shows a {direction} outlook ({prob_pct:.1f}% probability) with {confidence} confidence. "
    
    if top_features:
        top = top_features[0]
        summary += f"Primary driver: {top.name} ({top.direction}). "
    
    if signals:
        signal_strs = [f"{s.indicator} ({s.signal})" for s in signals[:2]]
        summary += f"Key signals: {', '.join(signal_strs)}."
    
    return summary


def _generate_portfolio_summary(
    analyses: list,
    overall: str,
    bullish: int,
    bearish: int,
) -> str:
    """Generate portfolio summary narrative."""
    total = len(analyses)
    
    if overall == "bullish":
        summary = f"Portfolio outlook is bullish with {bullish}/{total} tickers showing upward momentum. "
    elif overall == "bearish":
        summary = f"Portfolio outlook is bearish with {bearish}/{total} tickers showing downward pressure. "
    else:
        summary = f"Portfolio shows mixed signals with {bullish} bullish and {bearish} bearish tickers. "
    
    # Highlight strongest signals
    sorted_analyses = sorted(analyses, key=lambda x: abs(x.probability - 0.5), reverse=True)
    if sorted_analyses:
        top = sorted_analyses[0]
        summary += f"Strongest signal: {top.ticker} ({top.signal}, {top.probability:.1%})."
    
    return summary
