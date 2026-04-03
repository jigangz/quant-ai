"""
Sentiment Analysis Function

Analyzes news text and returns a sentiment score (-1 to 1).
Uses Claude Haiku when ANTHROPIC_API_KEY is set, otherwise falls back
to a keyword-based scorer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword-based fallback scorer
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS = [
    "beat", "beats", "exceeded", "surpass", "surge", "surged", "rally",
    "upgrade", "upgraded", "bullish", "growth", "profit", "record",
    "outperform", "buy", "positive", "gains", "soar", "strong",
    "innovation", "breakthrough", "partnership", "expansion", "dividend",
]

BEARISH_KEYWORDS = [
    "miss", "missed", "decline", "drop", "fell", "crash", "downgrade",
    "downgraded", "bearish", "loss", "losses", "selloff", "sell-off",
    "underperform", "sell", "negative", "weak", "warning", "risk",
    "lawsuit", "recall", "layoff", "layoffs", "bankruptcy", "debt",
]


def _keyword_sentiment(text: str) -> float:
    """Simple keyword-based sentiment scorer (-1 to 1)."""
    text_lower = text.lower()
    bullish = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bearish = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bullish + bearish
    if total == 0:
        return 0.0
    return round((bullish - bearish) / total, 4)


# ---------------------------------------------------------------------------
# Claude Haiku sentiment scorer
# ---------------------------------------------------------------------------

SENTIMENT_PROMPT = """\
You are a financial sentiment analyst. Rate the sentiment of the following text \
on a scale from -1.0 (very bearish) to 1.0 (very bullish). Respond with ONLY \
a JSON object: {"score": <float>, "reasoning": "<one sentence>"}

Text: {text}
"""


async def _claude_sentiment(text: str) -> dict[str, Any]:
    """Analyze sentiment using Claude Haiku."""
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed, using keyword fallback")
        return {"score": _keyword_sentiment(text), "method": "keyword"}

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": SENTIMENT_PROMPT.format(text=text),
                }
            ],
        )
        raw = response.content[0].text.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object in response")

        result = json.loads(raw[start:end])
        return {
            "score": round(float(result["score"]), 4),
            "reasoning": result.get("reasoning", ""),
            "method": "claude",
        }
    except Exception as e:
        logger.warning(f"Claude sentiment failed, falling back to keywords: {e}")
        return {"score": _keyword_sentiment(text), "method": "keyword"}


# ---------------------------------------------------------------------------
# Function handler (registered with LocalRunner)
# ---------------------------------------------------------------------------

async def handle_sentiment_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Sentiment analysis function handler.

    Payload:
        text (str): News text to analyze.

    Returns:
        score (float): Sentiment score from -1 to 1.
        method (str): "claude" or "keyword".
        reasoning (str): Explanation (Claude only).
    """
    text = payload.get("text", "")
    if not text:
        return {"success": False, "error": "Missing 'text' in payload"}

    if settings.ANTHROPIC_API_KEY:
        result = await _claude_sentiment(text)
    else:
        result = {
            "score": _keyword_sentiment(text),
            "method": "keyword",
        }

    return {"success": True, **result}
