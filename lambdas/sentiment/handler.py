"""
AWS Lambda handler — Sentiment Analysis.

Thin wrapper around app.functions.sentiment_analyzer.
"""

import asyncio
import json
import sys
import os

# Add project root to path for shared app code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def lambda_handler(event, context):
    """AWS Lambda entry point for sentiment analysis."""
    from app.functions.sentiment_analyzer import handle_sentiment_analysis

    payload = event if isinstance(event, dict) else json.loads(event)
    result = asyncio.get_event_loop().run_until_complete(
        handle_sentiment_analysis(payload)
    )
    return {
        "statusCode": 200,
        "body": json.dumps(result, default=str),
    }
