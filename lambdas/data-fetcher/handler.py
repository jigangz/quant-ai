"""
AWS Lambda handler — Scheduled Data Fetcher.

Thin wrapper around app.functions.data_fetcher.
Designed to be triggered by EventBridge schedule.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def lambda_handler(event, context):
    """AWS Lambda entry point for scheduled data fetch."""
    from app.functions.data_fetcher import handle_data_fetch

    # EventBridge events may not have a standard payload
    if isinstance(event, dict) and "symbols" in event:
        payload = event
    else:
        payload = {}

    result = asyncio.get_event_loop().run_until_complete(
        handle_data_fetch(payload)
    )
    return {
        "statusCode": 200,
        "body": json.dumps(result, default=str),
    }
