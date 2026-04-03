"""
AWS Lambda handler — Alert Trigger.

Thin wrapper around app.functions.alert_trigger.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def lambda_handler(event, context):
    """AWS Lambda entry point for alert evaluation."""
    from app.functions.alert_trigger import handle_alert_trigger

    payload = event if isinstance(event, dict) else json.loads(event)
    result = asyncio.get_event_loop().run_until_complete(
        handle_alert_trigger(payload)
    )
    return {
        "statusCode": 200,
        "body": json.dumps(result, default=str),
    }
