"""
AWS Lambda handler — SHAP Explainer.

Thin wrapper around app.functions.shap_generator.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def lambda_handler(event, context):
    """AWS Lambda entry point for SHAP explanation generation."""
    from app.functions.shap_generator import handle_shap_generation

    payload = event if isinstance(event, dict) else json.loads(event)
    result = asyncio.get_event_loop().run_until_complete(
        handle_shap_generation(payload)
    )
    return {
        "statusCode": 200,
        "body": json.dumps(result, default=str),
    }
