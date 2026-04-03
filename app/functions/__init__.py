"""
Serverless Functions Registration

Registers all local function handlers with the function runner on import.
"""

import logging

from app.infra.functions import get_function_runner
from app.functions.sentiment_analyzer import handle_sentiment_analysis
from app.functions.alert_trigger import handle_alert_trigger
from app.functions.data_fetcher import handle_data_fetch
from app.functions.shap_generator import handle_shap_generation

logger = logging.getLogger(__name__)

# Function name → handler mapping
FUNCTIONS = {
    "sentiment-analyzer": handle_sentiment_analysis,
    "alert-trigger": handle_alert_trigger,
    "data-fetcher": handle_data_fetch,
    "shap-generator": handle_shap_generation,
}


def register_all_functions() -> None:
    """Register all function handlers with the configured runner."""
    runner = get_function_runner()
    for name, handler in FUNCTIONS.items():
        runner.register(name, handler)
    logger.info(f"Registered {len(FUNCTIONS)} serverless functions")
