from __future__ import annotations

from prometheus_client import Counter, Histogram

PREDICT_TOTAL = Counter(
    "predict_total",
    "Total number of prediction requests",
    ["ticker", "model_type"],
)

PREDICT_CONFIDENCE = Histogram(
    "predict_confidence",
    "Distribution of positive-class prediction confidence scores",
    ["ticker"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_INFERENCE_SECONDS = Histogram(
    "model_inference_seconds",
    "Time spent in model inference (seconds)",
    ["model_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
