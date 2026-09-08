from prometheus_client import Counter, Histogram, Gauge

APP_REQUESTS = Counter(
    "vit_app_requests_total",
    "Total application requests",
    ["page"],
)

INFERENCE_REQUESTS = Counter(
    "vit_inference_requests_total",
    "Total model inference requests",
    ["prediction"],
)

INFERENCE_LATENCY = Histogram(
    "vit_inference_latency_seconds",
    "Model inference latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10),
)

PREDICTION_CONFIDENCE = Histogram(
    "vit_prediction_confidence",
    "Confidence score for model predictions",
    buckets=(0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_LOADED = Gauge(
    "vit_model_loaded",
    "Whether the model is loaded: 1=yes, 0=no",
)