"""Prometheus metrics for the pasture inference API."""

from prometheus_client import Counter, Gauge, Histogram

infer_latency = Histogram(
    "pasture_infer_latency_seconds",
    "End-to-end inference latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

requests_total = Counter(
    "pasture_requests_total",
    "Total inference requests",
    labelnames=["status"],
)

health_score_gauge = Gauge(
    "pasture_health_score_last",
    "Health score of the most recent inference",
)

ndvi_mean_gauge = Gauge(
    "pasture_ndvi_mean_last",
    "NDVI mean of the most recent inference",
)
