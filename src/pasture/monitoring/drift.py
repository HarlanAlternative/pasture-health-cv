"""
NDVI distribution drift detector.

Compares a rolling window of recent inference NDVI means against a baseline
computed from the training tiles (Waikato + Canterbury 2024 summer).
Emits a Prometheus gauge so Prometheus alerting rules can fire.

## Assumptions
- Baseline: mean=0.766, std=0.202 (from prepare_data.py training tiles)
- Window: last 10 inferences (configurable)
- Z-score > 2 or < -2 → meaningful drift
- Thread-safe via a simple lock (single-worker uvicorn)
"""

from __future__ import annotations

import threading

import numpy as np
from prometheus_client import Gauge

# Sentinel-2 NDVI baseline from NZ training data (Waikato + Canterbury, 2024)
NDVI_BASELINE_MEAN = 0.766
NDVI_BASELINE_STD  = 0.202

ndvi_drift_zscore = Gauge(
    "pasture_ndvi_drift_zscore",
    "Z-score of rolling NDVI mean vs training baseline (|z|>2 → drift alert)",
)

ndvi_rolling_mean = Gauge(
    "pasture_ndvi_rolling_mean",
    "Rolling mean of NDVI across recent inferences",
)


class DriftDetector:
    """Rolling-window NDVI drift detector."""

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self._values: list[float] = []
        self._lock = threading.Lock()

    def update(self, ndvi_mean: float) -> float:
        """Record a new NDVI mean and return the current drift Z-score."""
        with self._lock:
            self._values.append(ndvi_mean)
            if len(self._values) > self.window:
                self._values.pop(0)
            zscore = self._compute_zscore()

        ndvi_drift_zscore.set(zscore)
        ndvi_rolling_mean.set(float(np.mean(self._values)))
        return zscore

    def _compute_zscore(self) -> float:
        if len(self._values) < 3:
            return 0.0
        mu = float(np.mean(self._values))
        return (mu - NDVI_BASELINE_MEAN) / (NDVI_BASELINE_STD + 1e-8)

    @property
    def zscore(self) -> float:
        with self._lock:
            return self._compute_zscore()
