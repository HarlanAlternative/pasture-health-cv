"""
NDVI drift detector — seasonal-aware.

Primary mode (when historical_ndvi.json exists):
    Z-score = (observed_ndvi - seasonal_mean[aoi][quarter]) / seasonal_std[aoi][quarter]
    Avoids false alerts from normal seasonal swings.

Fallback mode (no historical data):
    Z-score = (rolling_mean - NDVI_BASELINE_MEAN) / NDVI_BASELINE_STD
    Same behaviour as before.

Emits two Prometheus gauges:
    pasture_ndvi_drift_zscore   — current Z-score (seasonal or rolling)
    pasture_ndvi_rolling_mean   — rolling mean of recent observations
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from prometheus_client import Gauge

log = logging.getLogger(__name__)

# Fallback constants (Waikato summer annual mean)
_FALLBACK_MEAN = 0.766
_FALLBACK_STD  = 0.202

ndvi_drift_zscore = Gauge(
    "pasture_ndvi_drift_zscore",
    "Z-score of NDVI vs seasonal baseline (|z|>2 → drift alert)",
)
ndvi_rolling_mean = Gauge(
    "pasture_ndvi_rolling_mean",
    "Rolling mean of NDVI across recent inferences",
)


class DriftDetector:
    """Seasonal-aware NDVI drift detector with rolling-mean fallback."""

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self._values: list[float] = []
        self._lock = threading.Lock()

        # Try to load seasonal baseline; silent fallback if not available
        try:
            from pasture.monitoring.seasonal import SeasonalBaseline
            self._baseline = SeasonalBaseline()
            log.info("DriftDetector: seasonal baseline loaded (%s AOIs)",
                     len(self._baseline.aois))
        except FileNotFoundError:
            self._baseline = None
            log.info("DriftDetector: no historical_ndvi.json found, using rolling fallback")

    def update(
        self,
        ndvi_mean: float,
        bbox: tuple | list | None = None,
        date_from: str | None = None,
    ) -> float:
        """Record a new NDVI observation and return the current Z-score."""
        with self._lock:
            self._values.append(ndvi_mean)
            if len(self._values) > self.window:
                self._values.pop(0)
            rolling = float(np.mean(self._values))

        zscore = self._seasonal_zscore(ndvi_mean, bbox, date_from)
        ndvi_drift_zscore.set(zscore)
        ndvi_rolling_mean.set(rolling)
        return zscore

    def _seasonal_zscore(
        self,
        ndvi_mean: float,
        bbox: tuple | list | None,
        date_from: str | None,
    ) -> float:
        if self._baseline and bbox is not None and date_from is not None:
            try:
                from pasture.monitoring.seasonal import detect_aoi, detect_quarter
                aoi     = detect_aoi(bbox)
                quarter = detect_quarter(date_from)
                z = self._baseline.zscore(aoi, quarter, ndvi_mean)
                if z is not None:
                    return round(z, 3)
            except Exception as exc:
                log.warning("Seasonal Z-score failed, using fallback: %s", exc)

        # Fallback: rolling mean vs fixed annual baseline
        with self._lock:
            if len(self._values) < 3:
                return 0.0
            rolling = float(np.mean(self._values))
        return round((rolling - _FALLBACK_MEAN) / _FALLBACK_STD, 3)

    @property
    def zscore(self) -> float:
        return float(ndvi_drift_zscore._value.get())
