"""
Seasonal NDVI baseline and anomaly detection.

Method: climatological mean ± std per quarter, computed from 2022-2024 history.
Z-score = (current - seasonal_mean) / seasonal_std

A Z-score < -2 indicates the pasture is significantly greener than usual (good),
> +2 indicates drought / stress (alert).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DEFAULT_HISTORY = Path(__file__).resolve().parents[3] / "data" / "historical_ndvi.json"


class SeasonalBaseline:
    """Per-AOI, per-quarter NDVI climatology."""

    def __init__(self, history_path: str | Path = _DEFAULT_HISTORY) -> None:
        path = Path(history_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Historical NDVI not found at {path}. "
                "Run scripts/fetch_historical.py first."
            )
        self._data: dict[str, list] = json.loads(path.read_text())
        self._stats = self._build_stats()

    def _build_stats(self) -> dict[str, dict[str, dict]]:
        """stats[aoi][quarter] = {mean, std, n}"""
        stats: dict[str, dict] = {}
        for aoi, records in self._data.items():
            stats[aoi] = {}
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                vals = [r["ndvi_mean"] for r in records if r["quarter"] == q]
                if vals:
                    stats[aoi][q] = {
                        "mean": float(np.mean(vals)),
                        "std" : max(float(np.std(vals)), 0.01),
                        "n"   : len(vals),
                    }
        return stats

    def zscore(self, aoi: str, quarter: str, ndvi_mean: float) -> float | None:
        """Return seasonal Z-score for a given AOI + quarter + observed NDVI."""
        s = self._stats.get(aoi, {}).get(quarter)
        if s is None:
            return None
        return round((ndvi_mean - s["mean"]) / s["std"], 3)

    def summary(self, aoi: str) -> list[dict]:
        """Return seasonal profile for one AOI."""
        result = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            s = self._stats.get(aoi, {}).get(q, {})
            result.append({"quarter": q, **s})
        return result

    @property
    def aois(self) -> list[str]:
        return list(self._stats.keys())
