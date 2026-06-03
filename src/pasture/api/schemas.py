"""
Pydantic request/response models and pandera output validation schema.

Pydantic handles HTTP-level serialisation/validation (FastAPI integration).
Pandera validates the statistical properties of the model's prediction before
it is returned to the caller — catching silent model failures (e.g. all-background
output, NaN health scores) that Pydantic cannot catch.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pydantic import BaseModel, Field, field_validator


# ── Pydantic models ───────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    bbox: list[float] = Field(
        ...,
        min_length=4, max_length=4,
        description="[min_lon, min_lat, max_lon, max_lat] WGS84",
        examples=[[175.2, -38.1, 175.7, -37.6]],
    )
    date_from: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2024-01-01"])
    date_to:   str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2024-03-31"])
    size: list[int] = Field(default=[512, 512], min_length=2, max_length=2)
    max_cloud_coverage: float = Field(default=30.0, ge=0.0, le=100.0)

    @field_validator("bbox")
    @classmethod
    def bbox_valid(cls, v: list[float]) -> list[float]:
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError("bbox longitude values invalid")
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError("bbox latitude values invalid")
        return v


class NdviStats(BaseModel):
    mean: float
    std:  float
    p25:  float
    p75:  float


class TileMeta(BaseModel):
    bbox:  list[float]
    gsd_m: float


class InferResponse(BaseModel):
    mask_b64:     str   = Field(..., description="Base64-encoded PNG colour mask")
    health_score: float = Field(..., ge=0.0, le=100.0, description="0–100 pasture health score")
    ndvi_stats:   NdviStats
    class_pct:    dict[str, float]
    tile_meta:    TileMeta
    inference_ms: int


# ── Pandera output schema ─────────────────────────────────────────────────────
# Validates a single-row DataFrame built from the raw prediction dict.
# Catches silent model failures before the response is returned.

OUTPUT_SCHEMA = pa.DataFrameSchema(
    {
        "health_score": pa.Column(
            float,
            pa.Check.in_range(0.0, 100.0),
            description="must be a valid 0-100 score",
        ),
        "ndvi_mean": pa.Column(
            float,
            pa.Check.in_range(-1.0, 1.0),
            description="NDVI mean must be a valid reflectance ratio",
        ),
        "class_pct_sum": pa.Column(
            float,
            pa.Check.in_range(99.0, 101.0),
            description="class percentages must sum to ~100",
        ),
        "inference_ms": pa.Column(
            int,
            pa.Check.greater_than(0),
        ),
    },
    name="InferenceOutput",
)


def validate_prediction(result: dict) -> None:
    """Raise pandera.errors.SchemaError if prediction result fails validation."""
    row = {
        "health_score":   result["health_score"],
        "ndvi_mean":      result["ndvi_stats"]["mean"],
        "class_pct_sum":  sum(result["class_pct"].values()),
        "inference_ms":   result["inference_ms"],
    }
    df = pd.DataFrame([row])
    OUTPUT_SCHEMA.validate(df)
