"""
FastAPI inference service for pasture health detection.

Endpoints:
    GET  /health          — liveness probe
    POST /infer           — Sentinel-2 tile → mask + health score + NDVI stats
    GET  /drift           — current NDVI drift Z-score vs training baseline
    GET  /metrics         — Prometheus exposition

Environment variables:
    CKPT_PATH             — path to best_segformer.pt
    SENTINEL_CACHE        — tile cache directory
    DATABASE_URL          — PostgreSQL DSN (optional; disables persistence if unset/unreachable)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import pandera.errors
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from pasture.api.db import save_inference
from pasture.api.schemas import InferRequest, InferResponse, validate_prediction
from pasture.inference.predictor import Predictor
from pasture.monitoring.drift import DriftDetector
from pasture.monitoring.forecast import forecast_ndvi
from pasture.monitoring.metrics import (
    health_score_gauge,
    infer_latency,
    ndvi_mean_gauge,
    requests_total,
)

log = logging.getLogger(__name__)

_predictor: Predictor | None = None
_drift = DriftDetector(window=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    ckpt_path = os.environ.get("CKPT_PATH", "data/checkpoints/best_unet.pt")
    cache_dir  = os.environ.get("SENTINEL_CACHE", "data/sentinel_cache")
    _predictor = Predictor(ckpt_path=ckpt_path, sentinel_cache=cache_dir)
    yield


app = FastAPI(
    title="Pasture Health CV",
    description="Sentinel-2 pasture segmentation for NZ agritech",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/drift", tags=["ops"])
def drift() -> dict:
    """Current NDVI drift status vs training baseline."""
    z = _drift.zscore
    return {
        "ndvi_baseline_mean": 0.766,
        "drift_zscore": round(z, 3),
        "alert": abs(z) > 2.0,
        "window": _drift.window,
    }


@app.get("/forecast", tags=["analytics"])
def forecast(aoi: str = "waikato", quarters: int = 4) -> dict:
    """Prophet NDVI forecast for the next N quarters.

    Args:
        aoi: waikato | canterbury | hawkes_bay | marlborough
        quarters: number of future quarters to predict (default 4)
    """
    try:
        return forecast_ndvi(aoi=aoi, quarters_ahead=quarters)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/infer", response_model=InferResponse, tags=["inference"])
def infer(req: InferRequest, background_tasks: BackgroundTasks) -> InferResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        result = _predictor.predict(
            bbox=tuple(req.bbox),
            date_from=req.date_from,
            date_to=req.date_to,
            size=tuple(req.size),
            max_cloud_coverage=req.max_cloud_coverage,
        )
        validate_prediction(result)
    except pandera.errors.SchemaError as e:
        requests_total.labels(status="schema_error").inc()
        raise HTTPException(status_code=422, detail=f"Output schema violation: {e}")
    except RuntimeError as e:
        requests_total.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=str(e))

    elapsed = time.perf_counter() - t0
    infer_latency.observe(elapsed)
    requests_total.labels(status="ok").inc()
    health_score_gauge.set(result["health_score"])
    ndvi_mean_gauge.set(result["ndvi_stats"]["mean"])

    # Drift update — seasonal Z-score if baseline available, rolling fallback otherwise
    _drift.update(
        result["ndvi_stats"]["mean"],
        bbox=req.bbox,
        date_from=req.date_from,
    )

    # DB persistence (background, non-blocking)
    background_tasks.add_task(
        save_inference, result, req.bbox, req.date_from, req.date_to
    )

    return InferResponse(**result)


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"],
         include_in_schema=False)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
