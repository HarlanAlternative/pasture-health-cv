"""
FastAPI inference service for pasture health detection.

Endpoints:
    GET  /health          — liveness probe
    POST /infer           — Sentinel-2 tile → mask + health score + NDVI stats
    GET  /metrics         — Prometheus exposition

Environment variables:
    CKPT_PATH             — path to best_segformer.pt (default: data/checkpoints/best_segformer.pt)
    SENTINEL_CACHE        — tile cache directory (default: data/sentinel_cache)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import pandera.errors
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from pasture.api.schemas import InferRequest, InferResponse, validate_prediction
from pasture.inference.predictor import Predictor
from pasture.monitoring.metrics import (
    health_score_gauge, infer_latency, ndvi_mean_gauge, requests_total,
)

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    ckpt_path = os.environ.get("CKPT_PATH", "data/checkpoints/best_segformer.pt")
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


@app.post("/infer", response_model=InferResponse, tags=["inference"])
def infer(req: InferRequest) -> InferResponse:
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
        validate_prediction(result)  # pandera guard
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

    return InferResponse(**result)


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"],
         include_in_schema=False)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
