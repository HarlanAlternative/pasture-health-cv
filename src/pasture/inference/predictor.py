"""
End-to-end inference pipeline: bbox + date → mask + health score + NDVI stats.

## Assumptions
- Checkpoint is SegFormer-B0 by default (best from ablation); path from env CKPT_PATH
- Model runs on CUDA if available, CPU otherwise
- Tile is fetched fresh from Sentinel Hub (cached locally by sentinel_fetch.py)
- Mask encoded as base64 PNG using 5-class colour map
- Health score: weighted fraction of agricultural pixels that are healthy/stressed
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pasture.data.dataset import BAND_MEAN, BAND_STD
from pasture.data.sentinel_fetch import fetch_sentinel2_tile
from pasture.models.factory import build_model

# Colour map matching EDA notebook
_COLOUR_MAP: dict[int, tuple[int, int, int]] = {
    0: (170, 170, 170),  # ignored
    1: (45,  106,  45),  # healthy pasture
    2: (212, 225,  87),  # stressed
    3: (196, 163,  90),  # bare
    4: (79,  195, 247),  # water
}

_LABEL_NAMES = {0: "ignored", 1: "healthy_pasture", 2: "stressed", 3: "bare", 4: "water"}


def _normalise(image: np.ndarray) -> np.ndarray:
    return (image - BAND_MEAN[:, None, None]) / (BAND_STD[:, None, None] + 1e-8)


def _mask_to_png_b64(mask: np.ndarray) -> str:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, colour in _COLOUR_MAP.items():
        rgb[mask == cls] = colour
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _health_score(class_pct: dict[str, float]) -> float:
    """0–100 score over agricultural pixels (excludes water and ignored)."""
    healthy  = class_pct.get("healthy_pasture", 0.0)
    stressed = class_pct.get("stressed", 0.0)
    bare     = class_pct.get("bare", 0.0)
    agri_total = healthy + stressed + bare
    if agri_total < 1e-6:
        return 0.0
    return round(100.0 * (healthy + 0.4 * stressed) / agri_total, 2)


class Predictor:
    """Singleton-style inference engine. Call .predict() from the API handler."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | None = None,
        sentinel_cache: str = "data/sentinel_cache",
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sentinel_cache = sentinel_cache

        ckpt = torch.load(ckpt_path, map_location=self.device)
        arch    = ckpt.get("arch", "segformer")
        encoder = ckpt.get("encoder", "mit_b0")
        self.model = build_model(arch=arch, encoder=encoder, in_channels=4, num_classes=5)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        print(f"Predictor ready — {arch}/{encoder} on {self.device}")

    @torch.no_grad()
    def predict(
        self,
        bbox: tuple[float, float, float, float],
        date_from: str,
        date_to: str,
        size: tuple[int, int] = (512, 512),
        max_cloud_coverage: float = 30.0,
    ) -> dict:
        t0 = time.perf_counter()

        # 1. Fetch Sentinel-2 tile
        tile, meta = fetch_sentinel2_tile(
            bbox=bbox,
            time_interval=(date_from, date_to),
            size=size,
            max_cloud_coverage=max_cloud_coverage,
            cache_dir=self.sentinel_cache,
        )

        # 2. Compute NDVI from raw reflectance before normalisation
        B04, B08 = tile[2], tile[3]
        ndvi = (B08 - B04) / (B08 + B04 + 1e-8)
        ndvi = np.clip(ndvi, -1.0, 1.0)

        # 3. Normalise + run model
        image_norm = _normalise(tile.astype(np.float32))
        tensor = torch.from_numpy(image_norm).unsqueeze(0).to(self.device)
        logits = self.model(tensor)                  # (1, 5, H, W)
        mask   = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # 4. Class distribution (% of all pixels)
        total = mask.size
        class_pct = {
            _LABEL_NAMES[c]: round(float((mask == c).sum() / total * 100), 2)
            for c in range(5)
        }

        # 5. NDVI stats (all pixels)
        ndvi_stats = {
            "mean": round(float(ndvi.mean()), 4),
            "std":  round(float(ndvi.std()),  4),
            "p25":  round(float(np.percentile(ndvi, 25)), 4),
            "p75":  round(float(np.percentile(ndvi, 75)), 4),
        }

        inference_ms = round((time.perf_counter() - t0) * 1000)

        return {
            "mask_b64":     _mask_to_png_b64(mask),
            "health_score": _health_score(class_pct),
            "ndvi_stats":   ndvi_stats,
            "class_pct":    class_pct,
            "tile_meta":    {"bbox": list(bbox), "gsd_m": meta.resolution_m},
            "inference_ms": inference_ms,
        }
