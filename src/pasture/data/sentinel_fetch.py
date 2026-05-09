"""
Sentinel-2 L2A tile fetcher for NZ pasture health detection.

## Assumptions
- sentinelhub-py >= 3.10 (Process API, EvalScript v3)
- Auth via SH_TOKEN (Planet Insights Platform API key) in .env
- Band order returned: index 0=B02 (Blue), 1=B03 (Green), 2=B04 (Red), 3=B08 (NIR)
- Output reflectance is float32 in range [0, ~1]; most vegetated pixels sit in [0.0, 0.35]
- Default AOI covers Hamilton/Waikato dairy belt; Canterbury AOI provided as cloud-free backup
- Free-tier processing units: ~1 PU per 512×512 tile; 30 000 PU/month
- sentinelhub-py returns (H, W, bands) — this module transposes to (bands, H, W) for PyTorch convention
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    SHConfig,
)

# sentinel_fetch.py sits at src/pasture/data/ — project root is 3 levels up
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

# ── Default AOIs (WGS84: min_lon, min_lat, max_lon, max_lat) ─────────────────
WAIKATO_BBOX = (175.20, -38.10, 175.70, -37.60)   # Hamilton dairy belt
CANTERBURY_BBOX = (171.50, -43.80, 172.00, -43.30) # Backup: lower cloud risk

BAND_NAMES = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

# EvalScript V3: return 4 bands as float32 reflectance
_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08"], units: "REFLECTANCE" }],
    output: { bands: 4, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08];
}
"""


@dataclass
class TileMetadata:
    crs: str                  # always "EPSG:4326" for this fetcher
    bbox: tuple               # (min_lon, min_lat, max_lon, max_lat)
    shape: tuple              # (bands, H, W)
    resolution_m: float       # approximate ground sample distance in metres
    time_interval: tuple      # (start, end) ISO strings passed to request
    max_cloud_coverage: float # cloud filter threshold used


def _build_config() -> SHConfig:
    client_id = os.environ.get("SH_CLIENT_ID", "")
    client_secret = os.environ.get("SH_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "SH_CLIENT_ID and SH_CLIENT_SECRET must be set in .env.\n"
            "Get them from: Planet Insights Platform → Account → Users → (your user) → OAuth clients → + New\n"
            "These are NOT the same as the API Key on the profile page."
        )
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    return config


def fetch_sentinel2_tile(
    bbox: tuple = WAIKATO_BBOX,
    time_interval: tuple = ("2024-01-01", "2024-03-31"),
    size: tuple = (512, 512),
    max_cloud_coverage: float = 20.0,
    cache_dir: str | None = "data/sentinel_cache",
) -> tuple[np.ndarray, TileMetadata]:
    """Fetch Sentinel-2 L2A bands B02/B03/B04/B08 for a given AOI.

    Sentinel Hub mosaics the least-cloudy scene within the time window.
    Results are cached locally in cache_dir to save processing units on reruns.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        time_interval: (start, end) ISO date strings. Use NZ summer for lowest cloud.
        size: output pixel dimensions (width, height).
        max_cloud_coverage: reject scenes with cloud cover above this percent.
        cache_dir: local path for response caching. None disables caching.

    Returns:
        data: np.ndarray shape (4, H, W), float32, reflectance [0, ~1].
              band index: 0=B02-Blue  1=B03-Green  2=B04-Red  3=B08-NIR
        meta: TileMetadata with CRS, bbox, shape, and resolution.
    """
    config = _build_config()
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)

    request = SentinelHubRequest(
        evalscript=_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                maxcc=max_cloud_coverage / 100.0,  # API expects 0.0–1.0
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        data_folder=cache_dir,
        config=config,
    )

    data_list = request.get_data(save_data=cache_dir is not None)

    if not data_list or data_list[0] is None:
        raise RuntimeError(
            f"No Sentinel-2 tiles returned for bbox={bbox}, "
            f"time={time_interval}, cloud≤{max_cloud_coverage}%.\n"
            "Try widening the time window or switching to CANTERBURY_BBOX."
        )

    # sentinelhub-py returns (H, W, bands); transpose to (bands, H, W)
    tile = data_list[0].astype(np.float32)   # shape (H, W, 4)
    data = np.transpose(tile, (2, 0, 1))     # → (4, H, W)

    # Approximate GSD: span in degrees × metres/degree ÷ pixels
    lon_span = bbox[2] - bbox[0]
    lat_centre = (bbox[1] + bbox[3]) / 2
    metres_per_deg = 111_320 * np.cos(np.radians(lat_centre))
    resolution_m = round((lon_span * metres_per_deg) / size[0], 1)

    meta = TileMetadata(
        crs="EPSG:4326",
        bbox=bbox,
        shape=data.shape,
        resolution_m=resolution_m,
        time_interval=time_interval,
        max_cloud_coverage=max_cloud_coverage,
    )
    return data, meta
