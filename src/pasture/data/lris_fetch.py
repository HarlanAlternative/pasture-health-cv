"""
LCDB v5.2 land-cover polygon fetcher for NZ pasture health detection.

## Assumptions
- Primary source: LRIS Portal WFS (requires free API key from lris.scinfo.org.nz)
- Fallback: local GeoPackage at data/lris/lcdb-v52.gpkg (manual download)
- LCDB v5.2 class field: 'Class_2018' (int) — checked at runtime, falls back to 'Cl18'
- CRS of WFS response: EPSG:4326 (requested explicitly via srsName)
- AOI bbox format: (min_lon, min_lat, max_lon, max_lat) WGS84 — same as sentinel_fetch.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import numpy as np
import requests

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

# ── LRIS WFS endpoint ─────────────────────────────────────────────────────────
_WFS_BASE = "https://lris.scinfo.org.nz/services/wfs/"
# Full namespace required: lris.scinfo.org.nz:layer-ID
# layer-104400 = LCDB v5.0 Mainland NZ (v5.2 not separately accessible via WFS)
_LCDB_LAYER = "lris.scinfo.org.nz:layer-104400"

# ── LCDB class code → training label ──────────────────────────────────────────
# 0 = ignored (forest, urban buildings, etc.)
# 1 = healthy pasture   (High/Low Producing Exotic Grassland)
# 2 = stressed/sparse   (Depleted Grassland, crops, tussock)
# 3 = bare soil/gravel
# 4 = water
LCDB_LABEL_MAP: dict[int, int] = {
    40: 1,  # High Producing Exotic Grassland  ← primary dairy target
    41: 1,  # Low Producing Grassland
    42: 2,  # Mixed Exotic Shrubland
    44: 2,  # Depleted Grassland
    43: 2,  # Tall Tussock Grassland
    45: 2,  # Sub-alpine Shrubland
    46: 2,  # Matagouri or Grey Scrub
    30: 2,  # Short-rotation Cropland
    33: 2,  # Permanent Crops
    14: 3,  # Bare/Lightly-vegetated Surfaces
    15: 3,  # Gravel or Rock
    16: 3,  # Sand or Gravel
    64: 4,  # Lake or Pond
    68: 4,  # River
    71: 4,  # Estuarine Open Water
    75: 4,  # Coastal Sand and Gravel (water edge)
}

LABEL_NAMES = {0: "ignored", 1: "healthy_pasture", 2: "stressed", 3: "bare", 4: "water"}


def _class_field(gdf: gpd.GeoDataFrame) -> str:
    """Return whichever LCDB class-code column exists in this GDF."""
    for candidate in ("Class_2018", "Cl18", "CLASS_2018", "class_2018"):
        if candidate in gdf.columns:
            return candidate
    raise KeyError(f"No LCDB class field found. Columns: {list(gdf.columns)}")


def fetch_lcdb_wfs(
    bbox: tuple,
    api_key: str | None = None,
    cache_dir: str = "data/lris_cache",
) -> gpd.GeoDataFrame:
    """Download LCDB v5.2 polygons for a bbox via LRIS WFS.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) WGS84.
        api_key: LRIS Portal API key (lris.scinfo.org.nz — free registration).
                 Falls back to LRIS_API_KEY env var if None.
        cache_dir: local path for caching the downloaded GeoJSON.

    Returns:
        GeoDataFrame in EPSG:4326 with a 'label' column (0–4).
    """
    api_key = api_key or os.environ.get("LRIS_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "LRIS_API_KEY not set.\n"
            "Register free at https://lris.scinfo.org.nz → sign in → top-right avatar → API keys"
        )

    cache_path = Path(cache_dir) / f"lcdb_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.gpkg"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        gdf = gpd.read_file(cache_path, engine="pyogrio")
        return gdf

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": _LCDB_LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
    }
    # LRIS requires Authorization header, not key query param
    headers = {"Authorization": f"key {api_key}"}

    print(f"Downloading LCDB v5.0 for bbox={bbox} …")
    resp = requests.get(_WFS_BASE, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    gdf = gpd.read_file(io.BytesIO(resp.content), engine="pyogrio")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = _attach_labels(gdf)
    gdf.to_file(cache_path, driver="GPKG", engine="pyogrio")
    print(f"Cached to {cache_path}  ({len(gdf)} polygons)")
    return gdf


def load_lcdb_local(path: str | Path, bbox: tuple | None = None) -> gpd.GeoDataFrame:
    """Load LCDB from a locally downloaded file (Shapefile or GeoPackage).

    Manual download: lris.scinfo.org.nz → layer 104400 → Export → GeoPackage
    """
    gdf = gpd.read_file(path, engine="pyogrio", bbox=bbox)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return _attach_labels(gdf)


def _attach_labels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add integer 'label' column using LCDB_LABEL_MAP; unmapped classes → 0."""
    field = _class_field(gdf)
    gdf = gdf.copy()
    gdf["label"] = gdf[field].map(LCDB_LABEL_MAP).fillna(0).astype(np.uint8)
    return gdf


def get_lcdb(
    bbox: tuple,
    local_path: str | Path | None = None,
    cache_dir: str = "data/lris_cache",
) -> gpd.GeoDataFrame:
    """Convenience wrapper: WFS if LRIS_API_KEY set, local file otherwise.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) WGS84.
        local_path: path to locally downloaded LCDB file (optional override).
        cache_dir: WFS response cache directory.
    """
    if local_path and Path(local_path).exists():
        print(f"Loading LCDB from local file: {local_path}")
        return load_lcdb_local(local_path, bbox=bbox)

    return fetch_lcdb_wfs(bbox, cache_dir=cache_dir)
