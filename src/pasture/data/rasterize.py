"""
Vector-to-raster alignment for LCDB land-cover labels.

## Assumptions
- Input GeoDataFrame is in EPSG:4326 (enforced by lris_fetch.py)
- bbox format: (min_lon, min_lat, max_lon, max_lat) — same convention throughout
- Output mask shape: (H, W) uint8, matching the Sentinel-2 tile size
- Rasterio uses (row, col) = (lat-descending, lon-ascending) pixel convention
- Polygons that fall outside the bbox are silently excluded by rasterio
- Fill value 255 marks pixels with no LCDB coverage (should be rare inside NZ)
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize as _rio_rasterize
from rasterio.transform import from_bounds

from pasture.data.lris_fetch import LABEL_NAMES


def lcdb_to_mask(
    gdf: gpd.GeoDataFrame,
    bbox: tuple,
    size: tuple = (512, 512),
    label_col: str = "label",
    nodata: int = 255,
) -> np.ndarray:
    """Burn LCDB polygons into a pixel mask aligned with a Sentinel-2 tile.

    Args:
        gdf: GeoDataFrame in EPSG:4326 with an integer label column.
        bbox: (min_lon, min_lat, max_lon, max_lat) — same bbox used for S2 fetch.
        size: (width, height) in pixels — must match the S2 tile size.
        label_col: column name in gdf that holds integer class labels (0–4).
        nodata: fill value for pixels outside all polygons.

    Returns:
        mask: np.ndarray shape (H, W) uint8.
              Values: 0=ignored  1=healthy  2=stressed  3=bare  4=water  255=nodata
    """
    width, height = size
    # Affine transform: maps pixel (row=0,col=0) to top-left of bbox
    # from_bounds(west, south, east, north, width, height) — positional only
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

    # Build (geometry, label_value) pairs — rasterio burns in list order (last wins)
    shapes = [
        (geom, int(label))
        for geom, label in zip(gdf.geometry, gdf[label_col])
        if geom is not None and not geom.is_empty
    ]

    if not shapes:
        return np.full((height, width), nodata, dtype=np.uint8)

    mask = _rio_rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=np.uint8,
    )
    return mask


def ndvi_pseudo_mask(ndvi: np.ndarray) -> np.ndarray:
    """Generate weak labels from NDVI thresholds — no external data needed.

    This is a self-supervised fallback used when LRIS ground-truth is unavailable.
    Thresholds are calibrated for NZ dairy farmland in summer.

    Label scheme (matches LCDB labels):
        1 = healthy pasture  (NDVI > 0.5)
        2 = stressed/sparse  (NDVI 0.2–0.5)
        3 = bare soil        (NDVI 0.0–0.2)
        4 = water            (NDVI < 0.0)
    """
    mask = np.zeros(ndvi.shape, dtype=np.uint8)
    mask[ndvi < 0.0] = 4
    mask[(ndvi >= 0.0) & (ndvi < 0.2)] = 3
    mask[(ndvi >= 0.2) & (ndvi < 0.5)] = 2
    mask[ndvi >= 0.5] = 1
    return mask


def mask_coverage(mask: np.ndarray, nodata: int = 255) -> dict[str, float]:
    """Return per-class pixel percentage (excluding nodata pixels)."""
    valid = mask[mask != nodata]
    total = valid.size
    if total == 0:
        return {}
    return {
        LABEL_NAMES.get(cls, str(cls)): round((valid == cls).sum() / total * 100, 2)
        for cls in sorted(np.unique(valid))
    }
