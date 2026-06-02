"""
Batch-fetch all seasonal Sentinel-2 tiles + LCDB masks, extract patches.

Run once before training:
    uv run python scripts/prepare_data.py

Output layout:
    data/patches/train/  ← Waikato Q1-Q3 + Canterbury Q1-Q3
    data/patches/val/    ← Waikato Q4 + Canterbury Q4 (held out by season)

## Assumptions
- SH_CLIENT_ID, SH_CLIENT_SECRET, LRIS_API_KEY set in .env
- Higher cloud threshold for NZ winter (Q3, Jul-Sep) — relaxed to 50%
- Patches: 128x128 px, stride 64, min 20% labeled pixels
- Canterbury LCDB fetched separately (different bbox)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pasture.data.lris_fetch import get_lcdb
from pasture.data.rasterize import lcdb_to_mask
from pasture.data.sentinel_fetch import CANTERBURY_BBOX, WAIKATO_BBOX, fetch_sentinel2_tile
from pasture.data.tiling import extract_patches, save_patches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR    = PROJECT_ROOT / "data" / "patches"
SENTINEL_CACHE = str(PROJECT_ROOT / "data" / "sentinel_cache")
LRIS_CACHE     = str(PROJECT_ROOT / "data" / "lris_cache")

TILES = [
    # name              bbox              time_interval                   maxcc  split
    ("waikato_q1",    WAIKATO_BBOX,    ("2024-01-01", "2024-03-31"),   20,   "train"),
    ("waikato_q2",    WAIKATO_BBOX,    ("2024-04-01", "2024-06-30"),   30,   "train"),
    ("waikato_q3",    WAIKATO_BBOX,    ("2024-07-01", "2024-09-30"),   50,   "train"),
    ("waikato_q4",    WAIKATO_BBOX,    ("2024-10-01", "2024-12-31"),   30,   "val"),
    ("canterbury_q1", CANTERBURY_BBOX, ("2024-01-01", "2024-03-31"),   20,   "train"),
    ("canterbury_q2", CANTERBURY_BBOX, ("2024-04-01", "2024-06-30"),   30,   "train"),
    ("canterbury_q3", CANTERBURY_BBOX, ("2024-07-01", "2024-09-30"),   50,   "train"),
    ("canterbury_q4", CANTERBURY_BBOX, ("2024-10-01", "2024-12-31"),   30,   "val"),
]


def main() -> None:
    total_train = total_val = 0

    # Pre-fetch LCDB for both AOIs (cached after first download)
    print("=== Fetching LCDB land-cover polygons ===")
    lcdb_waikato    = get_lcdb(WAIKATO_BBOX,    cache_dir=LRIS_CACHE)
    lcdb_canterbury = get_lcdb(CANTERBURY_BBOX, cache_dir=LRIS_CACHE)
    lcdb_map = {WAIKATO_BBOX: lcdb_waikato, CANTERBURY_BBOX: lcdb_canterbury}
    print()

    for name, bbox, time_interval, maxcc, split in TILES:
        print(f"=== {name} ({split}) ===")

        # 1. Fetch S2 tile
        try:
            image, meta = fetch_sentinel2_tile(
                bbox=bbox,
                time_interval=time_interval,
                size=(512, 512),
                max_cloud_coverage=float(maxcc),
                cache_dir=SENTINEL_CACHE,
            )
        except RuntimeError as e:
            print(f"  SKIP — {e}")
            continue

        print(f"  S2 tile: {meta.shape}  GSD ~{meta.resolution_m} m/px")

        # 2. Rasterize LCDB mask
        gdf  = lcdb_map[bbox]
        mask = lcdb_to_mask(gdf, bbox=bbox, size=(512, 512))
        labeled_pct = (mask > 0).mean() * 100
        print(f"  Mask labeled: {labeled_pct:.1f}%")

        # 3. Extract patches
        patches = extract_patches(image, mask, patch_size=128, stride=64, min_label_frac=0.20)
        print(f"  Patches kept: {len(patches)}")

        # 4. Save
        out_dir = PATCH_DIR / split
        saved   = save_patches(patches, out_dir, prefix=name)
        if split == "train":
            total_train += len(saved)
        else:
            total_val += len(saved)
        print()

    print(f"Done.  Train patches: {total_train}  Val patches: {total_val}")
    print(f"Saved to: {PATCH_DIR}")


if __name__ == "__main__":
    main()
