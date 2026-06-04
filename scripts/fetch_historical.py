"""
Fetch 2022-2024 quarterly NDVI time series for all AOIs.

Stores only NDVI statistics (not full masks) into data/historical_ndvi.json.
Used as input for seasonal baseline and Prophet forecasting.

Run once:
    uv run python scripts/fetch_historical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from pasture.data.sentinel_fetch import fetch_sentinel2_tile

PROJECT_ROOT = Path(__file__).parent.parent
CACHE   = str(PROJECT_ROOT / "data" / "sentinel_cache")
OUT     = PROJECT_ROOT / "data" / "historical_ndvi.json"

AOIS = {
    "waikato"    : (175.20, -38.10, 175.70, -37.60),
    "canterbury" : (171.50, -43.80, 172.00, -43.30),
    "hawkes_bay" : (176.60, -39.80, 177.10, -39.30),
    "marlborough": (173.70, -41.80, 174.20, -41.30),
}

QUARTERS = [
    ("Q1", "{y}-01-01", "{y}-03-31", 20),
    ("Q2", "{y}-04-01", "{y}-06-30", 30),
    ("Q3", "{y}-07-01", "{y}-09-30", 50),
    ("Q4", "{y}-10-01", "{y}-12-31", 30),
]

YEARS = [2022, 2023, 2024]


def main() -> None:
    records: dict[str, list] = {aoi: [] for aoi in AOIS}

    for aoi_name, bbox in AOIS.items():
        print(f"\n=== {aoi_name} ===")
        for year in YEARS:
            for qname, t_start, t_end, maxcc in QUARTERS:
                start = t_start.format(y=year)
                end   = t_end.format(y=year)
                label = f"{year}-{qname}"
                try:
                    tile, _ = fetch_sentinel2_tile(
                        bbox=bbox,
                        time_interval=(start, end),
                        size=(512, 512),
                        max_cloud_coverage=float(maxcc),
                        cache_dir=CACHE,
                    )
                    B04, B08 = tile[2], tile[3]
                    ndvi = np.clip((B08 - B04) / (B08 + B04 + 1e-8), -1, 1)
                    entry = {
                        "date"     : label,
                        "year"     : year,
                        "quarter"  : qname,
                        "ndvi_mean": round(float(ndvi.mean()), 4),
                        "ndvi_std" : round(float(ndvi.std()),  4),
                        "ndvi_p25" : round(float(np.percentile(ndvi, 25)), 4),
                        "ndvi_p75" : round(float(np.percentile(ndvi, 75)), 4),
                    }
                    records[aoi_name].append(entry)
                    print(f"  {label}  ndvi_mean={entry['ndvi_mean']:.4f}")
                except RuntimeError as e:
                    print(f"  {label}  SKIP — {e}")

    OUT.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {sum(len(v) for v in records.values())} records → {OUT}")


if __name__ == "__main__":
    main()
