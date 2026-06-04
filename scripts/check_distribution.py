"""Check per-class pixel distribution across all AOIs and the full patch dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from pasture.data.lris_fetch import get_lcdb
from pasture.data.rasterize import lcdb_to_mask, mask_coverage

BBOXES = {
    "Waikato"    : (175.20, -38.10, 175.70, -37.60),
    "Canterbury" : (171.50, -43.80, 172.00, -43.30),
    "Hawkes Bay" : (176.60, -39.80, 177.10, -39.30),
    "Marlborough": (173.70, -41.80, 174.20, -41.30),
}

CACHE = "data/lris_cache"

print("\n=== LCDB mask class distribution per AOI ===")
print(f"{'AOI':<14} {'healthy':>9} {'stressed':>10} {'bare':>7} {'water':>8}")
print("-" * 52)
for name, bbox in BBOXES.items():
    gdf  = get_lcdb(bbox, cache_dir=CACHE)
    mask = lcdb_to_mask(gdf, bbox=bbox, size=(512, 512))
    cov  = mask_coverage(mask)
    h = cov.get("healthy_pasture", 0)
    s = cov.get("stressed", 0)
    b = cov.get("bare", 0)
    w = cov.get("water", 0)
    print(f"{name:<14} {h:>9.1f}% {s:>9.1f}% {b:>6.1f}% {w:>7.1f}%")

# Also check actual saved patches
print("\n=== Pixel distribution in saved .npz patches ===")
for split in ["train", "val"]:
    patch_dir = Path("data/patches") / split
    patches = sorted(patch_dir.glob("*.npz"))
    if not patches:
        continue
    all_masks = np.concatenate([np.load(p)["mask"].ravel() for p in patches])
    total = all_masks.size
    valid = all_masks[all_masks != 255]
    print(f"\n{split} ({len(patches)} patches, {total:,} pixels):")
    for cls, name in [(0,"ignored"),(1,"healthy"),(2,"stressed"),(3,"bare"),(4,"water")]:
        pct = (valid == cls).sum() / valid.size * 100
        bar = "#" * int(pct / 2)
        print(f"  {cls} {name:<12} {pct:5.1f}%  {bar}")
