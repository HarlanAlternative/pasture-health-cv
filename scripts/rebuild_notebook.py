"""Rebuild 01_eda_sentinel2.ipynb with correct UTF-8 (no BOM)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "01_eda_sentinel2.ipynb"


def cell(source: str, cell_type: str = "code") -> dict:
    base = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def md(source: str) -> dict:
    return cell(source, "markdown")


cells = [
    md(
        "# Sprint 1 EDA — Sentinel-2 NDVI + LCDB Weak Labels\n"
        "\n"
        "**Section 1:** Sentinel-2 L2A tile fetch + NDVI visualization  \n"
        "**Section 2:** LRIS LCDB v5.2 weak-label rasterization\n"
        "\n"
        "Prerequisites:\n"
        "- `.env` with `SH_CLIENT_ID`, `SH_CLIENT_SECRET`, `LRIS_API_KEY`\n"
        "- `uv sync` run from project root, kernel set to `pasture-cv`"
    ),
    cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "project_root = Path().resolve().parent\n"
        "if str(project_root / 'src') not in sys.path:\n"
        "    sys.path.insert(0, str(project_root / 'src'))\n"
        "\n"
        "from pasture.data.sentinel_fetch import (\n"
        "    fetch_sentinel2_tile, WAIKATO_BBOX, CANTERBURY_BBOX, BAND_NAMES,\n"
        ")\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "print('Imports OK')\n"
        "print(f'Project root: {project_root}')\n"
        "print(f'Waikato AOI : {WAIKATO_BBOX}')"
    ),
    md("## Section 1 — Sentinel-2 Fetch + NDVI"),
    cell(
        "print('Fetching tile (cached after first call)...')\n"
        "\n"
        "data, meta = fetch_sentinel2_tile(\n"
        "    bbox=WAIKATO_BBOX,\n"
        "    time_interval=('2024-01-01', '2024-03-31'),\n"
        "    size=(512, 512),\n"
        "    max_cloud_coverage=20.0,\n"
        "    cache_dir=str(project_root / 'data' / 'sentinel_cache'),\n"
        ")\n"
        "\n"
        "B02, B03, B04, B08 = data[0], data[1], data[2], data[3]\n"
        "print(f'Shape : {data.shape}  dtype: {data.dtype}')\n"
        "print(f'GSD   : ~{meta.resolution_m} m/px')\n"
        "for i, name in enumerate(BAND_NAMES):\n"
        "    b = data[i]\n"
        "    print(f'  {name}: min={b.min():.4f}  max={b.max():.4f}  mean={b.mean():.4f}')"
    ),
    cell(
        "ndvi = (B08 - B04) / (B08 + B04 + 1e-8)\n"
        "ndvi = np.clip(ndvi, -1.0, 1.0)\n"
        "\n"
        "print(f'NDVI min={ndvi.min():.3f}  max={ndvi.max():.3f}  mean={ndvi.mean():.3f}')\n"
        "print()\n"
        "total = ndvi.size\n"
        "for label, lo, hi in [\n"
        "    ('Water/shadow  (< 0.0)', -1.0, 0.0),\n"
        "    ('Bare soil     (0.0-0.2)', 0.0, 0.2),\n"
        "    ('Stressed      (0.2-0.4)', 0.2, 0.4),\n"
        "    ('Moderate      (0.4-0.6)', 0.4, 0.6),\n"
        "    ('Healthy       (> 0.6)',   0.6, 1.0),\n"
        "]:\n"
        "    pct = ((ndvi >= lo) & (ndvi < hi)).sum() / total * 100\n"
        "    print(f'  {label}: {pct:.1f}%')"
    ),
    cell(
        "fig, axes = plt.subplots(1, 2, figsize=(15, 7))\n"
        "fig.suptitle(\n"
        "    f'Sentinel-2 L2A -- Waikato, NZ | {meta.time_interval[0]} to {meta.time_interval[1]}\\n'\n"
        "    f'AOI: {meta.bbox} | GSD ~{meta.resolution_m} m/px | cloud <= {meta.max_cloud_coverage}%',\n"
        "    fontsize=11,\n"
        ")\n"
        "\n"
        "rgb = np.stack([B04, B03, B02], axis=-1)\n"
        "p2, p98 = np.percentile(rgb, [2, 98])\n"
        "rgb_display = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)\n"
        "axes[0].imshow(rgb_display)\n"
        "axes[0].set_title('True Colour (B04/B03/B02)')\n"
        "axes[0].axis('off')\n"
        "\n"
        "im = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)\n"
        "axes[1].set_title('NDVI  (Red=bare/stressed  Green=healthy pasture)')\n"
        "axes[1].axis('off')\n"
        "plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='NDVI')\n"
        "\n"
        "plt.tight_layout()\n"
        "out = project_root / 'data' / 'eda_ndvi_waikato.png'\n"
        "plt.savefig(out, dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved to {out}')"
    ),
    cell(
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.hist(ndvi.ravel(), bins=100, range=(-0.3, 1.0), color='#2ecc71', edgecolor='none', alpha=0.8)\n"
        "ax.axvline(0.4, color='red', linestyle='--', label='0.4 threshold')\n"
        "ax.set_xlabel('NDVI')\n"
        "ax.set_ylabel('Pixel count')\n"
        "ax.set_title('NDVI distribution -- Waikato tile')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "---\n"
        "## Section 2 — LRIS LCDB v5.2 Weak Labels\n"
        "\n"
        "Download land-cover polygons from the LRIS WFS and burn them into a\n"
        "512x512 mask aligned pixel-for-pixel with the Sentinel-2 tile.\n"
        "\n"
        "**Label scheme:**\n"
        "| Value | Class |\n"
        "|---|---|\n"
        "| 0 | ignored (forest, urban, ...) |\n"
        "| 1 | healthy pasture |\n"
        "| 2 | stressed / sparse |\n"
        "| 3 | bare soil / gravel |\n"
        "| 4 | water |\n"
        "| 255 | no LCDB coverage |\n"
        "\n"
        "**Prerequisite:** `LRIS_API_KEY` in `.env`  \n"
        "Register free at https://lris.scinfo.org.nz -> sign in -> avatar -> API keys -> + New key"
    ),
    cell(
        "from pasture.data.lris_fetch import get_lcdb, LABEL_NAMES\n"
        "from pasture.data.rasterize import lcdb_to_mask, mask_coverage\n"
        "\n"
        "gdf = get_lcdb(\n"
        "    bbox=WAIKATO_BBOX,\n"
        "    cache_dir=str(project_root / 'data' / 'lris_cache'),\n"
        ")\n"
        "\n"
        "print(f'Polygons : {len(gdf)}')\n"
        "print(f'CRS      : {gdf.crs}')\n"
        "print()\n"
        "print('Label distribution:')\n"
        "print(gdf['label'].value_counts().sort_index().rename(LABEL_NAMES))"
    ),
    cell(
        "mask = lcdb_to_mask(gdf, bbox=WAIKATO_BBOX, size=(512, 512))\n"
        "\n"
        "print(f'Mask shape : {mask.shape}  dtype: {mask.dtype}')\n"
        "print(f'Unique values: {np.unique(mask)}')\n"
        "print()\n"
        "print('Coverage (% of valid pixels):')\n"
        "for cls, pct in mask_coverage(mask).items():\n"
        "    print(f'  {cls:<20} {pct:5.1f}%')"
    ),
    cell(
        "import matplotlib.colors as mcolors\n"
        "from matplotlib.patches import Patch\n"
        "\n"
        "cmap = mcolors.ListedColormap(['#aaaaaa', '#2d6a2d', '#d4e157', '#c4a35a', '#4fc3f7'])\n"
        "norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n"
        "fig.suptitle('Waikato -- True Colour | NDVI | LCDB Weak Labels', fontsize=12)\n"
        "\n"
        "axes[0].imshow(rgb_display)\n"
        "axes[0].set_title('True Colour')\n"
        "axes[0].axis('off')\n"
        "\n"
        "axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)\n"
        "axes[1].set_title('NDVI')\n"
        "axes[1].axis('off')\n"
        "\n"
        "axes[2].imshow(mask, cmap=cmap, norm=norm, interpolation='nearest')\n"
        "axes[2].set_title('LCDB Weak Labels')\n"
        "axes[2].axis('off')\n"
        "\n"
        "legend_elements = [\n"
        "    Patch(facecolor='#aaaaaa', label='0 ignored'),\n"
        "    Patch(facecolor='#2d6a2d', label='1 healthy pasture'),\n"
        "    Patch(facecolor='#d4e157', label='2 stressed'),\n"
        "    Patch(facecolor='#c4a35a', label='3 bare'),\n"
        "    Patch(facecolor='#4fc3f7', label='4 water'),\n"
        "]\n"
        "axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8)\n"
        "\n"
        "plt.tight_layout()\n"
        "out = project_root / 'data' / 'eda_labels_waikato.png'\n"
        "plt.savefig(out, dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved to {out}')"
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "pasture-cv (Python 3.11)",
            "language": "python",
            "name": "pasture-cv",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {len(cells)} cells to {NB_PATH}")
