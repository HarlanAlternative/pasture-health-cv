"""
Patch extraction from Sentinel-2 tiles and LCDB masks.

## Assumptions
- image: np.ndarray (4, H, W) float32 — band order B02/B03/B04/B08
- mask:  np.ndarray (H, W)    uint8  — labels 0-4, 255=nodata
- 255 (nodata) is remapped to 0 (ignored) before saving patches
- Patches with < min_label_frac valid labeled pixels (label > 0) are discarded
- Patches saved as .npz with keys 'image' and 'mask'
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    min_label_frac: float = 0.20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slide a window over (image, mask) and return valid patches.

    Args:
        image: (4, H, W) float32 Sentinel-2 tile.
        mask:  (H, W) uint8 LCDB label mask.
        patch_size: square patch side length in pixels.
        stride: sliding window stride (stride < patch_size → overlap).
        min_label_frac: discard patches where labeled pixels / total < this.

    Returns:
        List of (image_patch, mask_patch) tuples. image_patch shape: (4, P, P).
    """
    _, H, W = image.shape
    clean_mask = np.where(mask == 255, 0, mask)  # remap nodata → ignored

    patches = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            img_p = image[:, r : r + patch_size, c : c + patch_size]
            msk_p = clean_mask[r : r + patch_size, c : c + patch_size]

            labeled_frac = (msk_p > 0).mean()
            if labeled_frac >= min_label_frac:
                patches.append((img_p.copy(), msk_p.copy()))

    return patches


def save_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Save each patch as a compressed .npz file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (img, msk) in enumerate(patches):
        p = out_dir / f"{prefix}_{i:04d}.npz"
        np.savez_compressed(p, image=img, mask=msk)
        paths.append(p)
    return paths
