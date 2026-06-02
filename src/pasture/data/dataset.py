"""
PyTorch Dataset for Sentinel-2 pasture segmentation patches.

## Assumptions
- Patches are .npz files with keys 'image' (4,H,W) float32 and 'mask' (H,W) uint8
- Mask values: 0=ignored, 1=healthy, 2=stressed, 3=bare, 4=water
- Normalisation: per-band z-score using Sentinel-2 L2A statistics typical for NZ
- Augmentations are spatial-only (flip, rotate) — safe for all 4 bands
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

# Per-band mean/std estimated from NZ S2 L2A summer imagery (B02,B03,B04,B08)
BAND_MEAN = np.array([0.033, 0.062, 0.043, 0.370], dtype=np.float32)
BAND_STD  = np.array([0.030, 0.040, 0.040, 0.150], dtype=np.float32)

_TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.3),
])


def _normalise(image: np.ndarray) -> np.ndarray:
    """Z-score normalise (4, H, W) float32 per band."""
    return (image - BAND_MEAN[:, None, None]) / (BAND_STD[:, None, None] + 1e-8)


class PastureDataset(Dataset):
    """Dataset loading pre-extracted patch .npz files.

    Args:
        patch_dir: directory containing .npz patch files.
        augment: apply spatial augmentations (train only).
    """

    def __init__(self, patch_dir: str | Path, augment: bool = False):
        self.patches = sorted(Path(patch_dir).glob("*.npz"))
        if not self.patches:
            raise FileNotFoundError(f"No .npz patches found in {patch_dir}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.patches[idx])
        image = data["image"].astype(np.float32)  # (4, H, W)
        mask  = data["mask"].astype(np.int64)     # (H, W)

        if self.augment:
            # albumentations expects (H, W, C) for image
            img_hwc = np.transpose(image, (1, 2, 0))
            result = _TRAIN_AUG(image=img_hwc, mask=mask)
            image = np.transpose(result["image"], (2, 0, 1))
            mask  = result["mask"]

        image = _normalise(image)
        return torch.from_numpy(image), torch.from_numpy(mask).long()
