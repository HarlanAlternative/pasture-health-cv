"""Evaluate all 3 checkpoints on the validation set and print per-class IoU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from pasture.data.dataset import PastureDataset
from pasture.models.factory import build_model
from pasture.training.metrics import compute_per_class_iou

LABEL = {1: "healthy", 2: "stressed", 3: "bare", 4: "water"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(arch: str) -> tuple[str, float, dict]:
    ckpt_path = Path(f"data/checkpoints/best_{arch}.pt")
    if not ckpt_path.exists():
        return arch, 0.0, {}
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(arch=ckpt.get("arch", arch), encoder=ckpt.get("encoder"),
                        in_channels=4, num_classes=5)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    loader = DataLoader(PastureDataset("data/patches/val", augment=False),
                        batch_size=16, shuffle=False, num_workers=0)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            all_preds.append(model(imgs.to(DEVICE)).argmax(1).cpu())
            all_targets.append(masks)

    per_cls = compute_per_class_iou(torch.cat(all_preds), torch.cat(all_targets))
    label = f"{ckpt.get('arch', arch)} / {ckpt.get('encoder', '')}"
    return label, ckpt["miou"], per_cls


def main():
    rows = [evaluate(a) for a in ["unet", "deeplabv3plus", "segformer"]]
    hdr = f"{'Model':<32} {'mIoU':>7}  {'healthy':>9}  {'stressed':>9}  {'bare':>6}  {'water':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, miou, per_cls in rows:
        cls_cols = "  ".join(
            f"{per_cls.get(c, float('nan')):9.4f}" for c in [1, 2, 3, 4]
        )
        print(f"{label:<32} {miou:>7.4f}  {cls_cols}")
    print()

if __name__ == "__main__":
    main()
