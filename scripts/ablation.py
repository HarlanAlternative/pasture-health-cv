"""
Ablation study: train all 3 architectures with identical hyperparameters.

Usage:
    uv run python scripts/ablation.py [--epochs 30]

Results saved to data/ablation_results.json and printed as a table.

Configs:
    1. U-Net       / ResNet50        (baseline)
    2. DeepLabV3+  / EfficientNet-B3 (target)
    3. SegFormer   / MiT-B0          (stretch)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pasture.training.train import train

CONFIGS = [
    {"arch": "unet",          "encoder": "resnet50",       "label": "U-Net / ResNet-50"},
    {"arch": "deeplabv3plus", "encoder": "efficientnet-b3","label": "DeepLabV3+ / EfficientNet-B3"},
    {"arch": "segformer",     "encoder": "mit_b0",           "label": "SegFormer-B0"},
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Model':<35} {'Params':>8} {'Best mIoU':>10} {'Best Epoch':>11}")
    print("-" * 72)
    for r in results:
        label = r.get("label", f"{r['arch']}/{r['encoder']}")
        print(f"{label:<35} {r['params_M']:>6.1f}M {r['best_miou']:>10.4f} {r['best_epoch']:>11}")
    print("=" * 72)
    best = max(results, key=lambda x: x["best_miou"])
    print(f"Best: {best.get('label', best['arch'])}  mIoU={best['best_miou']:.4f}\n")


def main(epochs: int = 30, batch_size: int = 16, lr: float = 1e-4) -> None:
    results = []
    out_path = PROJECT_ROOT / "data" / "ablation_results.json"

    for cfg in CONFIGS:
        print(f"\n{'='*72}")
        print(f"Training: {cfg['label']}")
        print(f"{'='*72}")

        result = train(
            arch=cfg["arch"],
            encoder=cfg["encoder"],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patch_dir=str(PROJECT_ROOT / "data" / "patches"),
            ckpt_dir=str(PROJECT_ROOT / "data" / "checkpoints"),
        )
        result["label"] = cfg["label"]
        results.append(result)

        # Save incrementally so partial results survive interrupts
        out_path.write_text(json.dumps(results, indent=2))

    print_table(results)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
