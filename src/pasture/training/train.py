"""
U-Net training loop with MLflow experiment tracking.

Usage:
    uv run python -m pasture.training.train [--epochs 30] [--encoder resnet50]

## Assumptions
- data/patches/train/ and data/patches/val/ must exist (run scripts/prepare_data.py first)
- GPU is used if available (CUDA); falls back to CPU
- MLflow logs to local mlruns/ directory (open UI with: mlflow ui)
- Best checkpoint saved to data/checkpoints/best.pt
- ignore_index=0 excluded from both CE loss and Dice loss
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pasture.data.dataset import PastureDataset
from pasture.models.factory import build_model, count_parameters
from pasture.training.metrics import compute_miou, compute_per_class_iou

try:
    from segmentation_models_pytorch.losses import DiceLoss
    _HAS_SMP_DICE = True
except ImportError:
    _HAS_SMP_DICE = False

LABEL_NAMES = {1: "healthy", 2: "stressed", 3: "bare", 4: "water"}

# Inverse-frequency class weights derived from 4-AOI training distribution:
#   healthy 59.4%  stressed 15.1%  bare 1.6%  water 6.7%
# Weights = median_freq / class_freq, capped at 15, normalised so healthy=1.0
CLASS_WEIGHTS = torch.tensor([0.0, 1.0, 4.0, 15.0, 8.0], dtype=torch.float32)


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if _HAS_SMP_DICE:
        criterion = DiceLoss(mode="multiclass", ignore_index=0, from_logits=True)
        return criterion(logits, targets)
    return torch.tensor(0.0, device=logits.device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ce_criterion: nn.CrossEntropyLoss,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)                                  # (B, C, H, W)
        loss = 0.5 * ce_criterion(logits, masks) + 0.5 * _dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_criterion: nn.CrossEntropyLoss,
    num_classes: int = 5,
) -> tuple[float, float, dict[int, float]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = 0.5 * ce_criterion(logits, masks) + 0.5 * _dice_loss(logits, masks)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    miou    = compute_miou(all_preds, all_targets, num_classes)
    per_cls = compute_per_class_iou(all_preds, all_targets, num_classes)
    return total_loss / len(loader), miou, per_cls


def train(
    arch: str = "unet",
    encoder: str | None = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    patch_dir: str = "data/patches",
    ckpt_dir: str = "data/checkpoints",
    num_classes: int = 5,
) -> dict:
    """Train one model config; returns result dict for ablation table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = PastureDataset(Path(patch_dir) / "train", augment=True)
    val_ds   = PastureDataset(Path(patch_dir) / "val",   augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Train patches: {len(train_ds)}  Val patches: {len(val_ds)}")

    model = build_model(arch=arch, encoder=encoder, in_channels=4, num_classes=num_classes).to(device)
    n_params = count_parameters(model)
    print(f"Model: {arch}/{encoder or 'default'}  Params: {n_params/1e6:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device), ignore_index=0
    )

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_miou  = 0.0
    best_epoch = 0
    best_ckpt  = Path(ckpt_dir) / f"best_{arch}.pt"

    mlflow.set_experiment("pasture-health-ablation")
    run_name = f"{arch}-{encoder or 'default'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "arch": arch, "encoder": encoder, "epochs": epochs,
            "batch_size": batch_size, "lr": lr, "num_classes": num_classes,
            "params_M": round(n_params / 1e6, 2),
            "class_weights": CLASS_WEIGHTS.tolist(),
            "train_patches": len(train_ds), "val_patches": len(val_ds),
        })

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, ce_criterion)
            val_loss, miou, per_cls = validate(model, val_loader, device, ce_criterion, num_classes)
            scheduler.step()

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_miou": miou,
                **{f"iou_{LABEL_NAMES[c]}": v for c, v in per_cls.items()},
            }
            mlflow.log_metrics(metrics, step=epoch)

            flag = ""
            if miou > best_miou:
                best_miou  = miou
                best_epoch = epoch
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "miou": miou, "arch": arch, "encoder": encoder}, best_ckpt)
                flag = "  ← best"

            print(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"mIoU={miou:.4f}{flag}"
            )

        mlflow.log_artifact(str(best_ckpt))
        print(f"\nBest mIoU: {best_miou:.4f}  checkpoint: {best_ckpt}")

    return {
        "arch": arch,
        "encoder": encoder,
        "params_M": round(n_params / 1e6, 2),
        "best_miou": round(best_miou, 4),
        "best_epoch": best_epoch if "best_epoch" in dir() else epochs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       default="unet")
    parser.add_argument("--encoder",    default=None)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patch-dir",  default="data/patches")
    parser.add_argument("--ckpt-dir",   default="data/checkpoints")
    args = parser.parse_args()
    train(
        arch=args.arch, encoder=args.encoder, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        patch_dir=args.patch_dir, ckpt_dir=args.ckpt_dir,
    )
