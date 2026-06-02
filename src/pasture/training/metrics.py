"""Segmentation metrics for pasture health detection."""

from __future__ import annotations

import torch


def compute_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5,
    ignore_index: int = 0,
) -> float:
    """Mean IoU over classes 1..(num_classes-1), ignoring ignore_index.

    Args:
        preds:   (B, H, W) long — argmax predictions.
        targets: (B, H, W) long — ground-truth labels.

    Returns:
        Scalar mean IoU across foreground classes with at least one GT pixel.
    """
    valid = targets != ignore_index
    preds   = preds[valid]
    targets = targets[valid]

    ious = []
    for cls in range(1, num_classes):
        pred_c = preds == cls
        true_c = targets == cls
        intersection = (pred_c & true_c).sum().item()
        union        = (pred_c | true_c).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if ious else 0.0


def compute_per_class_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5,
    ignore_index: int = 0,
) -> dict[int, float]:
    """Per-class IoU dict (keys 1..num_classes-1)."""
    valid   = targets != ignore_index
    preds   = preds[valid]
    targets = targets[valid]

    result = {}
    for cls in range(1, num_classes):
        pred_c = preds == cls
        true_c = targets == cls
        inter  = (pred_c & true_c).sum().item()
        union  = (pred_c | true_c).sum().item()
        result[cls] = inter / union if union > 0 else float("nan")
    return result
