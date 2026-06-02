"""
Unified model factory for pasture segmentation architectures.

Supported:
    unet          — U-Net + any smp encoder (baseline)
    deeplabv3plus — DeepLabV3+ + any smp encoder (target)
    segformer     — SegFormer with MiT encoder (stretch)

## Assumptions
- in_channels=4: B02/B03/B04/B08; smp adapts pretrained 3-ch weights automatically
- num_classes=5: 0=ignored 1=healthy 2=stressed 3=bare 4=water
- activation=None: raw logits returned; softmax handled by loss functions
"""

from __future__ import annotations

import torch.nn as nn
import segmentation_models_pytorch as smp


ARCH_DEFAULTS: dict[str, str] = {
    "unet":          "resnet50",
    "deeplabv3plus": "efficientnet-b3",
    "segformer":     "mit_b0",
}


def build_model(
    arch: str = "unet",
    encoder: str | None = None,
    in_channels: int = 4,
    num_classes: int = 5,
) -> nn.Module:
    """Return a segmentation model ready for training.

    Args:
        arch: one of 'unet', 'deeplabv3plus', 'segformer'.
        encoder: timm/smp encoder name; defaults to ARCH_DEFAULTS[arch] if None.
        in_channels: number of input spectral bands.
        num_classes: number of output segmentation classes.
    """
    encoder = encoder or ARCH_DEFAULTS.get(arch, "resnet50")
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    if arch == "unet":
        return smp.Unet(**kwargs)
    elif arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**kwargs)
    elif arch == "segformer":
        return smp.Segformer(**kwargs)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCH_DEFAULTS)}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
