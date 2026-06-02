"""
U-Net segmentation model via segmentation-models-pytorch.

## Assumptions
- in_channels=4: B02/B03/B04/B08; encoder adapted from 3-ch ImageNet weights
  (smp copies mean of first 3 ch weights into the 4th channel automatically)
- num_classes=5: 0=ignored, 1=healthy, 2=stressed, 3=bare, 4=water
- Encoder choices: resnet50 (baseline), efficientnet-b3 (target), mit-b0 (stretch)
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

ENCODERS = {
    "resnet50":       "imagenet",
    "efficientnet-b3": "imagenet",
    "mit-b0":         "imagenet",
}


def build_unet(
    encoder: str = "resnet50",
    in_channels: int = 4,
    num_classes: int = 5,
) -> nn.Module:
    """Return a U-Net with a pretrained encoder.

    The first conv layer is adapted for in_channels=4 automatically by smp.
    """
    weights = ENCODERS.get(encoder, "imagenet")
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits — loss handles softmax
    )
    return model
