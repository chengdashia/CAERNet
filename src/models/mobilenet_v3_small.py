from __future__ import annotations

from torch import nn
from torchvision import models

from .common import replace_classifier_last_head


def build_mobilenet_v3_small(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the MobileNetV3-Small baseline used for lightweight comparison."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    return replace_classifier_last_head(model, num_classes=num_classes, dropout=dropout)
