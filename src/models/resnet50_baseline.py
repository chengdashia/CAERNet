from __future__ import annotations

from torch import nn
from torchvision import models

from .common import replace_fc_head


def build_resnet50(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the ResNet50 baseline for ArtBench-10 comparisons."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    return replace_fc_head(model, num_classes=num_classes, dropout=dropout)
