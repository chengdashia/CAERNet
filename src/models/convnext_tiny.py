from __future__ import annotations

from torch import nn
from torchvision import models

from .common import replace_classifier_last_head


def build_convnext_tiny(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the ConvNeXt-Tiny baseline with a project-specific classifier."""
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    return replace_classifier_last_head(model, num_classes=num_classes, dropout=dropout)
