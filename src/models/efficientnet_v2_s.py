from __future__ import annotations

from torch import nn
from torchvision import models

from .common import replace_classifier_last_head


def build_efficientnet_v2_s(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the EfficientNetV2-S baseline with a project-specific classifier."""
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    return replace_classifier_last_head(model, num_classes=num_classes, dropout=dropout)
