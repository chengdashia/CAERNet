from __future__ import annotations

from torch import nn
from torchvision import models

from .common import replace_fc_head


def build_regnet_y_400mf(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the RegNetY-400MF baseline."""
    weights = models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
    model = models.regnet_y_400mf(weights=weights)
    return replace_fc_head(model, num_classes=num_classes, dropout=dropout)


def build_regnet_y_3_2gf(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the stronger RegNetY-3.2GF baseline."""
    weights = models.RegNet_Y_3_2GF_Weights.DEFAULT if pretrained else None
    model = models.regnet_y_3_2gf(weights=weights)
    return replace_fc_head(model, num_classes=num_classes, dropout=dropout)
