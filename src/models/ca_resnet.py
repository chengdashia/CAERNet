from __future__ import annotations

from torch import nn
from torchvision import models

from .coord_attention import CoordAttention


class ResNetWithCoordAttention(nn.Module):
    """Single-scale CAERNet: ResNet features followed by Coordinate Attention."""

    _SPECS = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    }

    def __init__(
        self,
        architecture: str,
        num_classes: int,
        pretrained: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        if architecture not in self._SPECS:
            supported = ", ".join(sorted(self._SPECS))
            raise ValueError(
                "Coordinate attention is implemented only for "
                f"{supported}; got {architecture}."
            )

        builder, default_weights, channels = self._SPECS[architecture]
        weights = default_weights if pretrained else None
        backbone = builder(weights=weights)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.attention = CoordAttention(channels=channels)
        self.pool = backbone.avgpool
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone.fc.in_features, num_classes),
            )
        else:
            self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_ca_resnet(
    architecture: str,
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> ResNetWithCoordAttention:
    """Build a ResNet backbone with a final Coordinate Attention block."""
    return ResNetWithCoordAttention(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
