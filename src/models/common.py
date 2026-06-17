from __future__ import annotations

from torch import nn


def replace_fc_head(model: nn.Module, num_classes: int, dropout: float = 0.0) -> nn.Module:
    """Replace a torchvision model's `fc` classifier with the project head."""
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def replace_classifier_last_head(
    model: nn.Module,
    num_classes: int,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace the final layer of models that expose a `classifier` sequence."""
    last_layer = model.classifier[-1]
    in_features = last_layer.in_features
    layers = [
        child
        for child in list(model.classifier.children())[:-1]
        if not isinstance(child, nn.Dropout)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_features, num_classes))
    model.classifier = nn.Sequential(*layers)
    return model
