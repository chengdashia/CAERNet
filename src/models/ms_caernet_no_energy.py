"""MS-CAERNet builder for the no-energy ablation.

This ablation uses the same network as the full model. Its difference is the
loss setting `energy_lambda: 0.0` in the matching YAML config.
"""

from .ms_caernet import MultiScaleCAERNet


def build_ms_caernet_no_energy(
    num_classes: int,
    pretrained: bool = False,
    embed_dim: int = 512,
    dropout: float = 0.15,
    attention_reduction: int = 8,
) -> MultiScaleCAERNet:
    """Build MS-CAERNet for the no-energy ablation run."""
    return MultiScaleCAERNet(
        num_classes=num_classes,
        pretrained=pretrained,
        embed_dim=embed_dim,
        dropout=dropout,
        attention_reduction=attention_reduction,
    )
