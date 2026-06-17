"""Full MS-CAERNet experiment builder.

The network implementation is shared by all MS-CAERNet ablations. The "full"
experiment enables both energy regularization and contrastive loss in its YAML
config, while this file keeps the architecture name explicit in code.
"""

from .ms_caernet import MultiScaleCAERNet


def build_ms_caernet_full(
    num_classes: int,
    pretrained: bool = False,
    embed_dim: int = 512,
    dropout: float = 0.15,
    attention_reduction: int = 8,
) -> MultiScaleCAERNet:
    """Build the shared MS-CAERNet architecture for the full experiment."""
    return MultiScaleCAERNet(
        num_classes=num_classes,
        pretrained=pretrained,
        embed_dim=embed_dim,
        dropout=dropout,
        attention_reduction=attention_reduction,
    )
