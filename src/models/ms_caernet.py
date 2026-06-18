from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .coord_attention import CoordAttention


class ContextAwareScaleGate(nn.Module):
    """Cross-scale context-aware gating for multi-scale feature fusion.

    Unlike a naive per-scale MLP gate that scores each scale independently,
    CASG first aggregates global descriptors from **all** scales into a joint
    context vector, then produces attention weights conditioned on the full
    multi-scale picture.  This allows the fusion to model inter-scale
    complementarity — for example, fine texture cues at C3 can reinforce
    semantic layout information at C5.

    Reference concept: non-local attention (Wang et al., 2018) adapted to
    discrete scale selection.
    """

    def __init__(self, embed_dim: int, num_scales: int = 3):
        super().__init__()
        hidden = max(embed_dim // 2, 64)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_scales, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_scales),
        )

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted fusion of stacked multi-scale embeddings.

        Args:
            stacked: ``(B, num_scales, embed_dim)`` tensor of per-scale
                     embeddings.

        Returns:
            ``(B, embed_dim)`` fused embedding.
        """
        # Cross-scale context: each scale's attention score is conditioned on
        # the individual per-scale embeddings (not a collapsed mean), so the
        # gate can model real inter-scale complementarity.
        context = torch.cat(
            [stacked[:, i] for i in range(stacked.size(1))], dim=-1,
        )
        scores = self.gate(context)
        weights = torch.softmax(scores, dim=1)  # (B, num_scales)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)


class MultiScaleCAERNet(nn.Module):
    """Multi-scale coordinate-attention ResNet for art style classification.

    Architecture overview:
        1. **Multi-scale feature extraction** — ResNet50 stages C3/C4/C5.
        2. **Per-scale Coordinate Attention** — decoupled H/W spatial
           recalibration at each receptive-field level.
        3. **Shared-dimension projection** — per-scale MLP projectors map
           heterogeneous channel counts into a common embedding space.
        4. **Context-Aware Scale Gate (CASG)** — cross-scale context-aware
           gating that models inter-scale complementarity before fusion.
        5. **Joint classification + regularization** — CE loss with optional
           energy regularization and supervised contrastive loss on the fused
           embedding.

    Returns ``(logits, embedding)`` so the training loop can apply supervised
    contrastive loss on the fused embedding when configured.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        embed_dim: int = 512,
        dropout: float = 0.15,
        attention_reduction: int = 8,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ResNet50 stages C3/C4/C5 provide progressively richer receptive
        # fields. We attend, pool, and project each stage before fusion.
        self.num_scales = 3
        channels = (512, 1024, 2048)
        self.attentions = nn.ModuleList(
            CoordAttention(channel, reduction=attention_reduction)
            for channel in channels
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projectors = nn.ModuleList(
            self._make_projector(channel, embed_dim, dropout)
            for channel in channels
        )

        # Context-Aware Scale Gate (CASG): cross-scale attention over fused
        # embeddings — each scale's weight is conditioned on all scales.
        self.scale_gate = ContextAwareScaleGate(embed_dim, self.num_scales)
        self.classifier = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def _make_projector(channels: int, embed_dim: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(channels, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        feature2 = self.layer2(x)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)

        # Convert each attended feature map into an embedding with the same
        # dimensionality so the three scales can be fused cleanly.
        embeddings = []
        for feature, attention, projector in zip(
            (feature2, feature3, feature4),
            self.attentions,
            self.projectors,
        ):
            attended = attention(feature)
            pooled = self.pool(attended).flatten(1)
            embeddings.append(projector(pooled))

        stacked = torch.stack(embeddings, dim=1)  # (B, 3, embed_dim)

        # CASG: cross-scale context-aware gating
        embedding = self.scale_gate(stacked)  # (B, embed_dim)

        # Training expects `(logits, embedding)` when contrastive loss is on.
        logits = self.classifier(embedding)
        return logits, embedding
