from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class DynamicChannelSpatialAttention(nn.Module):
    """Dynamic Channel-Spatial Attention Module (DCSAM).

    Paper formula (parallel additive fusion, SE-style channel + depthwise spatial):
        M_c = sigmoid( W2 · ReLU( W1 · GAP(F) ) )          (Eq. 5-6)
        M_s = sigmoid( DepthwiseConv3x3( Concat(MaxPool(F), AvgPool(F)) ) )  (Eq. 7-8)
        F_out = F ⊗ M_c  +  F ⊗ M_s                        (Eq. 9)
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 3):
        super().__init__()
        hidden = max(channels // reduction, 8)
        padding = spatial_kernel // 2

        # Channel attention MLP (paper Eq. 6: GAP only, SE-style)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Spatial attention (paper Eq. 7-8: 3×3 conv on max+avg pooled maps → 1ch)
        # Paper calls it "DepthwiseConv3×3" but output must be 1-channel spatial map;
        # equivalent to CBAM-style Conv2d(2, 1, k) that mixes the two pooled maps.
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=padding, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention: GAP -> MLP -> sigmoid (paper Eq. 5-6)
        mc = torch.sigmoid(self.mlp(self.avg_pool(x)))  # (N, C, 1, 1)

        # Spatial attention: max + avg along channel -> depthwise conv -> sigmoid
        max_map = x.amax(dim=1, keepdim=True)
        avg_map = x.mean(dim=1, keepdim=True)
        ms = torch.sigmoid(self.spatial_conv(torch.cat([max_map, avg_map], dim=1)))

        # Additive fusion (paper Eq. 9: F*Mc + F*Ms)
        return x * mc + x * ms


class MobileNetTransformerHybrid(nn.Module):
    """MTH + DCSAM + CSFT — reproduction of the lightweight paper model.

    Architecture (paper-aligned):
      1. MobileNetV3 shallow backbone  → 16×16 spatial features
      2. DCSAM on deep backbone (8×8, 288ch with reduced_tail)
      3. Transformer branch on shallow features (16×16, proj'd to embed_dim)
      4. CNN pool on deep attended features
      5. Fused embedding = [CNN_deep, Transformer_shallow] → classifier
      6. Embedding returned for contrastive (CSFT) loss
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        embed_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 1,
        dropout: float = 0.2,
        attention_reduction: int = 8,
        attention_kernel: int = 3,
        reduced_tail: bool = True,
    ):
        super().__init__()
        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT
            if pretrained and not reduced_tail
            else None
        )
        backbone = models.mobilenet_v3_small(weights=weights, reduced_tail=reduced_tail)

        # --- Split backbone ---
        # Shallow: layers 0-8 → 16×16, 48 channels (for Transformer branch)
        # Deep:   layers 9-12 → 8×8, 288/576 channels (for DCSAM + CNN head)
        self.shallow = backbone.features[:9]
        self.deep = backbone.features[9:]

        shallow_channels = 48
        deep_channels = backbone.features[-1].out_channels

        # --- Shallow → Transformer branch ---
        self.shallow_proj = nn.Conv2d(shallow_channels, embed_dim, kernel_size=1)

        # --- Deep → DCSAM → CNN branch ---
        self.attention = DynamicChannelSpatialAttention(
            deep_channels,
            reduction=attention_reduction,
            spatial_kernel=attention_kernel,
        )
        self.deep_proj = nn.Conv2d(deep_channels, embed_dim, kernel_size=1)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # --- Fusion head ---
        # Input: concat(CNN_deep(256), Transformer_shallow(256)) = 512
        # Pre-ReLU path (Linear + BN) → returned for contrastive loss so
        # embeddings span the full hypersphere, not just the positive orthant.
        self.embedding_pre_relu = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.embedding_post_relu = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # Shallow path (for Transformer)
        shallow = self.shallow(x)                  # (N, 48, 16, 16)
        projected = self.shallow_proj(shallow)     # (N, 256, 16, 16)

        # Transformer head on shallow features
        tokens = projected.flatten(2).transpose(1, 2)   # (N, 256, 256)
        transformer_feat = self.transformer(tokens).mean(dim=1)  # (N, 256)

        # Deep path (for CNN + DCSAM)
        deep = self.deep(shallow)                  # (N, deep_ch, 8, 8)
        deep = self.attention(deep)                # DCSAM
        deep = self.deep_proj(deep)                # (N, 256, 8, 8)

        # CNN head on deep features
        cnn_feat = self.pool(deep).flatten(1)      # (N, 256)

        # Fuse and classify
        concat = torch.cat([cnn_feat, transformer_feat], dim=1)
        pre_relu = self.embedding_pre_relu(concat)        # (N, 256) — for contrastive
        post_relu = self.embedding_post_relu(pre_relu)    # (N, 256) — for classifier
        logits = self.classifier(post_relu)
        return logits, pre_relu
