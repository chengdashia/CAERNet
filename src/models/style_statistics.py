from __future__ import annotations

import torch
from torch import nn


def resolve_layer_indices(indices: list[int], block_count: int) -> list[int]:
    resolved = [index if index >= 0 else block_count + index for index in indices]
    if any(index < 0 or index >= block_count for index in resolved):
        raise ValueError(
            f"Style layer indices {indices} are invalid for {block_count} blocks."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Style layer indices resolve to duplicate layers: {indices}.")
    return resolved


def patch_token_statistics(
    tokens: torch.Tensor,
    include_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.size(1) < 2:
        raise ValueError("Expected token tensor shaped [batch, cls+patches, channels].")
    patches = tokens[:, 1:]
    mean = patches.mean(dim=1)
    if not include_std:
        return mean
    variance = patches.var(dim=1, unbiased=False)
    std = torch.sqrt(variance + eps)
    return torch.cat((mean, std), dim=1)


class MultiLevelStyleStatistics(nn.Module):
    def __init__(
        self,
        token_width: int,
        output_dim: int,
        style_dim: int,
        layer_count: int,
        fusion_hidden_dim: int,
        dropout: float,
        include_std: bool = True,
        learned_fusion: bool = True,
    ):
        super().__init__()
        self.include_std = include_std
        self.learned_fusion = learned_fusion
        statistic_dim = token_width * (2 if include_std else 1)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(statistic_dim),
                    nn.Linear(statistic_dim, style_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(style_dim, output_dim),
                )
                for _ in range(layer_count)
            ]
        )
        self.layer_scorer = nn.Sequential(
            nn.Linear(output_dim, fusion_hidden_dim),
            nn.Tanh(),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        layer_tokens: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(layer_tokens) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} token layers, got {len(layer_tokens)}."
            )
        projected = torch.stack(
            [
                projection(
                    patch_token_statistics(tokens, include_std=self.include_std)
                )
                for projection, tokens in zip(self.projections, layer_tokens)
            ],
            dim=1,
        )
        if self.learned_fusion:
            layer_weights = torch.softmax(self.layer_scorer(projected).squeeze(-1), dim=1)
        else:
            layer_weights = projected.new_full(
                projected.shape[:2],
                1.0 / projected.size(1),
            )
        style_features = torch.sum(projected * layer_weights.unsqueeze(-1), dim=1)
        return nn.functional.normalize(style_features, dim=1), layer_weights
