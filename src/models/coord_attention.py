import torch
from torch import nn


class CoordAttention(nn.Module):
    """Coordinate attention block for 2D feature maps."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.attn_h = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.attn_w = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        pooled = torch.cat([pooled_h, pooled_w], dim=2)
        encoded = self.shared(pooled)
        encoded_h, encoded_w = torch.split(encoded, [height, width], dim=2)
        encoded_w = encoded_w.permute(0, 1, 3, 2)

        attention_h = torch.sigmoid(self.attn_h(encoded_h))
        attention_w = torch.sigmoid(self.attn_w(encoded_w))
        return x * attention_h * attention_w
