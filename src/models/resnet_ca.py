"""Compatibility imports for older code that referenced `models.resnet_ca`.

New model code should live in architecture-specific files and be registered in
`registry.py`. Keeping this wrapper avoids breaking existing scripts and tests.
"""

from .ca_resnet import ResNetWithCoordAttention, build_ca_resnet
from .registry import build_model

__all__ = ["ResNetWithCoordAttention", "build_ca_resnet", "build_model"]
