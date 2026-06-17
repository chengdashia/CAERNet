"""Public model factory used by training and evaluation scripts."""

from .registry import build_model

__all__ = ["build_model"]
