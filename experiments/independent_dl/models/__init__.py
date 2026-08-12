"""Lazy model adapters for the independent deep-learning campaign."""

from .ft_transformer import FTTransformerAdapter
from .mlp_resnet import MLPResNetAdapter
from .tabm import TabMAdapter

__all__ = ("FTTransformerAdapter", "MLPResNetAdapter", "TabMAdapter")
