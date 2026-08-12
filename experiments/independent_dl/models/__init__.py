"""Lazy model adapters for the independent deep-learning campaign."""

from .ft_transformer import FTTransformerAdapter
from .mlp_resnet import MLPResNetAdapter
from .tabm import TabMAdapter
from .tabr import TabRAdapter
from .tabicl_v2 import TabICLv2Result, fit_predict_tabicl_v2

__all__ = (
    "FTTransformerAdapter",
    "MLPResNetAdapter",
    "TabMAdapter",
    "TabRAdapter",
    "TabICLv2Result",
    "fit_predict_tabicl_v2",
)
