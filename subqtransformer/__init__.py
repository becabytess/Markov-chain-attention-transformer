"""
SubQTransformer: Sub-Quadratic Iterative Transformer with Adaptive Dynamical Halting.
"""

from subqtransformer.config import SubQConfig
from subqtransformer.layers import SubQSurfer, SubQBlock
from subqtransformer.model import SubQTransformerLM, SubQTransformerClassifier

__version__ = "1.0.0"
__all__ = [
    "SubQConfig",
    "SubQSurfer",
    "SubQBlock",
    "SubQTransformerLM",
    "SubQTransformerClassifier",
]
