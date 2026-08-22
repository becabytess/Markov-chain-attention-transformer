"""
Gravimem compatibility alias layer for SubQTransformer.
"""

from subqtransformer.config import SubQConfig
from subqtransformer.layers import SubQSurfer as FusedPositionalJumpSurfer, SubQBlock as GravimemBlock
from subqtransformer.model import SubQTransformerLM as GravimemLM, SubQTransformerClassifier as GravimemClassifier

__version__ = "1.0.0"
__all__ = [
    "SubQConfig",
    "FusedPositionalJumpSurfer",
    "GravimemBlock",
    "GravimemLM",
    "GravimemClassifier",
]
