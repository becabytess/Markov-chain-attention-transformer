"""
Gravimem layer compatibility alias for SubQTransformer.
"""

from subqtransformer.layers import SubQSurfer as FusedPositionalJumpSurfer, SubQBlock as GravimemBlock

__all__ = ["FusedPositionalJumpSurfer", "GravimemBlock"]
