"""
Gravimem model compatibility alias for SubQTransformer.
"""

from subqtransformer.model import SubQTransformerLM as GravimemLM, SubQTransformerClassifier as GravimemClassifier

__all__ = ["GravimemLM", "GravimemClassifier"]
