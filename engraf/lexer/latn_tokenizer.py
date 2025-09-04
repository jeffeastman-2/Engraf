"""
LATN tokenizer facade that exposes Layer 1 entry points expected by tests.
"""
from .latn_tokenizer_layer1 import latn_tokenize, latn_tokenize_best
from .hypothesis import TokenizationHypothesis

__all__ = ["latn_tokenize", "latn_tokenize_best", "TokenizationHypothesis"]
