#!/usr/bin/env python3
"""
LATN Tokenization Hypothesis

This module defines a unified hypothesis class for all layers of the 
LATN (Layer-Aware Tokenization Network) system.
"""

from typing import List, Optional, Any
from dataclasses import dataclass, field

# Import moved here to avoid circular imports
from engraf.lexer.vector_space import VectorSpace


@dataclass
class TokenizationHypothesis:
    """Unified tokenization hypothesis for all LATN layers.
    
    This provides a consistent interface across Layer 1, 2, 3, and 4.
    The replacements field is used only for debugging/analysis to track
    what transformations each layer performed.
    """
    tokens: List[VectorSpace]
    confidence: float
    description: str
    replacements: List[tuple] = field(default_factory=list)  # For debugging: (start_idx, end_idx, new_token)
    
    def __len__(self):
        """Return the number of tokens in this hypothesis."""
        return len(self.tokens)
    
    def __getitem__(self, index):
        """Allow indexing into the tokens list."""
        return self.tokens[index]
    
    def __iter__(self):
        """Allow iteration over the tokens."""
        return iter(self.tokens)
    
    def get_tokens_of_type(self, token_type: str) -> List[VectorSpace]:
        """Get all tokens of a specific type (e.g., 'NP', 'PP', 'VP')."""
        return [token for token in self.tokens if token.isa(token_type)]
    
    def has_token_type(self, token_type: str) -> bool:
        """Check if this hypothesis contains any tokens of the specified type."""
        return any(token.isa(token_type) for token in self.tokens)
    
    def token_words(self) -> List[str]:
        """Get the word strings for all tokens."""
        return [token.word for token in self.tokens]
    
    def print_tokens(self):
        """Print all tokens, each on a new line. Useful for demo examples."""
        for i, token in enumerate(self.tokens):
            print(f"  [{i}] {token}")
            
            # Show grounding information if available
            if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                grounded_phrase = token._grounded_phrase
                if hasattr(grounded_phrase, 'grounding') and grounded_phrase.grounding:
                    grounding_info = grounded_phrase.grounding
                    if 'scene_object' in grounding_info:
                        scene_obj = grounding_info['scene_object']
                        confidence = grounding_info.get('confidence', 'unknown')
                        print(f"      → GROUNDED to {scene_obj.object_id} (confidence: {confidence:.3f})")
                    else:
                        print(f"      → GROUNDED: {grounding_info}")
                else:
                    print(f"      → Ungrounded NP")
            elif token.isa("NP"):
                print(f"      → Ungrounded NP")
    
    def __repr__(self):
        """Standard representation showing tokens and confidence."""
        token_words = self.token_words()
        return f"TokenizationHypothesis(conf={self.confidence:.2f}, tokens={token_words}, desc='{self.description}')"
