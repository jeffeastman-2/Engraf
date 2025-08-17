"""
LATN (Layered Augmented Transition Network) Lexical Layer

This module implements multi-hypothesis tokenization that returns ranked alternatives
instead of committing early to a single tokenization.
"""

from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vocabulary_builder import vector_from_word
from engraf.lexer.vector_space import VectorSpace, vector_from_features
from engraf.utils.noun_inflector import singularize_noun
from engraf.utils.verb_inflector import find_root_verb
import re
from typing import List, Tuple


class TokenizationHypothesis:
    """Represents a single tokenization hypothesis with confidence score."""
    
    def __init__(self, tokens: List[VectorSpace], confidence: float, description: str = ""):
        self.tokens = tokens
        self.confidence = confidence
        self.description = description
    
    def __repr__(self):
        token_words = [t.word for t in self.tokens]
        return f"TokenizationHypothesis(conf={self.confidence:.2f}, tokens={token_words}, desc='{self.description}')"


def latn_tokenize(sentence: str) -> List[TokenizationHypothesis]:
    """
    LATN lexical layer: Return multiple ranked tokenization hypotheses.
    
    Args:
        sentence: Input sentence string
        
    Returns:
        List of TokenizationHypothesis objects, ranked by confidence (highest first)
    """
    # First, extract raw tokens using the same regex as original
    pattern = re.compile(
        r"""\s*(
            \[             # opening bracket
            \s*-?\d+(\.\d+)?\s*,     # x
            \s*-?\d+(\.\d+)?\s*,     # y
            \s*-?\d+(\.\d+)?\s*      # z
            \]             # closing bracket
            | -?\d+(?:\.\d+)?         # standalone number (int or float)
            | '[\w\s]+'     # quoted words (single quotes)
            | \w+           # normal word
            | [^\w\s]       # punctuation
        )""",
        re.VERBOSE,
    )
    tokens = pattern.findall(sentence)
    flat_tokens = [t[0] for t in tokens]
    
    # Generate all possible tokenization hypotheses
    hypotheses = []
    
    # Generate hypotheses recursively
    def generate_hypotheses(token_index: int, current_tokens: List[VectorSpace], current_confidence: float, description: str):
        """Recursively generate all possible tokenization hypotheses."""
        
        if token_index >= len(flat_tokens):
            # Reached end - add this hypothesis
            hypotheses.append(TokenizationHypothesis(
                tokens=current_tokens.copy(), 
                confidence=current_confidence,
                description=description
            ))
            return
        
        # Try different grouping options at current position
        remaining_tokens = len(flat_tokens) - token_index
        max_group_size = min(3, remaining_tokens)  # Try up to 3-word compounds
        
        for group_size in range(1, max_group_size + 1):
            if token_index + group_size > len(flat_tokens):
                continue
                
            # Extract the token group
            if group_size == 1:
                tok = flat_tokens[token_index]
                compound_key = tok.lower()
            else:
                tok_group = flat_tokens[token_index:token_index + group_size]
                compound_key = " ".join(tok_group).lower()
                tok = compound_key
            
            # Try to process this token/compound
            processed_token, confidence_boost, process_description = process_token_group(tok, group_size)
            
            if processed_token is not None:
                # Calculate confidence for this choice
                # Use average confidence rather than sum to avoid bias toward more tokens
                decision_confidence = current_confidence * len(current_tokens) + confidence_boost
                new_confidence = decision_confidence / (len(current_tokens) + 1)
                new_description = f"{description} | {process_description}" if description else process_description
                
                # Recursively generate the rest
                generate_hypotheses(
                    token_index + group_size,
                    current_tokens + [processed_token],
                    new_confidence,
                    new_description
                )
    
    # Start recursive generation
    generate_hypotheses(0, [], 0.0, "")
    
    # Sort by confidence (descending) and return
    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    return hypotheses


def process_token_group(tok: str, group_size: int) -> Tuple[VectorSpace, float, str]:
    """
    Process a token or token group, returning the VectorSpace, confidence boost, and description.
    
    Args:
        tok: Token string (single word or multi-word compound)
        group_size: Number of original tokens this represents
        
    Returns:
        (VectorSpace object, confidence_boost, description) or (None, 0, "") if invalid
    """
    
    # Handle special tokens first (vectors, quoted strings, numbers)
    if tok.startswith("'") and tok.endswith("'"):
        inner_tok = tok[1:-1]
        vs = vector_from_features(pos="quoted")
        vs.word = inner_tok
        return vs, 0.8, f"quoted-string({inner_tok})"
    
    elif tok.startswith("[") and tok.endswith("]"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", tok)
        if len(nums) == 3:
            x, y, z = map(float, nums)
            vs = vector_from_features(loc=[x, y, z], pos="vector")
            vs.word = tok
            return vs, 1.0, f"vector({x},{y},{z})"
        return None, 0, ""
    
    elif re.fullmatch(r"-?\d+(?:\.\d+)?", tok):
        vs = vector_from_features(pos="det def number")
        vs.word = tok
        vs["number"] = float(tok)
        return vs, 0.9, f"number({tok})"
    
    # Handle word tokens
    else:
        # First check direct vocabulary lookup
        if tok.lower() in SEMANTIC_VECTOR_SPACE:
            vs = vector_from_word(tok.lower())
            
            # Handle noun singularization
            if vs["noun"] > 0:
                singular_form, was_plural = singularize_noun(tok)
                vs.word = singular_form
            else:
                vs.word = tok.lower()
            
            # Calculate confidence based on compound length
            # Longer compounds get higher confidence when they exist
            if group_size == 1:
                confidence = 0.7  # Base confidence for single words
                description = f"single-word({tok})"
            elif group_size == 2:
                confidence = 1.0  # Higher confidence for two-word compounds
                description = f"two-word-compound({tok})"
            elif group_size == 3:
                confidence = 1.2  # Highest confidence for three-word compounds
                description = f"three-word-compound({tok})"
            else:
                confidence = 0.5
                description = f"{group_size}-word-compound({tok})"
            
            return vs, confidence, description
        
        # Try verb inflection if single word and not in vocabulary
        elif group_size == 1:
            root_verb, inflection_type, found_root = find_root_verb(tok)
            if found_root:
                vs = vector_from_word(root_verb)
                vs.word = tok.lower()
                if inflection_type:
                    vs[inflection_type] = 1.0
                return vs, 0.6, f"inflected-verb({tok}→{root_verb})"
        
        # Token not found in vocabulary
        return None, 0, ""


def latn_tokenize_best(sentence: str) -> List[VectorSpace]:
    """
    Convenience function that returns just the best hypothesis tokens.
    For compatibility with existing code.
    """
    hypotheses = latn_tokenize(sentence)
    if hypotheses:
        return hypotheses[0].tokens
    else:
        # Fallback to original tokenize if no hypotheses generated
        from engraf.lexer.token_stream import tokenize
        return tokenize(sentence)


if __name__ == "__main__":
    # Test the LATN tokenizer
    test_sentences = [
        "draw a light house at [0,0,0]",
        "draw a box at [1,2,3]",
        "draw a very light house"
    ]
    
    for sentence in test_sentences:
        print(f"\nInput: '{sentence}'")
        hypotheses = latn_tokenize(sentence)
        print(f"Generated {len(hypotheses)} hypotheses:")
        for i, hyp in enumerate(hypotheses[:3], 1):  # Show top 3
            print(f"  {i}. {hyp}")
