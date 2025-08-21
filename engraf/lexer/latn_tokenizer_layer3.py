#!/usr/bin/env python3
"""
LATN Layer 3: Prepositional Phrase Token Replacement

This module implements Layer 3 of the LATN (Layer-Aware Tokenization Network) system.
Layer 3 replaces prepositional phrase constructions with single PrepositionalPhrase tokens.

Layer 3 builds on:
- Layer 1: Multi-hypothesis tokenization with morphological inflection
- Layer 2: NounPhrase token replacement
- Layer 3: PrepositionalPhrase token replacement

This layer identifies prepositional phrases like "at [1,2,3]", "on the table", "above the red sphere"
and replaces them with single PrepositionalPhrase tokens containing the semantic meaning.
"""

from typing import List, Optional
from dataclasses import dataclass
import copy

from engraf.lexer.latn_tokenizer import TokenizationHypothesis  
from engraf.lexer.latn_tokenizer_layer2 import NPTokenizationHypothesis
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet_pp import run_pp
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.utils.predicates import is_preposition


@dataclass
class PPTokenizationHypothesis:
    """Extended tokenization hypothesis that includes PP token replacement."""
    tokens: List[VectorSpace]
    confidence: float
    description: str
    pp_replacements: List[tuple] = None  # List of (start_idx, end_idx, pp_token) tuples
    
    def __post_init__(self):
        if self.pp_replacements is None:
            self.pp_replacements = []


def create_pp_token(pp: PrepositionalPhrase) -> VectorSpace:
    """Create a PrepositionalPhrase token from a parsed PrepositionalPhrase object.
    
    This creates a single token that represents the entire prepositional phrase,
    similar to how we create NounPhrase tokens in Layer 2.
    """
    # Create a new token with the PP's semantic vector
    pp_token = VectorSpace()
    
    # Copy the semantic content from the PrepositionalPhrase
    if pp.vector:
        from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
        for dim in VECTOR_DIMENSIONS:
            value = pp.vector[dim]
            if value != 0.0:  # Only copy non-zero values
                pp_token[dim] = value
    
    # Mark this as a prepositional phrase token
    pp_token["PP"] = 1.0
    
    # Set the word to a descriptive representation
    pp_token.word = f"PP({pp.preposition})"
    if pp.noun_phrase:
        pp_token.word = f"PP({pp.preposition} {pp.noun_phrase.get_original_text() if hasattr(pp.noun_phrase, 'get_original_text') else str(pp.noun_phrase)})"
    
    # Store the original PrepositionalPhrase object for later access
    pp_token._original_pp = pp
    
    return pp_token


def find_pp_sequences(tokens: List[VectorSpace]) -> List[tuple]:
    """Find sequences in tokens that can be parsed as prepositional phrases.
    
    Uses greedy left-to-right parsing like Layer 2: try PP at each position, if successful
    consume those tokens and continue from the next position.
    
    This works with tokens that may already contain NounPhrase tokens from Layer 2.
    The PP ATN will handle NounPhrase tokens directly.
    
    Returns:
        List of (start_idx, end_idx, pp_object) tuples for successful PP parses
    """
    pp_sequences = []
    i = 0
    
    while i < len(tokens):
        # Must start with a preposition
        if not tokens[i].isa("prep"):
            i += 1
            continue
            
        # Try to parse PP starting at position i
        # Use remaining tokens for parsing
        subsequence = tokens[i:]
        best_pp = None
        best_end = i
        
        # Try different ending positions (PPs are typically 2-4 tokens)
        for length in range(2, min(5, len(subsequence) + 1)):
            subseq = subsequence[:length]
            
            try:
                # Try to parse as PP using the enhanced PP ATN
                pp = run_pp(subseq)
                if pp and pp.preposition and pp.noun_phrase:
                    best_pp = pp
                    best_end = i + length - 1
                    # Continue trying longer sequences (greedy - take longest match)
            except Exception:
                # Failed to parse as PP, continue trying
                continue
        
        if best_pp is not None:
            # Found a PP, add it and skip past it
            pp_sequences.append((i, best_end + 1, best_pp))  # end_idx is exclusive
            i = best_end + 1
        else:
            # No PP found starting at position i, move to next position
            i += 1
    
    return pp_sequences


def replace_pp_sequences(tokens: List[VectorSpace], pp_sequences: List[tuple]) -> List[VectorSpace]:
    """Replace PP sequences with PrepositionalPhrase tokens.
    
    Args:
        tokens: Original token list
        pp_sequences: List of (start_idx, end_idx, pp_object) tuples
        
    Returns:
        New token list with PP sequences replaced by single PP tokens
    """
    if not pp_sequences:
        return tokens
    
    # Sort by start position (reverse order for safe replacement)
    pp_sequences.sort(key=lambda x: x[0], reverse=True)
    
    # Create a copy of tokens to modify
    new_tokens = tokens[:]
    
    # Replace each PP sequence with a single PP token
    for start_idx, end_idx, pp in pp_sequences:
        pp_token = create_pp_token(pp)
        # Replace the sequence with the single PP token
        new_tokens[start_idx:end_idx] = [pp_token]
    
    return new_tokens


def latn_tokenize_layer3(layer2_hypotheses: List[NPTokenizationHypothesis]) -> List[PPTokenizationHypothesis]:
    """LATN Layer 3: Generate tokenization hypotheses with PP token replacement.
    
    This function:
    1. Takes Layer 2 hypotheses (with NP tokens)
    2. For each hypothesis, finds prepositional phrases using the enhanced PP ATN
    3. Replaces PP sequences with single PrepositionalPhrase tokens
    4. Returns hypotheses ranked by confidence
    
    Args:
        layer2_hypotheses: List of NPTokenizationHypothesis from Layer 2
        
    Returns:
        List of PPTokenizationHypothesis objects, ranked by confidence
    """
    layer3_hypotheses = []
    
    for base_hyp in layer2_hypotheses:
        # Find PP sequences in this tokenization using the enhanced PP ATN
        pp_sequences = find_pp_sequences(base_hyp.tokens)
        
        if pp_sequences:
            # Create hypothesis with PP replacements
            new_tokens = replace_pp_sequences(base_hyp.tokens, pp_sequences)
            
            # Calculate confidence (slight boost for successful PP identification)
            new_confidence = base_hyp.confidence * 1.05
            
            # Create description
            pp_count = len(pp_sequences)
            description = f"{base_hyp.description} + {pp_count} PP token(s)"
            
            layer3_hyp = PPTokenizationHypothesis(
                tokens=new_tokens,
                confidence=new_confidence,
                description=description,
                pp_replacements=[(start, end, pp) for start, end, pp in pp_sequences]
            )
            layer3_hypotheses.append(layer3_hyp)
        else:
            # No PP replacements, but still include as Layer 3 hypothesis
            layer3_hyp = PPTokenizationHypothesis(
                tokens=base_hyp.tokens,
                confidence=base_hyp.confidence,
                description=base_hyp.description,
                pp_replacements=[]
            )
            layer3_hypotheses.append(layer3_hyp)
    
    # Sort by confidence (highest first)
    layer3_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    return layer3_hypotheses
