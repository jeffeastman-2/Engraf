#!/usr/bin/env python3
"""
LATN Layer 2: NounPhrase Token Replacement

This module implements Layer 2 of the LATN (Layer-Aware Tokenization Network) system.
Layer 2 replaces noun phrase constructions with single NounPhrase tokens.

Layer 2 builds on:
- Layer 1: Multi-hypothesis tokenization with morphological inflection

This layer identifies noun phrases like "the red box", "a very large sphere", "[1,2,3]"
and replaces them with single NounPhrase tokens containing the semantic meaning.
"""

from typing import List, Optional
from dataclasses import dataclass
import copy

from engraf.lexer.latn_tokenizer import TokenizationHypothesis  
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet_np import run_np
from engraf.atn.np import build_np_atn
from engraf.atn.core import run_atn
from engraf.pos.noun_phrase import NounPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.utils.predicates import is_determiner, is_adjective, is_noun, is_vector


@dataclass
class NPTokenizationHypothesis:
    """Extended tokenization hypothesis that includes NP token replacement."""
    tokens: List[VectorSpace]
    confidence: float
    description: str
    np_replacements: List[tuple] = None  # List of (start_idx, end_idx, np_token) tuples
    
    def __post_init__(self):
        if self.np_replacements is None:
            self.np_replacements = []


def create_np_token(np: NounPhrase) -> VectorSpace:
    """Create a NounPhrase token from a parsed NounPhrase object.
    
    This creates a single token that represents the entire noun phrase.
    """
    # Create a new token with the NP's semantic vector
    np_token = VectorSpace()
    
    # Copy the semantic content from the NounPhrase
    if hasattr(np, 'vector') and np.vector:
        # Copy the vector data from all dimensions
        from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
        for dim in VECTOR_DIMENSIONS:
            value = np.vector[dim]
            if value != 0.0:  # Only copy non-zero values
                np_token[dim] = value
    
    # Mark this as a NounPhrase token
    np_token["NP"] = 1.0
    
    # Create a descriptive word for the token
    if hasattr(np, 'get_original_text'):
        np_token.word = f"NP({np.get_original_text()})"
    else:
        # Fallback: construct description from components
        parts = []
        if hasattr(np, 'determiner') and np.determiner:
            parts.append(np.determiner)
        if hasattr(np, 'adjectives') and np.adjectives:
            parts.extend(np.adjectives)
        if hasattr(np, 'noun') and np.noun:
            parts.append(np.noun)
        if hasattr(np, 'vector_text') and np.vector_text:
            parts.append(np.vector_text)
        
        text = " ".join(parts) if parts else "NP"
        np_token.word = f"NP({text})"
    
    # Store reference to original NP for Layer 3
    np_token._original_np = np
    
    return np_token


def find_np_sequences(tokens: List[VectorSpace]) -> List[tuple]:
    """Find noun phrase sequences in a list of tokens using the NP ATN.
    
    Returns list of (start_idx, end_idx, np_object) tuples.
    Uses greedy left-to-right parsing: try NP at each position, if successful
    consume those tokens and continue from the next position.
    """
    np_sequences = []
    i = 0
    
    while i < len(tokens):
        # Try to parse NP starting at position i
        # Use TokenStream position tracking to determine how many tokens were consumed
        subsequence = tokens[i:]  # Use all remaining tokens
        try:
            ts = TokenStream(subsequence)
            np = NounPhrase()
            np_start, np_end = build_np_atn(np, ts)
            initial_pos = ts.position
            result = run_atn(np_start, np_end, ts, np)
            
            if result is not None:
                # Found a valid NP - use actual consumed token count
                tokens_consumed = ts.position - initial_pos
                best_np = result
                best_end = i + tokens_consumed - 1  # Convert to absolute index
            else:
                best_np = None
                best_end = i
        except Exception:
            # NP parsing failed
            best_np = None
            best_end = i
        
        if best_np is not None:
            # Found an NP, add it and skip past it
            np_sequences.append((i, best_end, best_np))
            i = best_end + 1
        else:
            # No NP found starting at position i, move to next position
            i += 1
    
    return np_sequences


def replace_np_sequences(tokens: List[VectorSpace], np_sequences: List[tuple]) -> List[VectorSpace]:
    """Replace NP sequences with NP tokens."""
    if not np_sequences:
        return tokens
    
    new_tokens = []
    i = 0
    
    for start_idx, end_idx, np in np_sequences:
        # Add tokens before this NP
        while i < start_idx:
            new_tokens.append(tokens[i])
            i += 1
        
        # Add the NP token
        np_token = create_np_token(np)
        new_tokens.append(np_token)
        
        # Skip the original NP tokens
        i = end_idx + 1
    
    # Add remaining tokens
    while i < len(tokens):
        new_tokens.append(tokens[i])
        i += 1
    
    return new_tokens


def latn_tokenize_layer2(layer1_hypotheses: List[TokenizationHypothesis]) -> List[NPTokenizationHypothesis]:
    """LATN Layer 2: Replace noun phrase sequences with NounPhrase tokens.
    
    This is the main entry point for Layer 2 tokenization. It takes Layer 1 
    hypotheses and identifies noun phrase constructions, replacing them with 
    single NounPhrase tokens.
    
    Args:
        layer1_hypotheses: List of TokenizationHypothesis from Layer 1
        
    Returns:
        List of NPTokenizationHypothesis objects, ranked by confidence
    """
    layer2_hypotheses = []
    
    for base_hyp in layer1_hypotheses:
        # Find NP sequences in this tokenization using the NP ATN
        np_sequences = find_np_sequences(base_hyp.tokens)
        
        if np_sequences:
            # Create hypothesis with NP replacements
            new_tokens = replace_np_sequences(base_hyp.tokens, np_sequences)
            
            # Calculate confidence (slight boost for successful NP identification)
            new_confidence = base_hyp.confidence * 1.05
            
            layer2_hyp = NPTokenizationHypothesis(
                tokens=new_tokens,
                confidence=new_confidence,
                description=f"Layer 2: {len(np_sequences)} NP tokens from '{base_hyp.description}'",
                np_replacements=[(start, end, create_np_token(np)) for start, end, np in np_sequences]
            )
            layer2_hypotheses.append(layer2_hyp)
        else:
            # No NP sequences found, convert base hypothesis
            layer2_hyp = NPTokenizationHypothesis(
                tokens=base_hyp.tokens,
                confidence=base_hyp.confidence,
                description=f"Layer 2: No NPs from '{base_hyp.description}'",
                np_replacements=[]
            )
            layer2_hypotheses.append(layer2_hyp)
    
    # Sort by confidence
    layer2_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    return layer2_hypotheses
