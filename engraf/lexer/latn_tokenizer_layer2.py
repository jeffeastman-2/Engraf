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

from engraf.lexer.hypothesis import TokenizationHypothesis  
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet_np import run_np
from engraf.atn.np import build_np_atn
from engraf.atn.coordinated_np import build_coordinated_np_atn
from engraf.atn.core import run_atn
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.utils.predicates import is_determiner, is_adjective, is_noun, is_vector


def create_np_token(np_or_conj) -> VectorSpace:
    """Create a token from a parsed NounPhrase or ConjunctionPhrase object.
    
    This creates a single token that represents the entire noun phrase or 
    coordinated noun phrase construction.
    """
    # Create a new token with the semantic vector
    token = VectorSpace()
    
    # Handle ConjunctionPhrase (coordinated NPs)
    if isinstance(np_or_conj, ConjunctionPhrase):
        # Copy the semantic content from the ConjunctionPhrase
        if hasattr(np_or_conj, 'vector') and np_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = np_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value
        
        # Mark this as a ConjunctionPhrase token
        token["conj"] = 1.0
        token["NP"] = 1.0  # Also mark as NP since it functions as one
        
        # Create descriptive word
        if hasattr(np_or_conj, 'get_original_text'):
            token.word = f"CONJ-NP({np_or_conj.get_original_text()})"
        else:
            token.word = "CONJ-NP"
    
    # Handle regular NounPhrase
    else:
        # Copy the semantic content from the NounPhrase
        if hasattr(np_or_conj, 'vector') and np_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = np_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value
        
        # Mark this as a NounPhrase token
        token["NP"] = 1.0
        
        # Create descriptive word
        if hasattr(np_or_conj, 'get_original_text'):
            token.word = f"NP({np_or_conj.get_original_text()})"
        else:
            # Fallback: construct description from components
            parts = []
            if hasattr(np_or_conj, 'determiner') and np_or_conj.determiner:
                parts.append(np_or_conj.determiner)
            if hasattr(np_or_conj, 'adjectives') and np_or_conj.adjectives:
                parts.extend(np_or_conj.adjectives)
            if hasattr(np_or_conj, 'noun') and np_or_conj.noun:
                parts.append(np_or_conj.noun)
            if hasattr(np_or_conj, 'vector_text') and np_or_conj.vector_text:
                parts.append(np_or_conj.vector_text)
            
            text = " ".join(parts) if parts else "NP"
            token.word = f"NP({text})"
    
    # Store reference to original object for Layer 3
    token._original_np = np_or_conj
    
    return token


def find_np_sequences(tokens: List[VectorSpace]) -> List[tuple]:
    """Find noun phrase sequences in a list of tokens using the NP ATN.
    
    Returns list of (start_idx, end_idx, np_object) tuples.
    Uses greedy left-to-right parsing: try NP at each position, if successful
    consume those tokens and continue from the next position.
    """
    np_sequences = []
    i = 0
    
    while i < len(tokens):
        # Try to parse coordinated NP first, then fall back to simple NP
        # Use TokenStream position tracking to determine how many tokens were consumed
        subsequence = tokens[i:]  # Use all remaining tokens
        best_np = None
        best_end = i
        
        # First, try to parse a simple NP
        try:
            ts = TokenStream(subsequence)
            np = NounPhrase()
            np_start, np_end = build_np_atn(np, ts)
            result = run_atn(np_start, np_end, ts, np)
            
            if result is not None:
                # Found a valid simple NP
                best_np = result
                best_end = i + ts.position - 1
                
                # Check for conjunctions to build coordinated NPs
                while ts.peek() and ts.peek().isa("conj"):
                    # There's a conjunction! Try to parse another NP
                    conj_token = ts.next()  # consume the conjunction
                    np2 = NounPhrase()
                    np2_start, np2_end = build_np_atn(np2, ts)
                    np2_result = run_atn(np2_start, np2_end, ts, np2)
                    
                    if np2_result is not None:
                        # Successfully parsed another NP - create/extend coordination
                        if isinstance(best_np, NounPhrase):
                            # Convert to ConjunctionPhrase
                            coord_np = ConjunctionPhrase(conj_token, left=best_np, right=np2_result)
                            best_np = coord_np
                        elif isinstance(best_np, ConjunctionPhrase):
                            # Extend existing coordination by chaining
                            new_coord = ConjunctionPhrase(conj_token, left=best_np, right=np2_result)
                            best_np = new_coord
                        
                        # Update best_end to include the newly parsed NP
                        best_end = i + ts.position - 1
                    else:
                        # Failed to parse second NP - break out of coordination loop
                        # Put the conjunction token back by rewinding
                        ts.position -= 1
                        break
        except Exception:
                # Simple NP parsing also failed
                pass
        
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


def latn_tokenize_layer2(layer1_hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
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
            
            layer2_hyp = TokenizationHypothesis(
                tokens=new_tokens,
                confidence=new_confidence,
                description=f"Layer 2: {len(np_sequences)} NP tokens from '{base_hyp.description}'",
                replacements=[(start, end, create_np_token(np)) for start, end, np in np_sequences]
            )
            layer2_hypotheses.append(layer2_hyp)
        else:
            # No NP sequences found, convert base hypothesis
            layer2_hyp = TokenizationHypothesis(
                tokens=base_hyp.tokens,
                confidence=base_hyp.confidence,
                description=f"Layer 2: No NPs from '{base_hyp.description}'",
                replacements=[]
            )
            layer2_hypotheses.append(layer2_hyp)
    
    # Sort by confidence
    layer2_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    return layer2_hypotheses
