#!/usr/bin/env python3
"""
LATN Layer 3: PrepositionalPhrase Token Replacement

This module implements Layer 3 of the LATN (Layer-Aware Tokenization Network) system.
Layer 3 replaces prepositional phrase constructions with single PrepositionalPhrase tokens.

Layer 3 builds on:
- Layer 1: Multi-hypothesis tokenization with morphological inflection

This layer identifies prepositional phrases like "in the red box", "on a very large sphere", "[1,2,3]"
and replaces them with single PrepositionalPhrase tokens containing the semantic meaning.
"""

from typing import List

from engraf.lexer.hypothesis import TokenizationHypothesis  
from engraf.lexer.token_stream import TokenStream
from engraf.atn.pp import build_pp_atn
from engraf.atn.core import run_atn
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.lexer.vector_space import VectorSpace


def create_pp_token(pp_or_conj) -> VectorSpace:
    """Create a token from a parsed PrepositionalPhrase object.

    This creates a single token that represents the entire prepositional phrase construction.
    """
    # Create a new token with the semantic vector
    token = VectorSpace()
    
    # Handle ConjunctionPhrase (coordinated NPs)
    if isinstance(pp_or_conj, ConjunctionPhrase):
        # Copy the semantic content from the ConjunctionPhrase
        if hasattr(pp_or_conj, 'vector') and pp_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = pp_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value
        
        # Mark this as a ConjunctionPhrase token
        token["conj"] = 1.0
        token["PP"] = 1.0  # Also mark as PP since it functions as one

        # Create descriptive word - use phrase-level display if available
        if hasattr(pp_or_conj, '_phrase_level_display'):
            token.word = pp_or_conj._phrase_level_display
        elif hasattr(pp_or_conj, 'get_original_text'):
            token.word = f"CONJ-PP({pp_or_conj.get_original_text()})"
        else:
            token.word = "CONJ-PP"

    # Handle regular PrepositionalPhrase
    else:
        # Copy the semantic content from the PrepositionalPhrase    
        if hasattr(pp_or_conj, 'vector') and pp_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = pp_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value

        # Mark this as a PrepositionalPhrase token
        token["PP"] = 1.0

        # Create descriptive word
        if hasattr(pp_or_conj, 'get_original_text'):
            token.word = f"PP({pp_or_conj.get_original_text()})"
        else:
            # Fallback: construct description from components
            parts = []
            if hasattr(pp_or_conj, 'preposition') and pp_or_conj.preposition:
                parts.append(pp_or_conj.preposition)
            if hasattr(pp_or_conj, 'vector_text') and pp_or_conj.vector_text:
                parts.append(pp_or_conj.vector_text)

            text = " ".join(parts) if parts else "PP"
            token.word = f"PP({text})"

    # Store reference to original object for Layer 3
    token._original_pp = pp_or_conj

    return token


def find_pp_sequences(tokens: List[VectorSpace], build_conjunctions: bool = False) -> List[tuple]:
    """Find prepositional phrase sequences in a list of tokens using the PP ATN.

    Returns list of (start_idx, end_idx, pp_object) tuples.
    Uses greedy left-to-right parsing: try PP at each position, if successful
    consume those tokens and continue from the next position.
    """
    pp_sequences = []
    i = 0
    
    while i < len(tokens):
        # Use TokenStream position tracking to determine how many tokens were consumed
        subsequence = tokens[i:]  # Use all remaining tokens
        best_pp = None
        best_end = i

        # First, try to parse a simple PP
        try:
            ts = TokenStream(subsequence)
            pp = PrepositionalPhrase()
            pp_start, pp_end = build_pp_atn(pp, ts)
            result = run_atn(pp_start, pp_end, ts, pp)
            
            if result is not None:
                # Found a valid simple PP
                best_pp = result
                best_end = i + ts.position - 1
                
                # Check for conjunctions to build coordinated NPs
                while build_conjunctions and ts.peek() and ts.peek().isa("conj"):
                    # There's a conjunction! Try to parse another NP
                    conj_token = ts.next()  # consume the conjunction
                    pp2 = PrepositionalPhrase()
                    pp2_start, pp2_end = build_pp_atn(pp2, ts)
                    pp2_result = run_atn(pp2_start, pp2_end, ts, pp2)

                    if pp2_result is not None:
                        # Successfully parsed another PP - create/extend coordination
                        if isinstance(best_pp, PrepositionalPhrase):
                            # Convert to ConjunctionPhrase
                            coord_pp = ConjunctionPhrase(conj_token, left=best_pp, right=pp2_result)
                            best_pp = coord_pp
                        elif isinstance(best_pp, ConjunctionPhrase):
                            # Extend existing coordination by chaining
                            new_coord = ConjunctionPhrase(conj_token, left=best_pp, right=pp2_result)
                            best_pp = new_coord

                        # Update best_end to include the newly parsed PP
                        best_end = i + ts.position - 1
                    else:
                        # Failed to parse second PP - break out of coordination loop
                        # Put the conjunction token back by rewinding
                        ts.position -= 1
                        break
        except Exception:
                # Simple PP parsing also failed
                pass

        if best_pp is not None:
            # Found a PP, add it and skip past it
            pp_sequences.append((i, best_end, best_pp))
            i = best_end + 1
        else:
            # No PP found starting at position i, move to next position
            i += 1

    return pp_sequences


def replace_pp_sequences(tokens: List[VectorSpace], pp_sequences: List[tuple]) -> List[VectorSpace]:
    """Replace PP sequences with PP tokens."""
    if not pp_sequences:
        return tokens
    
    new_tokens = []
    i = 0

    for start_idx, end_idx, pp in pp_sequences:
        # Add tokens before this PP
        while i < start_idx:
            new_tokens.append(tokens[i])
            i += 1

        # Add the PP token
        pp_token = create_pp_token(pp)
        new_tokens.append(pp_token)

        # Skip the original PP tokens
        i = end_idx + 1
    
    # Add remaining tokens
    while i < len(tokens):
        new_tokens.append(tokens[i])
        i += 1
    
    return new_tokens


def find_coordination_hypotheses(tokens: List[VectorSpace]) -> List[List[tuple]]:
    """Generate multiple coordination hypotheses for ambiguous structures.
    
    For sentences with coordination ambiguity, this generates alternative
    interpretations that can be disambiguated by higher layers.
    
    Args:
        tokens: Input token sequence
        
    Returns:
        List of hypothesis alternatives, each containing NP sequences
    """
    hypotheses = []

    def _is_different(a, b):
        return a.__repr__() != b.__repr__()

    # Hypothesis 1: Current greedy algorithm (local coordination)
    greedy_sequences = find_pp_sequences(tokens, False)
    hypotheses.append(greedy_sequences)
    
    # Hypothesis 2: Phrase-level coordination (respecting PP boundaries)
    phrase_sequences = find_pp_sequences(tokens, True)
    if _is_different(phrase_sequences, greedy_sequences):  # Only add if different
        hypotheses.append(phrase_sequences)
    
    return hypotheses


def calculate_coordination_confidence(pp_sequences: List[tuple], is_phrase_level: bool = False) -> float:
    """Calculate confidence penalty/bonus based on coordination naturalness.
    
    Args:
        pp_sequences: The PP sequences found by the algorithm
        is_phrase_level: Whether this represents phrase-level coordination
        
    Returns:
        Confidence multiplier (1.0 = no change, >1.0 = bonus, <1.0 = penalty)
    """
    if not pp_sequences:    
        return 1.0
    
    confidence = 1.0
    
    # Count coordinated phrases
    coord_count = sum(1 for _, _, pp in pp_sequences if isinstance(pp, ConjunctionPhrase))
    
    if coord_count > 0:
        if is_phrase_level:
            # Bonus for phrase-level coordination (more natural)
            confidence *= 1.15
        else:
            # Small penalty for local coordination when phrase-level is possible
            confidence *= 0.95
    
    return confidence


def latn_tokenize_layer3(layer2_hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
    """LATN Layer 3: Replace prepositional phrase sequences with PrepositionalPhrase tokens.

    This is the main entry point for Layer 3 tokenization. It takes Layer 2
    hypotheses and identifies prepositional phrase constructions, replacing them with
    single PrepositionalPhrase tokens.
    
    For ambiguous coordination structures, this generates multiple hypotheses
    that can be disambiguated by higher layers.
    
    Args:
        layer2_hypotheses: List of TokenizationHypothesis from Layer 2

    Returns:
        List of PPTokenizationHypothesis objects, ranked by confidence
    """
    layer3_hypotheses = []

    for base_hyp in layer2_hypotheses:
        # Generate multiple coordination hypotheses for ambiguous structures
        coordination_hypotheses = find_coordination_hypotheses(base_hyp.tokens)

        for i, pp_sequences in enumerate(coordination_hypotheses):
            if pp_sequences:
                # Create hypothesis with PP replacements
                new_tokens = replace_pp_sequences(base_hyp.tokens, pp_sequences)

                # Calculate confidence with coordination naturalness
                is_phrase_level = (i == 1)  # Second hypothesis is phrase-level
                coord_confidence = calculate_coordination_confidence(pp_sequences, is_phrase_level)
                new_confidence = base_hyp.confidence * 1.05 * coord_confidence
                
                # Create description for this hypothesis
                coord_type = "phrase-level" if is_phrase_level else "local"
                description = f"Layer 3 ({coord_type}): {len(pp_sequences)} PP sequences"

                layer3_hyp = TokenizationHypothesis(
                    tokens=new_tokens,
                    confidence=new_confidence,
                    description=description,
                    replacements=[(start, end, create_pp_token(pp)) for start, end, pp in pp_sequences]
                )
                layer3_hypotheses.append(layer3_hyp)
            else:
                # No PP sequences found, convert base hypothesis
                layer3_hyp = TokenizationHypothesis(
                    tokens=base_hyp.tokens,
                    confidence=base_hyp.confidence,
                    description="Layer 3: No PP sequences found",
                    replacements=[]
                )
                layer3_hypotheses.append(layer3_hyp)

    # Sort by confidence (highest first)
    layer3_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

    return layer3_hypotheses
