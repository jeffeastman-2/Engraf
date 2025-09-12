#!/usr/bin/env python3
"""
LATN Layer 5: SentencePhrase Token Replacement

This module implements Layer 5 of the LATN (Layered Augmented Transition Network) system.
Layer 5 replaces sentence phrase constructions with single SentencePhrase tokens.

Layer 5 builds on:
- Layer 4: Multi-hypothesis tokenization with morphological inflection

This layer identifies sentence phrases like "in the red box", "on a very large sphere", "[1,2,3]"
and replaces them with single SentencePhrase tokens containing the semantic meaning.
"""

from typing import List

from engraf.lexer.hypothesis import TokenizationHypothesis  
from engraf.lexer.token_stream import TokenStream
from engraf.atn.sentence import build_sentence_atn
from engraf.atn.core import run_atn
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.sentence_phrase import SentencePhrase   
from engraf.lexer.vector_space import VectorSpace


def create_sp_token(sentence_or_conj) -> VectorSpace:
    """Create a token from a parsed SentencePhrase object.

    This creates a single token that represents the entire sentence phrase construction.
    """
    # Create a new token with the semantic vector
    token = VectorSpace()
    
    # Handle ConjunctionPhrase (coordinated NPs)
    if isinstance(sentence_or_conj, ConjunctionPhrase):
        # Copy the semantic content from the ConjunctionPhrase
        if hasattr(sentence_or_conj, 'vector') and sentence_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = sentence_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value
        
        # Mark this as a ConjunctionPhrase token
        token["conj"] = 1.0
        token["SP"] = 1.0  # Also mark as S since it functions as one

        # Create descriptive word - use phrase-level display if available
        if hasattr(sentence_or_conj, '_phrase_level_display'):
            token.word = sentence_or_conj._phrase_level_display
        elif hasattr(sentence_or_conj, 'get_original_text'):
            token.word = f"CONJ-SP({sentence_or_conj.get_original_text()})"
        else:
            token.word = "CONJ-SP"

    # Handle regular SentencePhrase
    else:
        # Copy the semantic content from the SentencePhrase
        if hasattr(sentence_or_conj, 'vector') and sentence_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = sentence_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value

        # Mark this as a SentencePhrase token
        token["SP"] = 1.0

        # Create descriptive word
        if hasattr(sentence_or_conj, 'get_original_text'):
            token.word = f"SP({sentence_or_conj.get_original_text()})"
        else:
            # Fallback: construct description from components
            parts = []
            #if hasattr(sentence_or_conj, 'subject') and sentence_or_conj.subject:
            #    parts.append(sentence_or_conj.subject)
            #if hasattr(sentence_or_conj, 'vector_text') and sentence_or_conj.vector_text:
            #    parts.append(sentence_or_conj.vector_text)

            text = " ".join(parts) if parts else "SP"
            token.word = f"SP({text})"

    # Store reference to original object for Layer 3
    token._original_sp = sentence_or_conj

    return token


def find_sp_sequences(tokens: List[VectorSpace], build_conjunctions: bool = False) -> List[tuple]:
    """Find sentence phrase sequences in a list of tokens using the SP ATN.

    Returns list of (start_idx, end_idx, sp_object) tuples.
    Uses greedy left-to-right parsing: try SP at each position, if successful
    consume those tokens and continue from the next position.
    """
    sp_sequences = []
    i = 0
    
    while i < len(tokens):
        # Use TokenStream position tracking to determine how many tokens were consumed
        subsequence = tokens[i:]  # Use all remaining tokens
        best_sp = None
        best_end = i

        # First, try to parse a simple SP
        try:
            ts = TokenStream(subsequence)
            sp = SentencePhrase()
            sp_start, sp_end = build_sentence_atn(sp, ts)
            result = run_atn(sp_start, sp_end, ts, sp)

            if result is not None:
                # Found a valid simple SP
                best_sp = result
                best_end = i + ts.position - 1
                
                # Check for conjunctions to build coordinated NPs
                while build_conjunctions and ts.peek() and ts.peek().isa("conj"):
                    # There's a conjunction! Try to parse another NP
                    conj_token = ts.next()  # consume the conjunction
                    sp2 = SentencePhrase()
                    sp2_start, sp2_end = build_sentence_atn(sp2, ts)
                    sp2_result = run_atn(sp2_start, sp2_end, ts, sp2)

                    if sp2_result is not None:
                        # Successfully parsed another SP - create/extend coordination
                        if isinstance(best_sp, SentencePhrase):
                            # Convert to ConjunctionPhrase
                            coord_sp = ConjunctionPhrase(conj_token, left=best_sp, right=sp2_result)
                            coord_sp.vector["plural"] = 1.0
                            best_sp = coord_sp
                        elif isinstance(best_sp, ConjunctionPhrase):
                            # Extend existing coordination by chaining
                            new_coord = ConjunctionPhrase(conj_token, left=best_sp.right, right=sp2_result)
                            best_sp.right = new_coord

                        # Update best_end to include the newly parsed SP
                        best_end = i + ts.position - 1
                    else:
                        # Failed to parse second SP - break out of coordination loop
                        # Put the conjunction token back by rewinding
                        ts.position -= 1
                        break
        except Exception:
                # Simple SP parsing also failed
                pass

        if best_sp is not None:
            # Found a SP, add it and skip past it
            sp_sequences.append((i, best_end, best_sp))
            i = best_end + 1
        else:
            # No SP found starting at position i, move to next position
            i += 1

    return sp_sequences


def replace_sp_sequences(tokens: List[VectorSpace], sp_sequences: List[tuple]) -> List[VectorSpace]:
    """Replace SP sequences with SP tokens."""
    if not sp_sequences:
        return tokens
    
    new_tokens = []
    i = 0

    for start_idx, end_idx, sp in sp_sequences:
        # Add tokens before this SP
        while i < start_idx:
            new_tokens.append(tokens[i])
            i += 1

        # Add the SP token
        sp_token = create_sp_token(sp)
        new_tokens.append(sp_token)

        # Skip the original SP tokens
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
    greedy_sequences = find_sp_sequences(tokens, False)
    hypotheses.append(greedy_sequences)

    # Hypothesis 2: Phrase-level coordination (respecting SP boundaries)
    phrase_sequences = find_sp_sequences(tokens, True)
    if _is_different(phrase_sequences, greedy_sequences):  # Only add if different
        hypotheses.append(phrase_sequences)
    
    return hypotheses


def calculate_coordination_confidence(sp_sequences: List[tuple], is_phrase_level: bool = False) -> float:
    """Calculate confidence penalty/bonus based on coordination naturalness.
    
    Args:
        sp_sequences: The SP sequences found by the algorithm
        is_phrase_level: Whether this represents phrase-level coordination
        
    Returns:
        Confidence multiplier (1.0 = no change, >1.0 = bonus, <1.0 = penalty)
    """
    if not sp_sequences:    
        return 1.0
    
    confidence = 1.0
    
    # Count coordinated phrases
    coord_count = sum(1 for _, _, sp in sp_sequences if isinstance(sp, ConjunctionPhrase))

    if coord_count > 0:
        if is_phrase_level:
            # Bonus for phrase-level coordination (more natural)
            confidence *= 1.15
        else:
            # Small penalty for local coordination when phrase-level is possible
            confidence *= 0.95
    
    return confidence


def latn_tokenize_layer5(layer4_hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
    """LATN Layer 5: Replace sentence boundary sequences with SentenceBoundary tokens.

    This is the main entry point for Layer 5 tokenization. It takes Layer 4
    hypotheses and identifies sentence boundary constructions, replacing them with
    single SentenceBoundary tokens.
    
    For ambiguous coordination structures, this generates multiple hypotheses
    that can be disambiguated by higher layers.
    
    Args:
        layer2_hypotheses: List of TokenizationHypothesis from Layer 2

    Returns:
        List of SPTokenizationHypothesis objects, ranked by confidence
    """
    layer5_hypotheses = []

    for base_hyp in layer4_hypotheses:
        # Generate multiple coordination hypotheses for ambiguous structures
        coordination_hypotheses = find_coordination_hypotheses(base_hyp.tokens)

        for i, sp_sequences in enumerate(coordination_hypotheses):
            if sp_sequences:
                # Create hypothesis with SP replacements
                new_tokens = replace_sp_sequences(base_hyp.tokens, sp_sequences)

                # Calculate confidence with coordination naturalness
                is_phrase_level = (i == 1)  # Second hypothesis is phrase-level
                coord_confidence = calculate_coordination_confidence(sp_sequences, is_phrase_level)
                new_confidence = base_hyp.confidence * 1.05 * coord_confidence
                
                # Create description for this hypothesis
                coord_type = "phrase-level" if is_phrase_level else "local"
                description = f"Layer 5 ({coord_type}): {len(sp_sequences)} SP sequences"

                layer5_hyp = TokenizationHypothesis(
                    tokens=new_tokens,
                    confidence=new_confidence,
                    description=description,
                    replacements=[(start, end, create_sp_token(sp)) for start, end, sp in sp_sequences]
                )
                layer5_hypotheses.append(layer5_hyp)
            else:
                # No SP sequences found, convert base hypothesis
                layer5_hyp = TokenizationHypothesis(
                    tokens=base_hyp.tokens,
                    confidence=base_hyp.confidence,
                    description="Layer 5: No SP sequences found",
                    replacements=[]
                )
                layer5_hypotheses.append(layer5_hyp)

    # Sort by confidence (highest first)
    layer5_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

    return layer5_hypotheses
