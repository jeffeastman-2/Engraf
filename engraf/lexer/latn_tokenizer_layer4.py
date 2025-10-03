#!/usr/bin/env python3
"""
LATN Layer 4: VerbPhrase Token Replacement

This module implements Layer 4 of the LATN (Layered Tokenization Network) system.
Layer 4 replaces verb phrase constructions with single VerbPhrase tokens.

Layer 4 builds on:
- Layer 3: Multi-hypothesis PP tokenization with morphological inflection

This layer identifies verb phrases like "running quickly", "jumps over the lazy dog", "[1,2,3]"
and replaces them with single VerbPhrase tokens containing the semantic meaning.
"""

from typing import List

from engraf.lexer.hypothesis import TokenizationHypothesis  
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import run_atn
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.atn.vp import build_vp_atn
from engraf.pos.verb_phrase import VerbPhrase
from engraf.utils.debug import set_debug


def create_vp_token(vp_or_conj) -> VectorSpace:
    """Create a token from a parsed VerbPhrase object.

    This creates a single token that represents the entire verb phrase construction.
    """
    # Create a new token with the semantic vector
    token = VectorSpace()
    
    # Handle ConjunctionPhrase (coordinated VPs)
    if isinstance(vp_or_conj, ConjunctionPhrase):
        # Copy the semantic content from the ConjunctionPhrase
        if hasattr(vp_or_conj, 'vector') and vp_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = vp_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value
        
        # Mark this as a ConjunctionPhrase token
        token["conj"] = 1.0
        token["VP"] = 1.0  # Also mark as VP since it functions as one

        # Create descriptive word - use phrase-level display if available
        if hasattr(vp_or_conj, '_phrase_level_display'):
            token.word = vp_or_conj._phrase_level_display
        elif hasattr(vp_or_conj, 'get_original_text'):
            token.word = f"CONJ-VP({vp_or_conj.get_original_text()})"
        else:
            token.word = "CONJ-VP"

    # Handle regular VerbPhrase
    else:
        # Copy the semantic content from the VerbPhrase
        if hasattr(vp_or_conj, 'vector') and vp_or_conj.vector:
            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
            for dim in VECTOR_DIMENSIONS:
                value = vp_or_conj.vector[dim]
                if value != 0.0:
                    token[dim] = value

        # Mark this as a VerbPhrase token
        token["VP"] = 1.0

        # Create descriptive word
        if hasattr(vp_or_conj, 'get_original_text'):
            token.word = f"VP({vp_or_conj.get_original_text()})"
        else:
            # Fallback: construct description from components
            parts = []
            if hasattr(vp_or_conj, 'verb') and vp_or_conj.verb:    
                parts.append(vp_or_conj.verb)
            if hasattr(vp_or_conj, 'vector_text') and vp_or_conj.vector_text:
                parts.append(vp_or_conj.vector_text)

            text = " ".join(parts) if parts else "VP"
            token.word = f"VP({text})"

    # Store reference to original object for Layer 3
    token.phrase = vp_or_conj

    return token


def find_vp_sequences(tokens: List[VectorSpace], build_conjunctions: bool = False) -> List[tuple]:
    """Find verb phrase sequences in a list of tokens using the VP ATN.

    Returns list of (start_idx, end_idx, vp_object) tuples.
    Uses greedy left-to-right parsing: try VP at each position, if successful
    consume those tokens and continue from the next position.
    """
    vp_sequences = []
    i = 0
    
    while i < len(tokens):
        # Use TokenStream position tracking to determine how many tokens were consumed
        subsequence = tokens[i:]  # Use all remaining tokens
        best_vp = None
        best_end = i

        # First, try to parse a simple VP
        try:
            ts = TokenStream(subsequence)
            vp = VerbPhrase()
            vp_start, vp_end = build_vp_atn(vp, ts)
            result = run_atn(vp_start, vp_end, ts, vp)

            if result is not None:
                # Found a valid simple VP
                best_vp = result
                best_end = i + ts.position - 1
                
                # Check for conjunctions to build coordinated VPs
                while build_conjunctions and ts.peek() and (ts.peek().isa("conj") or ts.peek().isa("comma")):
                    # There's a conjunction! Try to parse another VP
                    conj_token = ts.next()  # consume the conjunction
                    while conj_token.isa("comma") and ts.peek().isa("conj"):
                        conj_token = ts.next()  # consume the conjunction after comma
                    vp2 = VerbPhrase()
                    vp2_start, vp2_end = build_vp_atn(vp2, ts)
                    vp2_result = run_atn(vp2_start, vp2_end, ts, vp2)

                    if vp2_result is not None:
                        # Successfully parsed another VP - create/extend coordination
                        if isinstance(best_vp, VerbPhrase):
                            # Convert to ConjunctionPhrase
                            coord_vp = ConjunctionPhrase(conj_token, [best_vp, vp2_result])
                            coord_vp.vector["plural"] = 1.0
                            best_vp = coord_vp
                        elif isinstance(best_vp, ConjunctionPhrase):
                            if best_vp.vector.isa("comma"):
                                best_vp.vector["comma"] = 0.0
                                best_vp.vector += conj_token.vector
                            if (best_vp.vector.isa("and") and conj_token.isa("or")) \
                                or (best_vp.vector.isa("or") and conj_token.isa("and")):
                                raise ValueError("Mixed conjunctions 'and' and 'or' not supported in coordination")
                            best_vp.phrases.append(vp2_result)

                        # Update best_end to include the newly parsed VP
                        best_end = i + ts.position - 1
                    else:
                        # Failed to parse second VP - break out of coordination loop
                        # Put the conjunction token back by rewinding
                        ts.position -= 1
                        break
        except Exception:
                # Simple VP parsing also failed
                pass

        if best_vp is not None:
            # Found a VP, add it and skip past it
            vp_sequences.append((i, best_end, best_vp))
            i = best_end + 1
        else:
            # No VP found starting at position i, move to next position
            i += 1

    return vp_sequences


def replace_vp_sequences(tokens: List[VectorSpace], vp_sequences: List[tuple]) -> List[VectorSpace]:
    """Replace VP sequences with VP tokens."""
    if not vp_sequences:
        return tokens
    
    new_tokens = []
    i = 0

    for start_idx, end_idx, vp in vp_sequences:
        # Add tokens before this VP
        while i < start_idx:
            new_tokens.append(tokens[i])
            i += 1

        # Add the VP token
        vp_token = create_vp_token(vp)
        new_tokens.append(vp_token)

        # Skip the original VP tokens
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
    Layer 4 Input: NP, PP tokens from Layer 4
    Layer 4 Output: VP tokens containing SentencePhrase structures

    Args:
        tokens: Input token sequence
        
    Returns:
        List of hypothesis alternatives, each containing VP sequences
    """
    hypotheses = []
    from engraf.lexer.latn_layer_executor import is_different_phrase_sequence

    # Hypothesis 1: Current greedy algorithm (local coordination)
    greedy_sequences = find_vp_sequences(tokens, True)
    hypotheses.append(greedy_sequences)

    # Hypothesis 2: Phrase-level coordination (respecting VP boundaries)
    #phrase_sequences = find_vp_sequences(tokens, True)
    #if is_different_phrase_sequence(phrase_sequences, greedy_sequences):  # Only add if different
    #    hypotheses.append(phrase_sequences)
    
    return hypotheses


def calculate_coordination_confidence(vp_sequences: List[tuple], is_phrase_level: bool = False) -> float:
    """Calculate confidence penalty/bonus based on coordination naturalness.
    
    Args:
        vp_sequences: The VP sequences found by the algorithm
        is_phrase_level: Whether this represents phrase-level coordination
        
    Returns:
        Confidence multiplier (1.0 = no change, >1.0 = bonus, <1.0 = penalty)
    """
    if not vp_sequences:
        return 1.0
    
    confidence = 1.0
    
    # Count coordinated phrases
    coord_count = sum(1 for _, _, vp in vp_sequences if isinstance(vp, ConjunctionPhrase))

    if coord_count > 0:
        if is_phrase_level:
            # Bonus for phrase-level coordination (more natural)
            confidence *= 1.15
        else:
            # Small penalty for local coordination when phrase-level is possible
            confidence *= 0.95
    
    return confidence


def latn_tokenize_layer4(layer3_hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
    """LATN Layer 4: Replace verb phrase sequences with VerbPhrase tokens.

    This is the main entry point for Layer 4 tokenization. It takes Layer 3
    hypotheses and identifies verb phrase constructions, replacing them with
    single VerbPhrase tokens.
    
    For ambiguous coordination structures, this generates multiple hypotheses
    that can be disambiguated by higher layers.
    
    Args:
        layer2_hypotheses: List of TokenizationHypothesis from Layer 2

    Returns:
        List of VPTokenizationHypothesis objects, ranked by confidence
    """
    layer4_hypotheses = []
    #set_debug(True)
    
    for base_hyp in layer3_hypotheses:
        # Generate multiple coordination hypotheses for ambiguous structures
        coordination_hypotheses = find_coordination_hypotheses(base_hyp.tokens)

        for i, vp_sequences in enumerate(coordination_hypotheses):
            if vp_sequences:
                # Create hypothesis with VP replacements
                new_tokens = replace_vp_sequences(base_hyp.tokens, vp_sequences)

                # Calculate confidence with coordination naturalness
                is_phrase_level = (i == 1)  # Second hypothesis is phrase-level
                coord_confidence = calculate_coordination_confidence(vp_sequences, is_phrase_level)
                new_confidence = base_hyp.confidence * 1.05 * coord_confidence
                
                # Create description for this hypothesis
                coord_type = "phrase-level" if is_phrase_level else "local"
                description = f"Layer 4 ({coord_type}): {len(vp_sequences)} VP sequences"

                layer4_hyp = TokenizationHypothesis(
                    tokens=new_tokens,
                    confidence=new_confidence,
                    description=description,
                    replacements=[(start, end, create_vp_token(vp)) for start, end, vp in vp_sequences]
                )
                layer4_hypotheses.append(layer4_hyp)
            else:
                # No VP sequences found, convert base hypothesis
                layer4_hyp = TokenizationHypothesis(
                    tokens=base_hyp.tokens,
                    confidence=base_hyp.confidence,
                    description="Layer 4: No VP sequences found",
                    replacements=[]
                )
                layer4_hypotheses.append(layer4_hyp)

    # Sort by confidence (highest first)
    layer4_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

    return layer4_hypotheses
