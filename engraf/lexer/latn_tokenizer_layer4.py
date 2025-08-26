#!/usr/bin/env python3
"""
LATN Layer 4: Verb Phrase Token Replacement

This module implements Layer 4 of the LATN (Layered Augmented Transition Network) system.
Layer 4 replaces verb phrase constructions with single VerbPhrase tokens.

Layer 4 builds on:
- Layer 1: Multi-hypothesis tokenization with morphological inflection
- Layer 2: NounPhrase token replacement
- Layer 3: PrepositionalPhrase token replacement
- Layer 4: VerbPhrase token replacement

This layer identifies verb phrases like "create a red box", "move the sphere", "rotate 45 degrees"
and replaces them with single VerbPhrase tokens containing the semantic meaning and action intent.
"""

from typing import List, Optional
from dataclasses import dataclass
import copy

from engraf.lexer.hypothesis import TokenizationHypothesis
from engraf.lexer.token_stream import TokenStream
from engraf.atn.vp import build_vp_atn
from engraf.pos.verb_phrase import VerbPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.utils.predicates import is_verb
from engraf.utils.debug import debug_print


def latn_tokenize_layer4(layer3_hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
    """
    Apply Layer 4 VP tokenization to Layer 3 hypotheses.
    
    Args:
        layer3_hypotheses: List of hypotheses from Layer 3 (with PP tokens)
        
    Returns:
        List of VPTokenizationHypothesis with verb phrase tokens
    """
    if not layer3_hypotheses:
        debug_print("Layer 4: No Layer 3 hypotheses provided")
        return []
    
    layer4_hypotheses = []
    
    for hyp in layer3_hypotheses:
        debug_print(f"Layer 4: Processing hypothesis of type {type(hyp)}")
        
        # Check if hyp has tokens attribute
        if not hasattr(hyp, 'tokens'):
            debug_print(f"Layer 4: Hypothesis has no tokens attribute: {hyp}")
            continue
            
        debug_print(f"Layer 4: Processing hypothesis with {len(hyp.tokens)} tokens")
        
        # Try to find verb phrases in this hypothesis
        vp_hypotheses = _find_verb_phrases_in_hypothesis(hyp)
        layer4_hypotheses.extend(vp_hypotheses)
    
    # Sort by confidence and return
    layer4_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    debug_print(f"Layer 4: Generated {len(layer4_hypotheses)} VP hypotheses")
    
    return layer4_hypotheses


def _find_verb_phrases_in_hypothesis(hyp: TokenizationHypothesis) -> List[TokenizationHypothesis]:
    """Find and replace verb phrases in a single hypothesis using greedy left-to-right parsing."""
    tokens = hyp.tokens[:]  # Copy tokens
    vp_replacements = []
    
    debug_print(f"Layer 4: Looking for VPs in tokens: {[t.word for t in tokens]}")
    
    # Greedy left-to-right VP parsing (like Layers 2 and 3)
    i = 0
    while i < len(tokens):
        if is_verb(tokens[i]):
            debug_print(f"Layer 4: Found verb '{tokens[i].word}' at position {i}")
            
            # Try to parse a VP starting at this position
            vp_result = _try_parse_verb_phrase(tokens, i)
            
            if vp_result:
                vp_token, end_idx = vp_result
                debug_print(f"Layer 4: Successfully parsed VP from {i} to {end_idx}: {vp_token.word}")
                
                # Replace the VP sequence with the single VP token
                tokens[i:end_idx + 1] = [vp_token]
                vp_replacements.append((i, end_idx, vp_token))
                
                # Continue from after the replacement (greedy approach)
                i += 1  # Move past the new VP token
            else:
                i += 1
        else:
            i += 1
    
    # Create a single hypothesis with all VP replacements applied
    vp_hyp = TokenizationHypothesis(
        tokens=tokens,
        confidence=hyp.confidence * (0.9 ** len(vp_replacements)),  # Confidence penalty per VP
        description=f"VP parsing: {len(vp_replacements)} verb phrase(s) found",
        replacements=vp_replacements
    )
    
    return [vp_hyp]


def _try_parse_verb_phrase(tokens: List[VectorSpace], start_idx: int) -> Optional[tuple]:
    """
    Try to parse a verb phrase starting at the given position.
    
    Returns:
        Tuple of (vp_token, end_index) if successful, None otherwise
    """
    if start_idx >= len(tokens) or not is_verb(tokens[start_idx]):
        return None
    
    debug_print(f"Layer 4: Attempting VP parse starting at '{tokens[start_idx].word}'")
    
    # Simple approach: Look for verb + NP pattern
    verb_token = tokens[start_idx]
    
    # Look for NP tokens after the verb
    np_end_idx = -1
    for i in range(start_idx + 1, len(tokens)):
        token = tokens[i]
        # Check if this is an NP token
        if (token.isa("NP") or 
            (hasattr(token, 'word') and token.word and token.word.startswith("NP("))):
            np_end_idx = i
            debug_print(f"Layer 4: Found NP token at index {i}: {token.word}")
            break
    
    if np_end_idx != -1:
        # Found a verb + NP pattern
        end_idx = np_end_idx
        np_token = tokens[np_end_idx]
        
        # Create a VerbPhrase object
        vp = VerbPhrase()
        vp.verb = verb_token
        
        # Create a simple NounPhrase representation for the VP
        from engraf.pos.noun_phrase import NounPhrase
        simple_np = NounPhrase()
        
        # Extract details from the NP token
        if hasattr(np_token, 'word') and np_token.word and np_token.word.startswith("NP("):
            np_text = np_token.word[3:-1]  # Remove "NP(" and ")"
            words = np_text.split()
            if words:
                simple_np.head_noun = words[-1]  # Last word is usually the noun
                # Set the noun property for to_vector() compatibility
                from engraf.lexer.vector_space import VectorSpace
                simple_np.noun = VectorSpace(word=words[-1])
                simple_np.noun["noun"] = 1.0
                simple_np.vector = np_token
        
        vp.noun_phrase = simple_np
        
        # Create VP token text
        vp_words = [t.word for t in tokens[start_idx:end_idx + 1]]
        vp_text = " ".join(vp_words)
        
        # Build VP vector
        try:
            vp_vector = _build_vp_vector(vp, vp_text)
        except Exception as e:
            debug_print(f"Layer 4: Error in _build_vp_vector: {e}")
            debug_print(f"Layer 4: vp = {vp}")
            debug_print(f"Layer 4: vp.preps = {vp.preps}, type = {type(vp.preps)}")
            raise
        
        # Store the original VP object in the token for later extraction
        vp_vector._original_vp = vp
        
        debug_print(f"Layer 4: VP parsed successfully: '{vp_text}' (tokens {start_idx}-{end_idx})")
        return (vp_vector, end_idx)
    
    debug_print(f"Layer 4: No NP found after verb '{verb_token.word}'")
    return None


def _build_vp_vector(vp: VerbPhrase, vp_text: str) -> VectorSpace:
    """Build a VectorSpace token representing the parsed verb phrase."""
    
    debug_print(f"Layer 4: _build_vp_vector called with vp={vp}, vp_text='{vp_text}'")
    
    # Start with the verb's vector
    if vp.verb:
        debug_print(f"Layer 4: vp.verb = {vp.verb}, type = {type(vp.verb)}")
        try:
            if hasattr(vp.verb, 'to_vector'):
                debug_print(f"Layer 4: Calling to_vector() on vp.verb")
                base_vector = vp.verb.to_vector()
            elif isinstance(vp.verb, VectorSpace):
                debug_print(f"Layer 4: vp.verb is already a VectorSpace, copying")
                base_vector = vp.verb.copy()
            else:
                debug_print(f"Layer 4: Creating VectorSpace from vp.verb")
                base_vector = VectorSpace(vp.verb)
        except Exception as e:
            debug_print(f"Layer 4: Error with vp.verb processing: {e}")
            raise
    else:
        debug_print(f"Layer 4: No verb, creating VectorSpace from vp_text")
        base_vector = VectorSpace(vp_text)
    
    # Mark this as a VP token
    base_vector = base_vector.copy()
    base_vector.word = f"VP({vp_text})"
    base_vector["VP"] = 1.0
    base_vector["action"] = 1.0
    
    # Add semantic information from the verb
    if vp.verb:
        if hasattr(vp.verb, 'word'):
            verb_word = vp.verb.word if hasattr(vp.verb, 'word') else str(vp.verb)
        else:
            verb_word = str(vp.verb)
            
        # Add verb-specific semantics
        if verb_word in ["create", "make", "build"]:
            base_vector["create"] = 1.0
        elif verb_word in ["move", "translate", "go"]:
            base_vector["move"] = 1.0
        elif verb_word in ["rotate", "turn", "spin"]:
            base_vector["rotate"] = 1.0
        elif verb_word in ["delete", "remove", "destroy"]:
            base_vector["edit"] = 1.0  # Use "edit" dimension for delete operations
        elif verb_word in ["scale", "resize", "size"]:
            base_vector["scale"] = 1.0
    
    # Add object information if VP has a direct object
    if vp.noun_phrase:
        debug_print(f"Layer 4: vp.noun_phrase type: {type(vp.noun_phrase)}")
        if hasattr(vp.noun_phrase, 'vector') and vp.noun_phrase.vector:
            debug_print(f"Layer 4: Using noun_phrase.vector directly")
            obj_vector = vp.noun_phrase.vector
            debug_print(f"Layer 4: obj_vector type: {type(obj_vector)}")
            # Merge object properties into VP
            try:
                debug_print(f"Layer 4: Starting obj_vector iteration")
                from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                for dimension in VECTOR_DIMENSIONS:
                    value = obj_vector[dimension]
                    debug_print(f"Layer 4: Processing obj_vector dimension: {dimension} = {value}")
                    if dimension not in ["word"] and isinstance(value, (int, float)) and value != 0.0:
                        base_vector[dimension] = value
                debug_print(f"Layer 4: Finished obj_vector iteration")
            except Exception as e:
                debug_print(f"Layer 4: Error in obj_vector iteration: {e}")
        elif hasattr(vp.noun_phrase, 'to_vector'):
            debug_print(f"Layer 4: Calling to_vector() on noun_phrase")
            obj_vector = vp.noun_phrase.to_vector()
            debug_print(f"Layer 4: obj_vector type: {type(obj_vector)}")
            # Merge object properties into VP
            try:
                debug_print(f"Layer 4: Starting obj_vector iteration (to_vector)")
                from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                for dimension in VECTOR_DIMENSIONS:
                    value = obj_vector[dimension]
                    if dimension not in ["word"] and isinstance(value, (int, float)) and value != 0.0:
                        base_vector[dimension] = value
                debug_print(f"Layer 4: Finished obj_vector iteration (to_vector)")
            except Exception as e:
                debug_print(f"Layer 4: Error in obj_vector iteration (to_vector): {e}")
    
    # Add prepositional information
    if vp.preps:
        debug_print(f"Layer 4: vp.preps type: {type(vp.preps)}, value: {vp.preps}")
        try:
            prep_len = len(vp.preps)
            debug_print(f"Layer 4: len(vp.preps) = {prep_len}")
            base_vector["has_prep"] = 1.0
            base_vector["prep_count"] = prep_len
        except Exception as e:
            debug_print(f"Layer 4: Error with len(vp.preps): {e}")
    
    # Add amount/measurement information
    if vp.amount:
        base_vector["has_amount"] = 1.0
    
    debug_print(f"Layer 4: Built VP vector for '{vp_text}': {base_vector}")
    return base_vector


# Utility function for extracting verb phrases from hypotheses
def extract_verb_phrases(vp_hypotheses: List[TokenizationHypothesis]) -> List[VerbPhrase]:
    """Extract VerbPhrase objects from VP tokenization hypotheses."""
    verb_phrases = []
    
    for hyp in vp_hypotheses:
        for start_idx, end_idx, vp_token in hyp.vp_replacements:
            # For now, return a simple representation
            # In a full implementation, we'd reconstruct the VerbPhrase object
            vp = VerbPhrase()
            vp.verb = vp_token.word
            verb_phrases.append(vp)
    
    return verb_phrases
