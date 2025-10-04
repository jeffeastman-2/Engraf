#!/usr/bin/env python3
"""
LATN Layer 3: PrepositionalPhrase Token Replacement

This module implements Layer 3 of the LATN (Layered Augmented Transition Network) system.
Layer 3 replaces prepositional phrase constructions with single PrepositionalPhrase tokens.

Layer 3 builds on:
- Layer 1: Multi-hypothesis tokenization with morphological inflection

This layer identifies prepositional phrases like "in the red box", "on a very large sphere", "[1,2,3]"
and replaces them with single PrepositionalPhrase tokens containing the semantic meaning.
"""

from typing import List

from engraf.lexer.hypothesis import TokenizationHypothesis  

def generate_pp_attachment_combinations(layer3_hypotheses):
    """Generate all possible PP attachment combinations."""
    from copy import deepcopy
    from itertools import product
    
    all_combinations = []
    
    for hypothesis in layer3_hypotheses:
        # Find PP tokens and their possible attachment targets
        pp_positions = []
        attachment_options = []
        
        for i, token in enumerate(hypothesis.tokens):
            if token.isa("PP"):
                pp_positions.append(i)
                
                # Find all preceding NP/PP tokens as potential attachment targets
                targets = [None]  # None = no attachment
                for j in range(i):
                    prev_token = hypothesis.tokens[j]
                    if prev_token.isa("NP") or prev_token.isa("PP"):
                        targets.append(j)

                attachment_options.append(targets)  # None = no attachment

        if not pp_positions:
            # No PPs to attach, keep original hypothesis
            all_combinations.append(hypothesis)
            continue
        
        # Generate cartesian product of all attachment combinations
        for combination in product(*attachment_options):
            # Create new hypothesis with this attachment combination
            new_hypothesis = deepcopy(hypothesis)
            
            # Track which PP tokens will be removed (those that attach to something)
            tokens_to_remove = set()
            
            # Add attachment references and mark for removal if attached
            for pp_idx, target_idx in zip(pp_positions, combination):
                pp_token = new_hypothesis.tokens[pp_idx]                
                if target_idx is not None:  # PP attaches to something
                    target_token = new_hypothesis.tokens[target_idx]
                    # Handle attachment to a NP
                    if target_token.isa("NP"):
                        np_obj = target_token.phrase
                        np_obj.add_prepositional_phrase(pp_token.phrase)
                    # Handle attachment to a PP (attach to its NP)
                    elif target_token.isa("PP"):
                        pp_obj = target_token.phrase
                        np_obj = pp_obj.noun_phrase
                        np_obj.add_prepositional_phrase(pp_token.phrase)
                    # Remove the PP token since it's now bound for identification
                    tokens_to_remove.add(pp_idx)
            
            # Remove attached PP tokens from the token sequence
            new_hypothesis.tokens = [token for i, token in enumerate(new_hypothesis.tokens)
                                   if i not in tokens_to_remove]

            # Update confidence based on token complexity
            num_tokens = len(new_hypothesis.tokens)
            new_hypothesis.confidence =  hypothesis.confidence / num_tokens if num_tokens > 0 else hypothesis.confidence
                        
            all_combinations.append(new_hypothesis)
    
    return all_combinations

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

    layer3_hypotheses = generate_pp_attachment_combinations(layer3_hypotheses) 

    # Sort by confidence (highest first)
    layer3_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

    return layer3_hypotheses
