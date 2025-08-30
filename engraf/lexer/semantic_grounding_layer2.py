#!/usr/bin/env python3
"""
LATN Layer 2 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 2 NounPhrase tokens.
It bridges between parsed NounPhrase structures and scene objects.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import copy
from itertools import product

from engraf.pos.noun_phrase import NounPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.hypothesis import TokenizationHypothesis


@dataclass
class Layer2GroundingResult:
    """Result of Layer 2 semantic grounding operation."""
    success: bool
    confidence: float
    resolved_objects: List[SceneObject] = None  # Changed to list to support plural pronouns
    grounded_noun_phrase: Optional[NounPhrase] = None  # The NP with grounding info added
    description: str = ""
    alternative_matches: List[Tuple[float, SceneObject]] = None
    
    def __post_init__(self):
        if self.resolved_objects is None:
            self.resolved_objects = []
        if self.alternative_matches is None:
            self.alternative_matches = []
    
    @property
    def resolved_object(self):
        """Backward compatibility: return first object if any."""
        return self.resolved_objects[0] if self.resolved_objects else None


class Layer2SemanticGrounder:
    """Semantic grounding for LATN Layer 2 NounPhrase tokens."""
    
    def __init__(self, scene_model: SceneModel):
        self.scene_model = scene_model
    
    def ground(self, np: NounPhrase, return_all_matches: bool = False) -> Layer2GroundingResult:
        """Ground a NounPhrase to scene objects using SceneModel.
        
        Args:
            np: The NounPhrase to ground
            return_all_matches: If True, return all possible matches with confidence scores
            
        Returns:
            Layer2GroundingResult with resolved object(s) and confidence information
        """
        if not isinstance(np, NounPhrase):
            return Layer2GroundingResult(
                success=False,
                confidence=0.0,
                description=f"Expected NounPhrase, got {type(np).__name__}"
            )
        
        # Handle pronoun NPs using specialized pronoun resolution
        if hasattr(np, 'pronoun') and np.pronoun:
            from engraf.visualizer.scene.scene_model import resolve_pronoun
            try:
                resolved_objects = resolve_pronoun(np.pronoun, self.scene_model)
                if not resolved_objects:
                    return Layer2GroundingResult(
                        success=False,
                        confidence=0.0,
                        description=f"No objects found for pronoun '{np.pronoun}'"
                    )
                
                # Handle singular vs plural pronouns based on grammatical features
                if np.vector.isa("singular"):
                    # Singular pronouns: take the most recent/relevant single object
                    resolved_object_list = [resolved_objects[0]]  # Single object in list
                    confidence = 1.0  # High confidence for singular reference
                    description = f"Resolved singular pronoun '{np.pronoun}' to {resolved_objects[0].object_id}"
                    pronoun_type = 'singular'
                elif np.vector.isa("plural"):
                    # Plural pronouns: ground to all resolved objects equally
                    resolved_object_list = resolved_objects  # All objects
                    confidence = 1.0  # High confidence for plural reference to known objects
                    description = f"Resolved plural pronoun '{np.pronoun}' to {len(resolved_objects)} objects: {[obj.object_id for obj in resolved_objects]}"
                    pronoun_type = 'plural'
                else:
                    # Unknown grammatical number - fallback
                    resolved_object_list = [resolved_objects[0]]  # Single object fallback
                    confidence = 0.5
                    description = f"Resolved pronoun '{np.pronoun}' (unknown number) to {resolved_objects[0].object_id}"
                    pronoun_type = 'unknown'
                
                # Add grounding information to the NounPhrase
                grounded_np = copy.deepcopy(np)
                grounding_info = {
                    'scene_objects': resolved_object_list,  # Store all resolved objects
                    'confidence': confidence,
                    'type': 'pronoun_resolution',
                    'pronoun_type': pronoun_type
                }
                
                # Keep backward compatibility with scene_object (singular)
                if resolved_object_list:
                    grounding_info['scene_object'] = resolved_object_list[0]
                
                grounded_np.grounding = grounding_info
                
                return Layer2GroundingResult(
                    success=True,
                    confidence=confidence,
                    resolved_objects=resolved_object_list,  # Use the new list field
                    grounded_noun_phrase=grounded_np,
                    description=description,
                    alternative_matches=[]  # No alternatives needed with new approach
                )
                
                return Layer2GroundingResult(
                    success=True,
                    confidence=confidence,
                    resolved_object=best_object,
                    grounded_noun_phrase=grounded_np,
                    description=description,
                    alternative_matches=[]  # No alternatives for pronouns
                )
                
            except ValueError as e:
                return Layer2GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"Pronoun resolution failed: {e}"
                )
        
        # Handle regular NPs using SceneModel - get all matching objects
        candidates = self.scene_model.find_noun_phrase(np, return_all_matches=True)
        
        if not candidates:
            return Layer2GroundingResult(
                success=False,
                confidence=0.0,
                description=f"No scene objects match NP: {np}"
            )
        
        # Determine if this NP should ground to multiple objects
        if np.vector.isa("plural"):
            # Ground to ALL matching objects (e.g., "all the red objects", "the cubes")
            resolved_object_list = [obj for conf, obj in candidates]
            avg_confidence = sum(conf for conf, obj in candidates) / len(candidates)
            object_ids = [obj.object_id for obj in resolved_object_list]
            description = f"Grounded NP '{np}' to {len(resolved_object_list)} objects: {object_ids}"
        else:
            # Ground to the best single match (e.g., "the cube", "a red object")
            best_confidence, best_object = candidates[0]
            resolved_object_list = [best_object]
            avg_confidence = best_confidence
            description = f"Grounded NP '{np}' to {best_object.object_id}"
        
        # Add grounding information directly to the NounPhrase
        grounded_np = copy.deepcopy(np)
        grounding_info = {
            'scene_objects': resolved_object_list,  # Store all resolved objects
            'confidence': avg_confidence,
            'type': 'scene_object',
            'multiple_objects': np.vector.isa("plural")
        }
        
        # Keep backward compatibility with scene_object (singular)
        if resolved_object_list:
            grounding_info['scene_object'] = resolved_object_list[0]
        
        grounded_np.grounding = grounding_info
        
        return Layer2GroundingResult(
            success=True,
            confidence=avg_confidence,
            resolved_objects=resolved_object_list,
            grounded_noun_phrase=grounded_np,
            description=description,
            alternative_matches=[]  # No longer needed with new approach
        )
    
    def ground_multiple(self, np_list: List[NounPhrase], return_all_matches: bool = False) -> List[Layer2GroundingResult]:
        """Ground multiple NounPhrase tokens.
        
        Returns:
            List of Layer2GroundingResult objects, each containing a SceneObjectPhrase if successful
        """
        results = []
        for np in np_list:
            result = self.ground(np, return_all_matches=return_all_matches)
            results.append(result)
        
        return results
    
    def extract_noun_phrases(self, layer2_hypotheses: List[TokenizationHypothesis]) -> List[NounPhrase]:
        """Extract NounPhrase objects from Layer 2 processing.
        
        The token stream is the single source of truth - NP tokens are already
        properly placed in the tokens list by replace_np_sequences().
        
        Clean semantics:
        - If grounding was enabled: Return SceneObjectPhrase for grounded NPs + NounPhrase for unbound NPs
        - If grounding was disabled: Return only NounPhrase objects (all NPs)
        """
        noun_phrases = []
        
        for i, hypothesis in enumerate(layer2_hypotheses):
            # Look for NP tokens in the hypothesis token stream
            for j, token in enumerate(hypothesis.tokens):
                if token._original_np is not None and isinstance(token._original_np, NounPhrase):
                    # Check if this token has been grounded
                    if token._grounded_phrase is not None:
                        # Use the grounded SceneObjectPhrase
                        noun_phrases.append(token._grounded_phrase)
                    else:
                        # Use the original NounPhrase (unbound)
                        noun_phrases.append(token._original_np)
        
        return noun_phrases
    
    def multiply_hypotheses_with_grounding(self, layer2_hypotheses: List[TokenizationHypothesis], 
                                         return_all_matches: bool = False) -> Tuple[List[TokenizationHypothesis], List[Layer2GroundingResult]]:
        """Multiply hypotheses based on grounding results using two-pass algorithm.
        
        Pass 1: Collect all scene object matches for each NP in each hypothesis
        Pass 2: Generate combinatorial hypotheses with one object per NP
        
        Returns:
            tuple: (grounded_hypotheses, all_grounding_results)
        """
        all_grounded_hypotheses = []
        all_grounding_results = []
        
        for hypothesis in layer2_hypotheses:
            # Pass 1: Collect all potential groundings for each NP in this hypothesis
            np_grounding_options = []  # List of lists: [[obj1, obj2], [obj3], ...]
            np_positions = []  # Track which tokens are NPs
            
            for i, token in enumerate(hypothesis.tokens):
                if token._original_np is not None and isinstance(token._original_np, NounPhrase):
                    np_positions.append(i)
                    np = token._original_np
                    
                    # Get all possible groundings for this NP
                    grounding_result = self.ground(np, return_all_matches=return_all_matches)
                    all_grounding_results.append(grounding_result)
                    
                    if grounding_result.success:
                        if return_all_matches and grounding_result.alternative_matches:
                            # Include best match + alternatives
                            options = [(grounding_result.confidence, grounding_result.resolved_object, grounding_result.grounded_noun_phrase)]
                            for conf, obj in grounding_result.alternative_matches:
                                # Create grounded noun phrase for alternative
                                alt_np = copy.deepcopy(np)
                                alt_np.grounding = {
                                    'scene_object': obj,
                                    'confidence': conf,
                                    'type': 'scene_object'
                                }
                                options.append((conf, obj, alt_np))
                            np_grounding_options.append(options)
                        else:
                            # Single best match
                            np_grounding_options.append([(grounding_result.confidence, grounding_result.resolved_object, grounding_result.grounded_noun_phrase)])
                    else:
                        # No grounding found - use original NP
                        np_grounding_options.append([(0.5, None, np)])  # Ungrounded gets neutral score
            
            # Pass 2: Generate combinatorial hypotheses
            if np_grounding_options:
                # Generate all combinations of groundings
                for combo in product(*np_grounding_options):
                    # Create new hypothesis for this grounding combination
                    new_hypothesis = copy.deepcopy(hypothesis)
                    
                    # Apply the grounding combination
                    combo_confidence = 1.0
                    for np_idx, (conf, scene_obj, grounded_phrase) in enumerate(combo):
                        token_idx = np_positions[np_idx]
                        token = new_hypothesis.tokens[token_idx]
                        
                        # Attach grounded phrase to token
                        token._grounded_phrase = grounded_phrase
                        
                        # Update token's vector to have grounded information if it's a grounded NP
                        if hasattr(grounded_phrase, 'grounding') and grounded_phrase.grounding:
                            # Copy the grounded phrase vector but preserve original token dimensions
                            original_vector = token.vector.copy()
                            token.vector = grounded_phrase.vector.copy()
                            
                            # Preserve the NP dimension to maintain token type
                            from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                            np_index = VECTOR_DIMENSIONS.index('NP')
                            if original_vector[np_index] > 0:  # If original token was an NP
                                token.vector[np_index] = original_vector[np_index]  # Preserve NP dimension
                        
                        # Update confidence
                        combo_confidence *= conf
                    
                    # Update hypothesis confidence
                    new_hypothesis.confidence = hypothesis.confidence * combo_confidence
                    all_grounded_hypotheses.append(new_hypothesis)
            else:
                # No NPs to ground - keep original hypothesis
                all_grounded_hypotheses.append(hypothesis)
        
        # Sort hypotheses by confidence (best first)
        all_grounded_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return all_grounded_hypotheses, all_grounding_results
