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
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.latn_tokenizer_layer2 import NPTokenizationHypothesis


@dataclass
class Layer2GroundingResult:
    """Result of Layer 2 semantic grounding operation."""
    success: bool
    confidence: float
    resolved_object: Optional[SceneObject] = None
    scene_object_phrase: Optional[SceneObjectPhrase] = None  # The converted SO phrase
    description: str = ""
    alternative_matches: List[Tuple[float, SceneObject]] = None
    
    def __post_init__(self):
        if self.alternative_matches is None:
            self.alternative_matches = []


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
        
        # Use SceneModel's find_noun_phrase method
        if return_all_matches:
            # Get all matching candidates with confidence scores
            candidates = self.scene_model.find_noun_phrase(np, return_all_matches=True)
            
            if not candidates:
                return Layer2GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"No scene objects match NP: {np}"
                )
            
            # Best match is the first one (highest confidence)
            best_confidence, best_object = candidates[0]
            
            # Create SceneObjectPhrase from the original NounPhrase
            scene_object_phrase = SceneObjectPhrase.from_noun_phrase(np)
            scene_object_phrase.resolve_to_scene_object(best_object)
            
            return Layer2GroundingResult(
                success=True,
                confidence=best_confidence,
                resolved_object=best_object,
                scene_object_phrase=scene_object_phrase,
                description=f"Grounded NP '{np}' to {best_object.object_id}",
                alternative_matches=candidates[1:]  # All except the best match
            )
        else:
            # Single best match
            matched_object = self.scene_model.find_noun_phrase(np, return_all_matches=False)
            
            if matched_object is None:
                return Layer2GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"No scene object matches NP: {np}"
                )
            
            # Calculate confidence based on semantic similarity
            if hasattr(np, 'vector') and np.vector:
                confidence = np.vector.semantic_similarity(matched_object.vector)
            else:
                confidence = 1.0  # Perfect name match without semantic constraints
            
            # Create SceneObjectPhrase from the original NounPhrase
            scene_object_phrase = SceneObjectPhrase.from_noun_phrase(np)
            scene_object_phrase.resolve_to_scene_object(matched_object)
            
            return Layer2GroundingResult(
                success=True,
                confidence=confidence,
                resolved_object=matched_object,
                scene_object_phrase=scene_object_phrase,
                description=f"Grounded NP '{np}' to {matched_object.object_id}"
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
    
    def extract_noun_phrases(self, layer2_hypotheses: List[NPTokenizationHypothesis]) -> List[NounPhrase]:
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
                if hasattr(token, '_original_np') and isinstance(token._original_np, NounPhrase):
                    # Check if this token has been grounded
                    if hasattr(token, '_grounded_phrase'):
                        # Use the grounded SceneObjectPhrase
                        noun_phrases.append(token._grounded_phrase)
                    else:
                        # Use the original NounPhrase (unbound)
                        noun_phrases.append(token._original_np)
        
        return noun_phrases
    
    def multiply_hypotheses_with_grounding(self, layer2_hypotheses: List[NPTokenizationHypothesis], 
                                         return_all_matches: bool = False) -> Tuple[List[NPTokenizationHypothesis], List[Layer2GroundingResult]]:
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
                if hasattr(token, '_original_np') and isinstance(token._original_np, NounPhrase):
                    np_positions.append(i)
                    np = token._original_np
                    
                    # Get all possible groundings for this NP
                    grounding_result = self.ground(np, return_all_matches=return_all_matches)
                    all_grounding_results.append(grounding_result)
                    
                    if grounding_result.success:
                        if return_all_matches and grounding_result.alternative_matches:
                            # Include best match + alternatives
                            options = [(grounding_result.confidence, grounding_result.resolved_object, grounding_result.scene_object_phrase)]
                            for conf, obj in grounding_result.alternative_matches:
                                # Create scene object phrase for alternative
                                alt_sop = SceneObjectPhrase.from_noun_phrase(np)
                                alt_sop.resolve_to_scene_object(obj)
                                options.append((conf, obj, alt_sop))
                            np_grounding_options.append(options)
                        else:
                            # Single best match
                            np_grounding_options.append([(grounding_result.confidence, grounding_result.resolved_object, grounding_result.scene_object_phrase)])
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
