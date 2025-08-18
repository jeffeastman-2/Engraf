#!/usr/bin/env python3
"""
LATN Layer 2 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 2 NounPhrase tokens.
It bridges between parsed NounPhrase structures and scene objects.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from engraf.pos.noun_phrase import NounPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject


@dataclass
class Layer2GroundingResult:
    """Result of Layer 2 semantic grounding operation."""
    success: bool
    confidence: float
    resolved_object: Optional[SceneObject] = None
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
            
            return Layer2GroundingResult(
                success=True,
                confidence=best_confidence,
                resolved_object=best_object,
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
            
            return Layer2GroundingResult(
                success=True,
                confidence=confidence,
                resolved_object=matched_object,
                description=f"Grounded NP '{np}' to {matched_object.object_id}"
            )
    
    def ground_multiple(self, np_list: List[NounPhrase], return_all_matches: bool = False) -> List[Layer2GroundingResult]:
        """Ground multiple NounPhrase tokens."""
        results = []
        for np in np_list:
            result = self.ground(np, return_all_matches=return_all_matches)
            
            # If grounding succeeds, resolve the NP to the found object
            if result.success and hasattr(np, 'resolve_to_scene_object'):
                np.resolve_to_scene_object(result.resolved_object)
            
            results.append(result)
        
        return results
