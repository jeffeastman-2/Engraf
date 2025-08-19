#!/usr/bin/env python3
"""
LATN Layer 3 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 3 PrepositionalPhrase tokens.
It bridges between parsed PrepositionalPhrase structures and spatial locations/relationships.
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder


@dataclass
class Layer3GroundingResult:
    """Result of Layer 3 semantic grounding operation."""
    success: bool
    confidence: float
    resolved_object: Optional[Union[SceneObject, VectorSpace]] = None
    description: str = ""
    alternative_matches: List[Tuple[float, Union[SceneObject, VectorSpace]]] = None
    
    def __post_init__(self):
        if self.alternative_matches is None:
            self.alternative_matches = []


class Layer3SemanticGrounder:
    """Semantic grounding for LATN Layer 3 PrepositionalPhrase tokens."""
    
    def __init__(self, scene_model: SceneModel):
        self.scene_model = scene_model
        self.layer2_grounder = Layer2SemanticGrounder(scene_model)
    
    def ground(self, pp: PrepositionalPhrase, return_all_matches: bool = False) -> Layer3GroundingResult:
        """Ground a PrepositionalPhrase to spatial locations or object relationships.
        
        Args:
            pp: The PrepositionalPhrase to ground
            return_all_matches: If True, return all possible spatial interpretations
            
        Returns:
            Layer3GroundingResult with resolved spatial location/relationship
        """
        if not isinstance(pp, PrepositionalPhrase):
            return Layer3GroundingResult(
                success=False,
                confidence=0.0,
                description=f"Expected PrepositionalPhrase, got {type(pp).__name__}"
            )
        
        # Handle different types of prepositional phrases
        if hasattr(pp, 'vector_text') and pp.vector_text:
            # Vector coordinates like "at [1,2,3]"
            return self._ground_vector_location(pp)
        elif hasattr(pp, 'preposition') and hasattr(pp, 'noun_phrase') and pp.noun_phrase:
            # Spatial relationships like "on the table", "above the red box"
            return self._ground_spatial_relationship(pp, return_all_matches)
        else:
            return Layer3GroundingResult(
                success=False,
                confidence=0.0,
                description=f"Cannot ground PP: {pp} (missing vector or NP object)"
            )
    
    def _ground_vector_location(self, pp: PrepositionalPhrase) -> Layer3GroundingResult:
        """Ground a prepositional phrase with vector coordinates."""
        try:
            # Extract coordinates from vector_text
            if hasattr(pp, 'vector') and pp.vector:
                # Use the PP's computed vector as the spatial location
                location_vector = VectorSpace()
                # Copy the vector data using proper VectorSpace iteration
                from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                for dim in VECTOR_DIMENSIONS:
                    value = pp.vector[dim]
                    if value != 0.0:
                        location_vector[dim] = value
                location_vector.word = f"Location({pp.vector_text})"
                
                return Layer3GroundingResult(
                    success=True,
                    confidence=1.0,
                    resolved_object=location_vector,
                    description=f"Grounded PP '{pp}' to absolute location {pp.vector_text}"
                )
            else:
                return Layer3GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"PP has vector_text but no computed vector: {pp}"
                )
        except Exception as e:
            return Layer3GroundingResult(
                success=False,
                confidence=0.0,
                description=f"Failed to parse vector coordinates: {e}"
            )
    
    def _ground_spatial_relationship(self, pp: PrepositionalPhrase, return_all_matches: bool = False) -> Layer3GroundingResult:
        """Ground a prepositional phrase with spatial relationship to an object."""
        # First, ground the noun phrase within the PP
        if hasattr(pp, 'noun_phrase') and pp.noun_phrase:
            # Use Layer 2 grounding for the contained NP
            np_grounding = self.layer2_grounder.ground(pp.noun_phrase, return_all_matches=return_all_matches)
            
            if not np_grounding.success:
                return Layer3GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"Failed to ground NP within PP: {pp.noun_phrase}"
                )
            
            # Create spatial relationship vector
            spatial_vector = VectorSpace()
            
            # Add preposition semantics
            if hasattr(pp, 'preposition') and pp.preposition:
                # Look up preposition in vocabulary for spatial semantics
                from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
                if pp.preposition in SEMANTIC_VECTOR_SPACE:
                    prep_vector = SEMANTIC_VECTOR_SPACE[pp.preposition]
                    spatial_vector += prep_vector
            
            # Add object reference
            if hasattr(np_grounding.resolved_object, 'vector'):
                spatial_vector += np_grounding.resolved_object.vector
            
            spatial_vector.word = f"SpatialRel({pp.preposition} {np_grounding.resolved_object.object_id})"
            
            # Store reference to the grounded object for spatial calculations
            spatial_vector._reference_object = np_grounding.resolved_object
            spatial_vector._preposition = pp.preposition
            
            result = Layer3GroundingResult(
                success=True,
                confidence=np_grounding.confidence,
                resolved_object=spatial_vector,
                description=f"Grounded PP '{pp}' to spatial relationship: {pp.preposition} {np_grounding.resolved_object.object_id}"
            )
            
            if return_all_matches and np_grounding.alternative_matches:
                # Create alternative spatial relationships for each alternative NP match
                alternatives = []
                for alt_confidence, alt_object in np_grounding.alternative_matches:
                    alt_spatial = VectorSpace()
                    # Copy the vector data instead of using update()
                    for dim, value in spatial_vector.items():
                        if value != 0.0:
                            alt_spatial[dim] = value
                    alt_spatial.word = f"SpatialRel({pp.preposition} {alt_object.object_id})"
                    alt_spatial._reference_object = alt_object
                    alt_spatial._preposition = pp.preposition
                    alternatives.append((alt_confidence, alt_spatial))
                result.alternative_matches = alternatives
            
            return result
        else:
            return Layer3GroundingResult(
                success=False,
                confidence=0.0,
                description=f"PP has no NP to ground: {pp}"
            )
    
    def ground_multiple(self, pp_list: List[PrepositionalPhrase], return_all_matches: bool = False) -> List[Layer3GroundingResult]:
        """Ground multiple PrepositionalPhrase tokens."""
        results = []
        for pp in pp_list:
            result = self.ground(pp, return_all_matches=return_all_matches)
            
            # If grounding succeeds, store the spatial relationship
            if result.success and hasattr(pp, 'resolve_to_spatial_location'):
                pp.resolve_to_spatial_location(result.resolved_object)
            
            results.append(result)
        
        return results
