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
from engraf.lexer.latn_tokenizer_layer3 import PPTokenizationHypothesis


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
    
    def process_pp_attachments(self, layer3_hypotheses, return_all_matches: bool = False):
        """Two-pass PP attachment resolution with spatial validation.
        
        Args:
            layer3_hypotheses: Hypotheses with PP tokens to process
            return_all_matches: Whether to return all valid combinations
            
        Returns:
            Processed hypotheses with validated PP attachments
        """
        # Pass 1: Generate all possible PP attachment combinations
        attachment_hypotheses = self._generate_pp_attachment_combinations(layer3_hypotheses)
        print(f"🔀 Pass 1: Generated {len(attachment_hypotheses)} PP attachment combinations")
        
        # Pass 2: Spatial validation filter
        validated_hypotheses = self._validate_spatial_attachments(attachment_hypotheses)
        print(f"✅ Pass 2: {len(validated_hypotheses)} spatially valid combinations")
        
        return validated_hypotheses if validated_hypotheses else layer3_hypotheses
    
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
    
    def _generate_pp_attachment_combinations(self, layer3_hypotheses):
        """Pass 1: Generate all possible PP attachment combinations."""
        from copy import deepcopy
        from itertools import product
        
        all_combinations = []
        
        for hypothesis in layer3_hypotheses:
            # Find PP tokens and their possible attachment targets
            pp_positions = []
            attachment_options = []
            
            for i, token in enumerate(hypothesis.tokens):
                if hasattr(token, 'word') and token.word and token.word.startswith('PP('):
                    pp_positions.append(i)
                    
                    # Find all preceding NP/PP tokens as potential attachment targets
                    targets = []
                    for j in range(i):
                        prev_token = hypothesis.tokens[j]
                        if (hasattr(prev_token, 'word') and prev_token.word and 
                            (prev_token.word.startswith('NP(') or prev_token.word.startswith('PP('))):
                            targets.append(j)
                    
                    attachment_options.append(targets if targets else [None])  # None = no attachment
            
            if not pp_positions:
                # No PPs to attach, keep original hypothesis
                all_combinations.append(hypothesis)
                continue
            
            # Generate cartesian product of all attachment combinations
            for combination in product(*attachment_options):
                # Create new hypothesis with this attachment combination
                new_hypothesis = deepcopy(hypothesis)
                
                # Add attachment metadata to PP tokens
                for pp_idx, target_idx in zip(pp_positions, combination):
                    pp_token = new_hypothesis.tokens[pp_idx]
                    
                    # Add attachment information
                    if not hasattr(pp_token, '_attachment_info'):
                        pp_token._attachment_info = {}
                    pp_token._attachment_info['attaches_to'] = target_idx
                    pp_token._attachment_info['combination_id'] = str(combination)
                
                # Update confidence based on attachment complexity
                attachment_penalty = len([t for t in combination if t is not None]) * 0.05
                new_hypothesis.confidence = max(0.1, hypothesis.confidence - attachment_penalty)
                
                all_combinations.append(new_hypothesis)
        
        return all_combinations
    
    def _validate_spatial_attachments(self, attachment_hypotheses):
        """Pass 2: Validate PP attachments using spatial reasoning."""
        validated = []
        
        for hypothesis in attachment_hypotheses:
            is_valid = True
            validation_scores = []
            
            # Check each PP attachment for spatial validity
            for token in hypothesis.tokens:
                if (hasattr(token, '_attachment_info') and 
                    hasattr(token, 'word') and token.word and token.word.startswith('PP(')):
                    
                    # Extract preposition and noun phrase from PP token
                    pp_content = token.word[3:-1]  # Remove 'PP(' and ')'
                    prep, np_part = pp_content.split(' ', 1) if ' ' in pp_content else (pp_content, '')
                    
                    # Get attachment target
                    target_idx = token._attachment_info.get('attaches_to')
                    
                    # Apply prep-specific spatial validation
                    spatial_score = self._validate_prep_spatial_relationship(prep, np_part, target_idx, hypothesis)
                    validation_scores.append(spatial_score)
                    
                    print(f"🔍 Spatial validation: PP '{prep} {np_part}' → score {spatial_score:.2f}")
                    
                    # Filter out impossible relationships (score < 0.3 for stricter filtering)
                    if spatial_score < 0.3:
                        print(f"❌ Filtering out combination due to low spatial score: {spatial_score:.2f}")
                        is_valid = False
                        break
            
            if is_valid and validation_scores:
                # Update hypothesis confidence based on spatial validation
                avg_spatial_score = sum(validation_scores) / len(validation_scores)
                hypothesis.confidence = hypothesis.confidence * avg_spatial_score
                validated.append(hypothesis)
        
        # Sort by confidence (best first)
        validated.sort(key=lambda h: h.confidence, reverse=True)
        return validated
    
    def _validate_prep_spatial_relationship(self, prep: str, np_part: str, target_idx, hypothesis) -> float:
        """Validate a specific prepositional relationship using spatial reasoning."""
        if not self.scene_model or target_idx is None:
            print(f"🔍 No scene model or target: scene={bool(self.scene_model)}, target_idx={target_idx}")
            return 0.5  # Neutral score if no scene or no attachment
        
        # Get scene objects involved in the relationship
        try:
            # Extract object names from the PP and target
            pp_object_name = self._extract_object_name_from_np(np_part)
            target_token = hypothesis.tokens[target_idx]
            target_object_name = self._extract_object_name_from_token(target_token)
            
            print(f"🔍 PP object: '{pp_object_name}', Target object: '{target_object_name}'")
            
            if not pp_object_name or not target_object_name:
                print(f"🔍 Missing object names: pp_object='{pp_object_name}', target_object='{target_object_name}'")
                return 0.5  # Can't determine objects
            
            # Find actual scene objects
            pp_object = self._find_scene_object(pp_object_name)
            target_object = self._find_scene_object(target_object_name)
            
            print(f"🔍 Found scene objects: pp_object={bool(pp_object)}, target_object={bool(target_object)}")
            if pp_object:
                print(f"🔍 PP object position: {pp_object.position if hasattr(pp_object, 'position') else pp_object.vector}")
            if target_object:
                print(f"🔍 Target object position: {target_object.position if hasattr(target_object, 'position') else target_object.vector}")
            
            if not pp_object or not target_object:
                print(f"🔍 Objects not found in scene")
                return 0.3  # Objects not found in scene
            
            # Apply prep-specific spatial tests
            score = self._apply_prep_spatial_test(prep, target_object, pp_object)
            print(f"🔍 Final spatial score for '{prep}' between '{target_object_name}' and '{pp_object_name}': {score}")
            return score
            
        except Exception as e:
            print(f"🔍 Exception in validation: {e}")
            return 0.2  # Error in validation
    
    def _apply_prep_spatial_test(self, prep: str, obj1, obj2) -> float:
        """Apply prep-specific spatial relationship test using shared utilities.
        
        Args:
            prep: Preposition (e.g., 'on', 'under', 'beside')
            obj1: Target object that should be positioned relative to obj2
            obj2: Reference object (PP object) 
            
        For "box on table": obj1=box, obj2=table, validates box is on table
        """
        from engraf.utils.spatial_validation import SpatialValidator
        # For spatial validation: obj2 (PP object) is reference, obj1 (target) is positioned relative to it
        return SpatialValidator.validate_spatial_relationship(prep, obj2, obj1, use_preposition_vector=True)
    
    def _extract_object_name_from_np(self, np_part: str):
        """Extract object name from noun phrase text."""
        words = np_part.lower().split()
        object_words = ['box', 'table', 'pyramid', 'sphere', 'cube', 'ball']
        for word in words:
            if word in object_words:
                return word
        return None
    
    def _extract_object_name_from_token(self, token):
        """Extract object name from a token."""
        if hasattr(token, 'word') and token.word:
            if token.word.startswith('NP(') or token.word.startswith('PP('):
                content = token.word[3:-1]  # Remove prefix and parentheses
                return self._extract_object_name_from_np(content)
        return None
    
    def _find_scene_object(self, object_name: str):
        """Find scene object by name."""
        if not self.scene_model:
            return None
        for obj in self.scene_model.objects:
            if hasattr(obj, 'word') and obj.word == object_name:
                return obj
            if hasattr(obj, 'object_id') and object_name in obj.object_id:
                return obj
        return None
    
    def extract_prepositional_phrases(self, layer3_hypotheses: List[PPTokenizationHypothesis]) -> List[PrepositionalPhrase]:
        """Extract PrepositionalPhrase objects from Layer 3 hypotheses.
        
        Args:
            layer3_hypotheses: List of Layer 3 tokenization hypotheses
            
        Returns:
            List of PrepositionalPhrase objects found in the hypotheses
        """
        prepositional_phrases = []
        
        for hypothesis in layer3_hypotheses:
            for token in hypothesis.tokens:
                if hasattr(token, '_original_pp') and isinstance(token._original_pp, PrepositionalPhrase):
                    prepositional_phrases.append(token._original_pp)
        
        return prepositional_phrases
