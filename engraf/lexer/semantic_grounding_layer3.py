#!/usr/bin/env python3
"""
LATN Layer 3 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 3 PrepositionalPhrase tokens.
It bridges between parsed PrepositionalPhrase structures and spatial locations/relationships.
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.hypothesis import TokenizationHypothesis


@dataclass
class Layer3GroundingResult:
    """Result of Layer 3 semantic grounding operation."""
    success: bool
    confidence: float
    resolved_object: Optional[VectorSpace] = None
    description: str = ""
    alternative_matches: List[Tuple[float, VectorSpace]] = None
    
    def __post_init__(self):
        if self.alternative_matches is None:
            self.alternative_matches = []


class Layer3SemanticGrounder:
    """Semantic grounding for LATN Layer 3 PrepositionalPhrase tokens."""
    
    def __init__(self):
        """Initialize Layer 3 grounding - no scene model needed, uses Layer 2 results."""
        pass
    
    def process_pp_attachments(self, layer3_hypotheses, return_all_matches: bool = False):
        """Two-pass PP attachment resolution with spatial validation and semantic grounding.
        
        Args:
            layer3_hypotheses: Hypotheses with PP tokens to process
            return_all_matches: Whether to return all valid combinations
            
        Returns:
            Processed hypotheses with validated PP attachments and semantic grounding
        """
        # Pass 1: Generate all possible PP attachment combinations
        attachment_hypotheses = self._generate_pp_attachment_combinations(layer3_hypotheses)
        print(f"🔀 Pass 1: Generated {len(attachment_hypotheses)} PP attachment combinations")
        
        # Pass 2: Spatial validation filter
        validated_hypotheses = self._validate_spatial_attachments(attachment_hypotheses)
        print(f"✅ Pass 2: {len(validated_hypotheses)} spatially valid combinations")
        
        # Pass 3: Semantic grounding for validated attachments
        if validated_hypotheses:
            # For now, return the validated hypotheses since PP merging already happened in validation
            # The _merge_ppso_into_np method in _validate_spatial_attachments already did the work
            print(f"🎯 Pass 3: Returning validated hypotheses with merged PPs")
            return validated_hypotheses
        
        return layer3_hypotheses
    
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
        if pp.vector and pp.vector.isa("vector"):
            # Vector coordinates like "at [1,2,3]"
            return self._ground_vector_location(pp)
        elif pp.preposition is not None and pp.noun_phrase is not None and pp.noun_phrase:
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
            # Extract coordinates from the PP's vector
            if pp.vector is not None and pp.vector:
                # Use the PP's computed vector as the spatial location
                location_vector = VectorSpace()
                # Copy the vector data using proper VectorSpace iteration
                from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                for dim in VECTOR_DIMENSIONS:
                    value = pp.vector[dim]
                    if value != 0.0:
                        location_vector[dim] = value
                location_vector.word = f"Location({pp.preposition})"
                
                return Layer3GroundingResult(
                    success=True,
                    confidence=1.0,
                    resolved_object=location_vector,
                    description=f"Grounded PP '{pp}' to absolute location from vector"
                )
            else:
                return Layer3GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"PP has no computed vector: {pp}"
                )
        except Exception as e:
            return Layer3GroundingResult(
                success=False,
                confidence=0.0,
                description=f"Failed to parse vector coordinates: {e}"
            )
    
    def _ground_spatial_relationship(self, pp: PrepositionalPhrase, return_all_matches: bool = False) -> Layer3GroundingResult:
        """Process a prepositional phrase with spatial relationship - requires grounded Scene Objects."""
        
        # Check if the PP contains grounded scene objects
        # PP grounding should only succeed if the NP within the PP is grounded to a Scene Object
        has_grounded_object = False
        if pp.noun_phrase is not None and pp.noun_phrase:
            np = pp.noun_phrase
            if hasattr(np, 'grounding') and np.grounding is not None and np.grounding.get('scene_object') is not None:
                has_grounded_object = True
                
        if not has_grounded_object:
            return Layer3GroundingResult(
                success=False,
                confidence=0.0,
                resolved_object=None,
                description=f"Failed to ground NP within PP '{pp.preposition}' - no scene object found"
            )
        
        # Create spatial relationship vector
        spatial_vector = VectorSpace()
        
        # Add preposition semantics
        if pp.preposition is not None and pp.preposition:
            # Look up preposition in vocabulary for spatial semantics
            from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
            
            prep_key = pp.preposition.lower()
            if prep_key in SEMANTIC_VECTOR_SPACE:
                prep_vector = SEMANTIC_VECTOR_SPACE[prep_key]
                spatial_vector += prep_vector
        
        # Add spatial location properties if available
        if pp.spatial_location is not None and pp.spatial_location:
            spatial_vector.spatial_location = pp.spatial_location
            
        # Add coordinate properties if available  
        if pp.locX is not None:
            spatial_vector.locX = pp.locX
        if pp.locY is not None:
            spatial_vector.locY = pp.locY
        if pp.locZ is not None:
            spatial_vector.locZ = pp.locZ
            
        # Get the actual scene object for reference
        scene_object = pp.noun_phrase.grounding['scene_object'] if hasattr(pp.noun_phrase, 'grounding') and pp.noun_phrase.grounding else None
        scene_obj_id = getattr(scene_object, 'object_id', 'unknown')
        
        # Add reference to the scene object and preposition
        spatial_vector._reference_object = scene_object
        spatial_vector._preposition = pp.preposition
            
        return Layer3GroundingResult(
            success=True,
            confidence=0.8,
            resolved_object=spatial_vector,
            description=f"Processed spatial PP: {pp.preposition} (grounded to {scene_obj_id})"
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
                    if pp_token._attachment_info is None:
                        pp_token._attachment_info = {}
                    pp_token._attachment_info['attaches_to'] = target_idx
                    pp_token._attachment_info['combination_id'] = str(combination)
                
                # Update confidence based on attachment complexity
                attachment_penalty = len([t for t in combination if t is not None]) * 0.05
                new_hypothesis.confidence = max(0.1, hypothesis.confidence - attachment_penalty)
                
                all_combinations.append(new_hypothesis)
        
        return all_combinations
    
    def _validate_spatial_attachments(self, attachment_hypotheses):
        """Pass 2: Validate PP attachments using spatial reasoning and immediately merge valid PPSOs."""
        validated = []
        
        for hypothesis in attachment_hypotheses:
            # First pass: validate all PP attachments and collect merge information
            pp_validations = []  # List of (pp_token_idx, target_idx, spatial_score, should_merge)
            
            for i, token in enumerate(hypothesis.tokens):
                if (hasattr(token, '_attachment_info') and token._attachment_info is not None and 
                    hasattr(token, 'word') and token.word and token.word.startswith('PP(')):
                    
                    # Get attachment target
                    target_idx = token._attachment_info.get('attaches_to')
                    
                    # Apply prep-specific spatial validation using the PP token
                    spatial_score = self._validate_prep_spatial_relationship(token, target_idx, hypothesis)
                    print(f"🔍 Spatial validation: PP '{token.word}' → score {spatial_score:.2f}")
                    
                    should_merge = spatial_score >= 0.3 and target_idx is not None
                    pp_validations.append((i, target_idx, spatial_score, should_merge))
                    
                    if not should_merge and spatial_score < 0.3:
                        print(f"❌ PP attachment failed spatial validation: {spatial_score:.2f} - leaving in hypothesis")
            
            # Process valid PP attachments (don't require all to be valid)
            if pp_validations:
                # Second pass: perform merges for valid attachments only
                tokens_to_remove = set()
                valid_attachments = 0
                
                for pp_idx, target_idx, spatial_score, should_merge in pp_validations:
                    if should_merge:
                        pp_token = hypothesis.tokens[pp_idx]
                        target_token = hypothesis.tokens[target_idx]
                        
                        # Merge PP into target NP
                        self._merge_ppso_into_np(target_token, pp_token)
                        tokens_to_remove.add(pp_idx)
                        valid_attachments += 1
                        print(f"✅ Merged and removing PP: {pp_token.word}")
                
                # Third pass: create new token list without consumed PPs
                new_tokens = [token for i, token in enumerate(hypothesis.tokens) if i not in tokens_to_remove]
                hypothesis.tokens = new_tokens
                
                if valid_attachments > 0:
                    print(f"🔧 Processed {valid_attachments} valid PP attachments, {len(new_tokens)} tokens remain")
                    # Update hypothesis confidence based on valid spatial validations only
                    valid_scores = [validation[2] for validation in pp_validations if validation[3]]
                    if valid_scores:
                        avg_spatial_score = sum(valid_scores) / len(valid_scores)
                        hypothesis.confidence = hypothesis.confidence * avg_spatial_score
                else:
                    print(f"🔧 No valid PP attachments found, keeping original hypothesis")
                
                validated.append(hypothesis)
            else:
                # No PP validations to process - keep original hypothesis
                validated.append(hypothesis)
        
        # Sort by confidence (best first)
        validated.sort(key=lambda h: h.confidence, reverse=True)
        return validated
    
    def _ground_validated_attachments(self, validated_hypotheses):
        """Ground validated PP attachments by enhancing existing SceneObjectPhrases.
        
        NOTE: This method is currently disabled as we're using the new NP.preps approach
        instead of the SceneObjectPhrase approach.
        
        Args:
            validated_hypotheses: List of hypotheses with validated PP attachments
            
        Returns:
            List of hypotheses with enhanced SceneObjectPhrases
        """
        # TODO: Update this method to work with the new NP.preps approach
        # For now, just return the input hypotheses since PP merging already happened
        return validated_hypotheses
        
        # Old SceneObjectPhrase code commented out:
        # grounded_hypotheses = []
        # ... (rest of old code)
    
    def _validate_prep_spatial_relationship(self, pp_token, target_idx, hypothesis) -> float:
        """Validate a specific prepositional relationship using Layer 3 tokenization structure."""
        if target_idx is None:
            print(f"🔍 No target index for spatial validation")
            return 0.5  # Neutral score if no attachment
        
        # Get the target token that the PP should attach to
        target_token = hypothesis.tokens[target_idx]
        
        # Extract grounded objects from Layer 3 tokenization structure
        target_obj = None
        try:
            grounded_phrase = target_token._grounded_phrase
            if grounded_phrase is not None and hasattr(grounded_phrase, 'get_resolved_object'):
                target_obj = grounded_phrase.get_resolved_object()
        except AttributeError:
            pass
        
        # Extract object from PP token's PrepositionalPhrase structure
        pp_obj = None
        try:
            pp = pp_token._original_pp
            if pp is not None:
                try:
                    noun_phrase = pp.noun_phrase
                    if noun_phrase is not None and hasattr(noun_phrase, 'get_resolved_object'):
                        pp_obj = noun_phrase.get_resolved_object()
                except AttributeError:
                    pass
        except AttributeError:
            pass
        
        if not target_obj or not pp_obj:
            print(f"🔍 Missing grounded objects: target_obj={bool(target_obj)}, pp_obj={bool(pp_obj)}")
            # PP grounding should only be attempted with Scene Objects, not NPs
            return 0.0  # No spatial validation possible without grounded scene objects
        
        print(f"🔍 Found grounded objects for spatial validation")
        if hasattr(target_obj, 'position'):
            print(f"🔍 Target object position: {target_obj.position}")
        if hasattr(pp_obj, 'position'):
            print(f"🔍 PP object position: {pp_obj.position}")
        
        # Apply prep-specific spatial tests using grounded objects
        try:
            score = self._apply_prep_spatial_test(pp_token, target_obj, pp_obj)
            print(f"🔍 Final spatial score: {score}")
        except Exception as e:
            print(f"🔍 Error in spatial test: {e}")
            score = 0.0
        return score
    
    def _apply_prep_spatial_test(self, pp_token, obj1, obj2) -> float:
        """Apply prep-specific spatial relationship test using PP token's VectorSpace features.
        
        Args:
            pp_token: VectorSpace token containing spatial features (spatial_location, locX, locY, locZ)
            obj1: Target object that should be positioned relative to obj2
            obj2: Reference object (PP object) 
            
        For "box above table": obj1=box, obj2=table, validates box is above table
        """
        from engraf.utils.spatial_validation import SpatialValidator
                
        # Use the correct SpatialValidator interface
        return SpatialValidator.validate_spatial_relationship(pp_token, obj2, obj1)
    
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
    
    def extract_prepositional_phrases(self, layer3_hypotheses: List[TokenizationHypothesis]) -> List[PrepositionalPhrase]:
        """Extract PrepositionalPhrase objects from Layer 3 hypotheses.
        
        Args:
            layer3_hypotheses: List of Layer 3 tokenization hypotheses
            
        Returns:
            List of PrepositionalPhrase objects found in the hypotheses
        """
        prepositional_phrases = []
        
        for hypothesis in layer3_hypotheses:
            for token in hypothesis.tokens:
                if token._original_pp is not None and isinstance(token._original_pp, PrepositionalPhrase):
                    prepositional_phrases.append(token._original_pp)
        
        return prepositional_phrases
    
    def _merge_ppso_into_np(self, target_token, ppso_token):
        """Merge a validated PP into its target NP using proper NounPhrase.apply_pp method."""
        try:
            # Get the original PrepositionalPhrase from the PP token
            pp = ppso_token._original_pp
            if pp is None:
                print(f"❌ No original PP found in PP token: {ppso_token.word}")
                return
                
            # Get the original NounPhrase from the target NP token
            if not hasattr(target_token, '_original_np') or target_token._original_np is None:
                print(f"❌ No original NP found in target token: {target_token.word}")
                return
                
            target_np = target_token._original_np
            
            # Use the proper NounPhrase.apply_pp method to integrate the PP
            target_np.apply_pp(pp)
            
            # Update the target token's word representation to show the integrated PP
            if hasattr(target_token, 'word') and target_token.word:
                if target_token.word.startswith('NP(') and target_token.word.endswith(')'):
                    # Extract original NP content
                    original_content = target_token.word[3:-1]  # Remove 'NP(' and ')'
                    # Add PP description
                    pp_desc = f"{pp.preposition}"
                    if pp.noun_phrase:
                        if hasattr(pp.noun_phrase, 'get_original_text'):
                            pp_desc += f" {pp.noun_phrase.get_original_text()}"
                        else:
                            pp_desc += f" {str(pp.noun_phrase)}"
                    
                    new_content = f"{original_content} PP({pp_desc})"
                    target_token.word = f"NP({new_content})"
            
            # Update the target token's vector to reflect the integrated PP
            # The apply_pp method already updated the NounPhrase vector, so sync it
            target_token.clear()  # Clear existing vector data
            new_vector = target_np.to_vector()
            for dim, value in new_vector.items():
                if value != 0.0:
                    target_token[dim] = value
                    
            print(f"🔗 Properly merged PP into NP: {pp.preposition} -> {target_token.word}")
                
        except Exception as e:
            print(f"❌ Error merging PP into NP: {e}")
            import traceback
            traceback.print_exc()
