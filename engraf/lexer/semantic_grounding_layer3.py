#!/usr/bin/env python3
"""
LATN Layer 3 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 3 PrepositionalPhrase tokens.
It bridges between parsed PrepositionalPhrase structures and spatial locations/relationships.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
 
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.hypothesis import TokenizationHypothesis
from engraf.utils.debug import debug_print
from engraf.utils.spatial_validation import SpatialValidator
from engraf.visualizer.scene.scene_model import SceneModel


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
    
    def __init__(self, scene_model: SceneModel):
        self.scene_model = scene_model
    
    def ground_layer3(self, layer3_hypotheses):
        """Two-pass PP attachment resolution with spatial validation and semantic grounding.
        
        Args:
            layer3_hypotheses: Hypotheses with PP tokens to process
            
        Returns:
            Processed hypotheses with validated PP attachments and semantic grounding
        """
        # Spatial validation filter
        validated_hypotheses = self._validate_spatial_attachments(layer3_hypotheses)
        debug_print(f"✅ Pass 2: {len(validated_hypotheses)} spatially valid combinations")
        
        # Pass 3: Semantic grounding for validated attachments
        if validated_hypotheses:
            # For now, return the validated hypotheses since PP merging already happened in validation
            # The _merge_ppso_into_np method in _validate_spatial_attachments already did the work
            debug_print(f"🎯 Pass 3: Returning validated hypotheses with merged PPs")
            return validated_hypotheses
        
        return layer3_hypotheses
    
    def validate_grounded_np(self, grounded_np):
        """Validate a grounded NP's PP attachments using spatial reasoning."""
        if not grounded_np:
            return 1.0  # No grounding to validate        
        if isinstance(grounded_np, ConjunctionPhrase):   
            nps = list(grounded_np.flatten())
            score = 0.0
            for np in nps:
                if np.grounding:
                    spatial_score = min(spatial_score, self.validate_grounded_np(np))
                    score += self.validate_grounded_np(np)
                else:
                    continue
            spatial_score = score / len(nps) if nps else 0.0
            return spatial_score
        else:    
            preps = grounded_np.preps
            if not preps:
                return 0.0  # No PPs to validate        
            spatial_score = self._validate_prep_spatial_relationships(grounded_np, preps)
            debug_print(f"🔍 Spatial validation: NP '{grounded_np.noun}' → score {spatial_score:.2f}")
            # Recursively validate nested PPs
            if grounded_np.preps:
                sub_score = 0.0
                for pp in grounded_np.preps:
                    g_np = pp.noun_phrase
                    sub_score += self.validate_grounded_np(g_np)
                sub_score += sub_score / len(grounded_np.preps)
                spatial_score = spatial_score + sub_score / 2
                debug_print(f"🔍 Recursive spatial score including PPs: {spatial_score:.2f}")
            return spatial_score
   
    def _validate_spatial_attachments(self, attachment_hypotheses):
        """Pass 2: Validate PP attachments using spatial reasoning and return valid hypotheses."""
        validated = []    
        x = 0    
        for hypothesis in attachment_hypotheses:
            x += 1       # for debugging
            spatial_score = 1.0     
            for i, token in enumerate(hypothesis.tokens):
                if token.isa("NP"):
                    if token.isa("conj"):
                        continue  # Skip conjunctions
                    grounded_np = token._grounded_phrase 
                    if not grounded_np:
                        continue
                    spatial_score = self.validate_grounded_np(grounded_np)
                elif token.isa("PP"):
                    pp = token._original_pp
                    grounded_np = pp.noun_phrase
                    spatial_score = self.validate_grounded_np(grounded_np)

            if spatial_score >= 0.75:  # Threshold for valid spatial relationship
                validated.append(hypothesis)  # Valid

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
    
    def _validate_prep_spatial_relationships(self, target_np, pp_objs) -> float:
        """Validate a proposed relationships using spatial reasoning."""
        score = 0.0
        for pp_obj in pp_objs:
            prep = pp_obj.vector
            debug_print(f"🔍 Validating PP '{prep}' for NP '{target_np.noun}'")
            obj1 = target_np.grounding.get('scene_object') if target_np.grounding else None
            obj2 = pp_obj.noun_phrase.grounding.get('scene_object') if pp_obj.noun_phrase.grounding else None
            if obj1 is None or obj2 is None:
                debug_print(f"❌ Cannot validate spatial relationship: missing grounding")
                return 0.0
            score += SpatialValidator.validate_spatial_relationship(prep, obj1, obj2)
            debug_print(f"🔍 Spatial score for '{prep}': {score:.2f}")            
        return score/len(pp_objs) if pp_objs else 0.0

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
            if token.isa("NP") or token.isa("PP"):
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
                debug_print(f"❌ No original PP found in PP token: {ppso_token.word}")
                return
                
            # Get the original NounPhrase from the target NP token
            if not hasattr(target_token, '_original_np') or target_token._original_np is None:
                debug_print(f"❌ No original NP found in target token: {target_token.word}")
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
            # TODO: Fix vector synchronization later
            # For now, just complete the merge without vector update
                    
            debug_print(f"🔗 Properly merged PP into NP: {pp.preposition} -> {target_token.word}")
                
        except Exception as e:
            debug_print(f"❌ Error merging PP into NP: {e}")
            from engraf.utils.debug import debug_enabled
            if debug_enabled:
                import traceback
                traceback.print_exc()
