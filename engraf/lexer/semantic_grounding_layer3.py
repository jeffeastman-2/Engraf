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
        return validated_hypotheses
    
    def validate_grounded_np(self, grounded_np):
        """Validate a grounded NP's PP attachments using spatial reasoning."""
        if not grounded_np:
            return 0.0  # No grounding to validate        
        if isinstance(grounded_np, ConjunctionPhrase):   
            # target NP is a conjunction of NPs
            nps = grounded_np.phrases
            score = 0.0
            for np in nps:
                if np.grounding:    # recursively validate each NP in the conjunction
                    score += self.validate_grounded_np(np)
            spatial_score = score / len(nps) if nps else 0.0
            return spatial_score
        else:    # target NP is a single NP
            preps = grounded_np.prepositions
            if not preps:
                return 1.0  # No PPs to validate        
            spatial_score = self._validate_prep_spatial_relationships(grounded_np, preps)
            debug_print(f"🔍 Spatial validation: NP '{grounded_np.noun}' → score {spatial_score:.2f}")
            return spatial_score
   
    def _validate_spatial_attachments(self, attachment_hypotheses):
        """Validate PP attachments using spatial reasoning and return valid hypotheses.
            A hypothesis is valid if all its grounded NPs have PPs with valid spatial relationships.
            Should also cull PPs with invalid groundings within them
        """
        validated = []    
        x = 0    
        for hypothesis in attachment_hypotheses:
            x += 1       # for debugging
            spatial_score = 1.0   
            num_gnps = 0  
            for i, token in enumerate(hypothesis.tokens):
                pString = token.phrase.printString() if token.phrase else "None"
                debug_print(f"Evaluating {pString}")
                if token.isa("NP"):
                    if token.isa("conj"):
                        for np in token.phrase.phrases:
                            grounded_np = np 
                            if not grounded_np or not grounded_np.grounding:
                                continue
                            num_gnps +=1
                            spatial_score += self.validate_grounded_np(grounded_np)
                    else:
                        grounded_np = token.phrase 
                        if not grounded_np or not grounded_np.grounding:
                            continue
                        num_gnps +=1
                        spatial_score += self.validate_grounded_np(grounded_np)
                elif token.isa("PP"):
                    if token.isa("conj"):
                        for pp in token.phrase.phrases:
                            pp = pp
                            grounded_np = pp.noun_phrase
                            if not grounded_np or not grounded_np.grounding:
                                continue
                            else:
                                num_gnps +=1
                                spatial_score += self.validate_grounded_np(grounded_np)
                    else:
                        pp = token.phrase
                        grounded_np = pp.noun_phrase
                        if not grounded_np :
                            continue
                        num_gnps +=1
                        spatial_score += self.validate_grounded_np(grounded_np)

            if spatial_score - num_gnps >= 1.0:
                num_tokens = len(hypothesis.tokens) 
                hypothesis.confidence = (hypothesis.confidence + spatial_score/num_tokens)/2 if num_tokens > 0 else hypothesis.confidence
                debug_print(f"✅ Hypothesis {x} spatially valid with score {spatial_score:.2f}")
                # Merge validated PPs into their target NPs
                validated.append(hypothesis)  # Valid

        # Sort by confidence (best first)
        validated.sort(key=lambda h: h.confidence, reverse=True)
        return validated
    
    def _validate_prep_spatial_relationships(self, target_np, pp_objs) -> float:
        """Validate a proposed relationships using spatial reasoning."""
        score = 0.0
        obj1s = target_np.grounding.get('scene_objects') if target_np.grounding else None
        for pp_obj in pp_objs:
            prep_vector = pp_obj.vector
            if isinstance(pp_obj, ConjunctionPhrase):
                # this PP is a conjunction of PPs
                np_score = 0.0
                for pp in pp_obj.phrases:
                    pp_np = pp.noun_phrase
                    if not (pp.vector.isa("spatial_location") or pp.vector.isa("spatial_proximity")):
                        return 0.0  # Cannot validate non-spatial pre
                    obj2s = pp_np.grounding.get('scene_objects') if pp_np.grounding else None
                    if obj1s is None or obj2s is None:
                        debug_print(f"❌ Cannot validate spatial relationship: missing grounding")
                        continue
                    matches = SpatialValidator.validate_spatial_relationship(pp.vector, obj1s, obj2s)
                    debug_print(f"🔍 Spatial score for '{prep_vector}': {matches}") 
                    new_grounding = []
                    for match, obj in zip(matches, obj1s):
                        if match:   # valid match => keep the object in the grounding
                            new_grounding.append(obj)
                    if new_grounding:
                        target_np.grounding['scene_objects'] = new_grounding
                        obj1s = new_grounding  # update for next PP in conjunction
                        score += 1.0
                score += np_score/len(pp_obj.phrases) if pp_obj.phrases else 0.0
            else:   # PP is a single PP
                if not (pp_obj.vector.isa("spatial_location") or pp_obj.vector.isa("spatial_proximity")):
                        return 0.0  # Cannot validate non-spatial prepositions
                pp_np = pp_obj.noun_phrase
                debug_print(f"🔍 Validating PP '{pp_obj.vector.word}' for NP '{target_np.noun}'")
                obj2s = pp_np.grounding.get('scene_objects') if pp_np.grounding else None
                if obj1s is None or obj2s is None:
                    debug_print(f"❌ Cannot validate spatial relationship: missing grounding")
                    return 0.0
                matches = SpatialValidator.validate_spatial_relationship(pp_obj.vector, obj1s, obj2s)
                debug_print(f"🔍 Spatial score for '{pp_obj.vector}': {matches}") 
                new_grounding = []
                for match, obj in zip(matches, obj1s):
                    if match:   # valid match => keep the object in the grounding
                        new_grounding.append(obj)
                if new_grounding:
                    target_np.grounding['scene_objects'] = new_grounding
                    score += 1.0
        return score/len(pp_objs) if pp_objs else 0.0
    
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
                if token.phrase is not None and isinstance(token.phrase, PrepositionalPhrase):
                    prepositional_phrases.append(token.phrase)
        
        return prepositional_phrases
    