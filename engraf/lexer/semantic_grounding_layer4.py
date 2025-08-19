#!/usr/bin/env python3
"""
LATN Layer 4 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 4 VerbPhrase tokens.
It handles verb phrase extraction and action execution.
"""

from typing import List, Optional
from dataclasses import dataclass

from engraf.pos.verb_phrase import VerbPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.latn_tokenizer_layer4 import VPTokenizationHypothesis


@dataclass
class Layer4GroundingResult:
    """Result of Layer 4 semantic grounding operation."""
    success: bool
    confidence: float
    description: str = ""


class Layer4SemanticGrounder:
    """Semantic grounding for LATN Layer 4 VerbPhrase tokens.
    
    Layer 4 performs semantic grounding of verb phrases - understanding their meaning
    and context within the scene. It does NOT execute actions or modify the scene.
    """
    
    def __init__(self, scene_model: SceneModel):
        self.scene_model = scene_model
    
    def extract_verb_phrases(self, layer4_hypotheses: List[VPTokenizationHypothesis]) -> List[VerbPhrase]:
        """Extract VerbPhrase objects from Layer 4 hypotheses.
        
        Args:
            layer4_hypotheses: List of Layer 4 tokenization hypotheses
            
        Returns:
            List of VerbPhrase objects found in the hypotheses
        """
        verb_phrases = []
        
        for hypothesis in layer4_hypotheses:
            for token in hypothesis.tokens:
                if hasattr(token, '_original_vp') and isinstance(token._original_vp, VerbPhrase):
                    verb_phrases.append(token._original_vp)
        
        return verb_phrases
    
    def ground_verb_phrases(self, verb_phrases: List[VerbPhrase]) -> List[Layer4GroundingResult]:
        """Semantically ground verb phrases by analyzing their meaning and scene context.
        
        This performs semantic analysis of verb phrases without executing actions.
        
        Args:
            verb_phrases: List of VerbPhrase objects to ground
            
        Returns:
            List of grounding results with semantic analysis
        """
        results = []
        
        for vp in verb_phrases:
            try:
                # Get verb word from VectorSpace
                verb_word = vp.verb.word if hasattr(vp.verb, 'word') else str(vp.verb)
                
                # Analyze the semantic meaning without executing actions
                if verb_word in ['create', 'make', 'build']:
                    result = self._analyze_creation_intent(vp)
                elif verb_word in ['move', 'translate']:
                    result = self._analyze_movement_intent(vp)
                elif verb_word in ['rotate', 'turn']:
                    result = self._analyze_rotation_intent(vp)
                elif verb_word in ['delete', 'remove', 'destroy']:
                    result = self._analyze_deletion_intent(vp)
                else:
                    result = Layer4GroundingResult(
                        success=True,
                        confidence=0.5,
                        description=f"Identified verb phrase with unknown action intent: {verb_word}"
                    )
                
                results.append(result)
                
            except Exception as e:
                results.append(Layer4GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"Verb phrase grounding failed: {e}"
                ))
        
        return results
    
    def _analyze_creation_intent(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Analyze creation intent without executing the action."""
        # Extract semantic properties that would be created
        object_type = "object"  # Default
        properties = {}
        
        # Extract object type from noun phrase
        if vp.noun_phrase:
            if hasattr(vp.noun_phrase, 'noun') and hasattr(vp.noun_phrase.noun, 'word'):
                noun_word = vp.noun_phrase.noun.word.lower()
                if 'cube' in noun_word or 'box' in noun_word:
                    object_type = "cube"
                elif 'sphere' in noun_word or 'ball' in noun_word:
                    object_type = "sphere"
                elif 'pyramid' in noun_word:
                    object_type = "pyramid"
            
            # Extract properties from noun phrase vector
            if hasattr(vp.noun_phrase, 'vector') and vp.noun_phrase.vector:
                vector = vp.noun_phrase.vector
                if vector.get("red", 0.0) > 0.0:
                    properties["color"] = "red"
                elif vector.get("blue", 0.0) > 0.0:
                    properties["color"] = "blue"
                elif vector.get("green", 0.0) > 0.0:
                    properties["color"] = "green"
                elif vector.get("yellow", 0.0) > 0.0:
                    properties["color"] = "yellow"
        
        # Extract position from prepositional phrases
        if vp.preps:
            for pp in vp.preps:
                if pp.preposition == 'at' and hasattr(pp, 'coordinates'):
                    properties["position"] = pp.coordinates
        
        description = f"Analyzed creation intent: {object_type}"
        if properties:
            description += f" with properties {properties}"
        
        return Layer4GroundingResult(
            success=True,
            confidence=1.0,
            description=description
        )
    
    def _analyze_movement_intent(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Analyze movement intent without executing the action."""
        return Layer4GroundingResult(
            success=True,
            confidence=0.8,
            description="Analyzed movement intent (placeholder)"
        )
    
    def _analyze_rotation_intent(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Analyze rotation intent without executing the action."""
        return Layer4GroundingResult(
            success=True,
            confidence=0.8,
            description="Analyzed rotation intent (placeholder)"
        )
    
    def _analyze_deletion_intent(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Analyze deletion intent without executing the action."""
        return Layer4GroundingResult(
            success=True,
            confidence=0.8,
            description="Analyzed deletion intent (placeholder)"
        )
