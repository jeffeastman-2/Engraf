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
    executed_action: Optional[str] = None
    description: str = ""


class Layer4SemanticGrounder:
    """Semantic grounding for LATN Layer 4 VerbPhrase tokens."""
    
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
    
    def execute_verb_phrase_actions(self, verb_phrases: List[VerbPhrase]) -> List[Layer4GroundingResult]:
        """Execute actions for verb phrases.
        
        Args:
            verb_phrases: List of VerbPhrase objects to execute
            
        Returns:
            List of execution results
        """
        results = []
        
        for vp in verb_phrases:
            try:
                # Get verb word from VectorSpace or direct attribute
                verb_word = vp.verb.word if hasattr(vp.verb, 'word') else str(vp.verb)
                
                if verb_word in ['create', 'make', 'build']:
                    result = self._execute_create_action(vp)
                elif verb_word in ['move', 'translate']:
                    result = self._execute_move_action(vp)
                elif verb_word in ['rotate', 'turn']:
                    result = self._execute_rotate_action(vp)
                elif verb_word in ['delete', 'remove', 'destroy']:
                    result = self._execute_delete_action(vp)
                else:
                    result = Layer4GroundingResult(
                        success=False,
                        confidence=0.0,
                        description=f"Unknown verb: {verb_word}"
                    )
                
                results.append(result)
                
            except Exception as e:
                results.append(Layer4GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"Action execution failed: {e}"
                ))
        
        return results
    
    def _execute_create_action(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Execute create action."""
        if not self.scene_model:
            return Layer4GroundingResult(
                success=False,
                confidence=0.0,
                description="No scene model available"
            )
        
        # Extract object properties from verb phrase
        object_type = "cube"  # Default
        position = [0, 0, 0]  # Default
        color = "blue"  # Default
        
        # Parse noun phrases and prepositional phrases for details
        if vp.noun_phrase:
            if hasattr(vp.noun_phrase, 'word'):
                np_word = vp.noun_phrase.word if hasattr(vp.noun_phrase, 'word') else str(vp.noun_phrase)
                if 'cube' in np_word or 'box' in np_word:
                    object_type = "cube"
                elif 'sphere' in np_word or 'ball' in np_word:
                    object_type = "sphere"
                elif 'pyramid' in np_word:
                    object_type = "pyramid"
                    
                # Extract color from noun phrase
                if 'red' in np_word:
                    color = "red"
                elif 'blue' in np_word:
                    color = "blue"
                elif 'green' in np_word:
                    color = "green"
                elif 'yellow' in np_word:
                    color = "yellow"
        
        if vp.prepositional_phrases:
            for pp in vp.prepositional_phrases:
                if pp.preposition == 'at' and hasattr(pp, 'coordinates'):
                    position = pp.coordinates
        
        # Create the object
        from engraf.visualizer.scene.scene_object import SceneObject
        from engraf.lexer.vector_space import VectorSpace
        import uuid
        
        # Create a vector for the object
        object_vector = VectorSpace(word=object_type)
        object_vector['noun'] = 1.0
        object_vector[color] = 1.0
        
        # Set position if provided
        if position != [0, 0, 0]:
            object_vector['locX'] = position[0]
            object_vector['locY'] = position[1]
            object_vector['locZ'] = position[2]
        
        # Create unique ID
        obj_id = f"{object_type}_{str(uuid.uuid4())[:8]}"
        
        # Create and add the scene object
        scene_obj = SceneObject(
            name=object_type,
            object_id=obj_id,
            vector=object_vector
        )
        
        self.scene_model.add_object(scene_obj)
        
        return Layer4GroundingResult(
            success=True,
            confidence=1.0,
            executed_action=f"create {object_type}",
            description=f"Created {object_type} at {position}"
        )
    
    def _execute_move_action(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Execute move action."""
        return Layer4GroundingResult(
            success=True,
            confidence=0.8,
            executed_action="move",
            description="Move action executed (placeholder)"
        )
    
    def _execute_rotate_action(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Execute rotate action."""
        return Layer4GroundingResult(
            success=True,
            confidence=0.8,
            executed_action="rotate",
            description="Rotate action executed (placeholder)"
        )
    
    def _execute_delete_action(self, vp: VerbPhrase) -> Layer4GroundingResult:
        """Execute delete action."""
        return Layer4GroundingResult(
            success=True,
            confidence=0.8,
            executed_action="delete",
            description="Delete action executed (placeholder)"
        )
