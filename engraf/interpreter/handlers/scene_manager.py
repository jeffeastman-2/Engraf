"""
Scene Manager Handler

This module handles scene-level operations and utilities,
including scene summary, clearing, and result formatting.
"""

from typing import Dict, Any, Union
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.visualizer.scene.scene_object import SceneObject


class SceneManager:
    """
    Handles scene-level operations and utilities.
    """
    
    def __init__(self, scene, renderer, execution_history_ref, object_counter_ref, object_resolver):
        """
        Initialize the scene manager.
        
        Args:
            scene: The scene model
            renderer: The renderer for visual updates
            execution_history_ref: Reference to execution history (list)
            object_counter_ref: Reference to object counter (list with single int)
            object_resolver: The object resolver for finding objects
        """
        self.scene = scene
        self.renderer = renderer
        self.execution_history_ref = execution_history_ref
        self.object_counter_ref = object_counter_ref
        self.object_resolver = object_resolver
    
    def get_scene_summary(self) -> Dict[str, Any]:
        """Get a summary of the current scene state."""
        return {
            'total_objects': len(self.scene.objects),
            'object_types': list(set(obj.name for obj in self.scene.objects)),
            'object_ids': [obj.object_id for obj in self.scene.objects],
            'execution_history': len(self.execution_history_ref),
            'last_action': self.execution_history_ref[-1] if self.execution_history_ref else None
        }
    
    def clear_scene(self):
        """Clear the current scene."""
        self.scene.objects = []
        self.scene.recent = []
        self.renderer.clear_scene()
        self.object_counter_ref[0] = 0
        self.execution_history_ref.clear()
        print("✅ Scene cleared")
    
    def create_result(self, success: bool, message: str, sentence: str) -> Dict[str, Any]:
        """Create a standardized result dictionary."""
        return {
            'success': success,
            'message': message,
            'sentence': sentence,
            'sentence_parsed': None,
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
    
    def execute_tobe_sentence(self, sentence: SentencePhrase) -> Dict[str, Any]:
        """Execute 'to be' sentences (e.g., 'the cube is red')."""
        result = {
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': ['describe']
        }
        
        # Find the subject object and apply the adjective
        if sentence.subject:
            target_objects = self.object_resolver.find_objects_by_description(sentence.subject)
            
            for obj_id in target_objects:
                # Find the scene object
                scene_obj = None
                for obj in self.scene.objects:
                    if obj.object_id == obj_id:
                        scene_obj = obj
                        break
                
                if scene_obj and hasattr(sentence, 'vector'):
                    # Apply adjective properties from the sentence vector
                    self._apply_sentence_vector(scene_obj, sentence.vector)
                    result['objects_modified'].append(obj_id)
        
        return result
    
    def execute_subject(self, subject: Union[NounPhrase, ConjunctionPhrase]) -> Dict[str, Any]:
        """Execute subject-related operations."""
        # For now, subjects are mainly used for context
        return {
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
    
    def _apply_sentence_vector(self, scene_obj: SceneObject, sentence_vector: VectorSpace):
        """Apply sentence vector properties to a scene object."""
        # Apply color properties
        if 'red' in sentence_vector:
            scene_obj.vector['red'] = sentence_vector['red']
        if 'green' in sentence_vector:
            scene_obj.vector['green'] = sentence_vector['green']
        if 'blue' in sentence_vector:
            scene_obj.vector['blue'] = sentence_vector['blue']
        
        # Apply size properties
        if 'scaleX' in sentence_vector:
            scene_obj.vector['scaleX'] = sentence_vector['scaleX']
        if 'scaleY' in sentence_vector:
            scene_obj.vector['scaleY'] = sentence_vector['scaleY']
        if 'scaleZ' in sentence_vector:
            scene_obj.vector['scaleZ'] = sentence_vector['scaleZ']
