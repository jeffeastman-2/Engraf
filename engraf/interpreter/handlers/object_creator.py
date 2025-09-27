"""
Object Creator Handler

This module handles the creation of new scene objects from parsed noun phrases.
It extracts object information, generates descriptive IDs, and applies default properties.
"""

from typing import Optional, List, Dict, Any, Union
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


class ObjectCreator:
    """
    Handles the creation of scene objects from linguistic structures.
    """
    
    def __init__(self, scene, object_counter_ref):
        """
        Initialize the object creator.
        
        Args:
            scene: The scene model to add objects to
            object_counter_ref: Reference to the object counter (list with single int)
        """
        self.scene = scene
        self.object_counter_ref = object_counter_ref
    
    def extract_objects_from_np(self, np: Union[NounPhrase, ConjunctionPhrase]) -> List[Dict[str, Any]]:
        """Extract object information from a noun phrase or conjunction."""
        objects = []
        
        if isinstance(np, ConjunctionPhrase):
            # Handle conjunctions (e.g., "a cube and a sphere")
            objects.extend(np.phrases)
        
        elif isinstance(np, NounPhrase):
            # Extract object information from single noun phrase
            obj_info = self._extract_single_object_info(np)
            if obj_info:
                objects.append(obj_info)
        
        return objects
    
    def _extract_single_object_info(self, np: NounPhrase) -> Optional[Dict[str, Any]]:
        """Extract object information from a single noun phrase."""
        if not np.noun:
            return None
        
        # Basic object info
        obj_info = {
            'type': np.noun,
            'determiner': np.determiner,
            'adjectives': getattr(np, 'adjectives', []),
            'vector_space': np.vector if hasattr(np, 'vector') else None,
            'custom_name': getattr(np, 'custom_name', None)
        }
        
        # Handle prepositional phrases for positioning
        if hasattr(np, 'preps') and np.preps:
            obj_info['prepositional_phrases'] = np.preps
        
        return obj_info
    
    def create_scene_object(self, obj_info: Dict[str, Any]) -> Optional[str]:
        """Create a new scene object from object information."""
        try:
            self.object_counter_ref[0] += 1
            
            # Create descriptive ID that includes key adjectives
            obj_id = self._generate_descriptive_id(obj_info)
            
            # Create vector space for the object
            vector_space = obj_info.get('vector_space') or VectorSpace()
            
            # Apply default properties based on object type
            self._apply_default_properties(vector_space, obj_info)
            
            # Create and add scene object
            scene_object = SceneObject(
                name=obj_info['type'],  # Store the base noun (e.g., 'cube')
                vector=vector_space,
                object_id=obj_id       # Store the descriptive ID (e.g., 'red_cube_1')
            )
            
            # Apply prepositional phrases (positioning) if present
            if 'prepositional_phrases' in obj_info:
                for prep_phrase in obj_info['prepositional_phrases']:
                    self._apply_movement(scene_object, prep_phrase)
            
            self.scene.add_object(scene_object)
            
            print(f"✅ Created object: {obj_id}")
            return obj_id
            
        except Exception as e:
            print(f"❌ Failed to create object: {e}")
            return None
    
    def _check_name_conflict(self, name: str) -> bool:
        """Check if an object ID already exists in the scene."""
        # Check objects
        for obj in self.scene.objects:
            if obj.object_id == name:
                return True
        
        # Check assemblies
        for assembly in getattr(self.scene, 'assemblies', []):
            if assembly.assembly_id == name:
                return True
        
        return False
    
    def _generate_descriptive_id(self, obj_info: Dict[str, Any]) -> str:
        """Generate a simple descriptive ID for the object using only base noun and counter."""
        base_type = obj_info['type']
        
        # Check if user provided a custom name
        custom_name = obj_info.get('custom_name')
        if custom_name:
            # Check for naming conflicts
            if self._check_name_conflict(custom_name):
                raise ValueError(f"Object named '{custom_name}' already exists. Please choose a different name.")
            return custom_name
        
        # Use simple format: noun-counter (no adjectives as they can change)
        return f"{base_type}-{self.object_counter_ref[0]}"
    
    def _extract_colors_from_vector(self, vector_space: VectorSpace) -> List[str]:
        """Extract color names from vector space."""
        colors = []
        if vector_space['red'] > 0.5:
            colors.append('red')
        if vector_space['green'] > 0.5:
            colors.append('green')
        if vector_space['blue'] > 0.5:
            colors.append('blue')
        return colors
    
    def _extract_sizes_from_vector(self, vector_space: VectorSpace) -> List[str]:
        """Extract size descriptors from vector space."""
        sizes = []
        if vector_space['scaleX'] > 1.5 or vector_space['scaleY'] > 1.5 or vector_space['scaleZ'] > 1.5:
            sizes.append('big')
        elif vector_space['scaleX'] < 0.75 or vector_space['scaleY'] < 0.75 or vector_space['scaleZ'] < 0.75:
            sizes.append('small')
        return sizes
    
    def _apply_default_properties(self, vector_space: VectorSpace, obj_info: Dict[str, Any]):
        """Apply default properties to a vector space based on object information."""
        # Default position
        vector_space['locX'] = 0.0
        vector_space['locY'] = 0.0
        vector_space['locZ'] = 0.0
        
        # Default size if not already set
        if vector_space['scaleX'] == 0.0:
            vector_space['scaleX'] = 1.0
        if vector_space['scaleY'] == 0.0:
            vector_space['scaleY'] = 1.0
        if vector_space['scaleZ'] == 0.0:
            vector_space['scaleZ'] = 1.0
        
        # Default color (white) only if no color is already set
        if vector_space['red'] == 0.0 and vector_space['green'] == 0.0 and vector_space['blue'] == 0.0:
            vector_space['red'] = 1.0
            vector_space['green'] = 1.0
            vector_space['blue'] = 1.0
        
        # Note: Adjectives are already applied by ATN parsing via NounPhrase.apply_adjective()
        # No need to manually apply them again here
    
    def _apply_movement(self, scene_obj: SceneObject, preposition):
        """Apply movement to an object based on prepositional phrase."""
        # This would integrate with the transform interpreter
        # For now, simple implementation
        if hasattr(preposition, 'noun_phrase') and hasattr(preposition.noun_phrase, 'vector'):
            vector = preposition.noun_phrase.vector
            if vector['locX'] != 0.0:
                scene_obj.vector['locX'] = vector['locX']
            if vector['locY'] != 0.0:
                scene_obj.vector['locY'] = vector['locY']
            if vector['locZ'] != 0.0:
                scene_obj.vector['locZ'] = vector['locZ']
