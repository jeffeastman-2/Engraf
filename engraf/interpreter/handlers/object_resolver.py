"""
Object Resolver Handler

This module handles finding and resolving references to scene objects,
including pronoun resolution and semantic matching.
"""

from typing import List, Tuple
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


class ObjectResolver:
    """
    Handles finding and resolving references to scene objects.
    """
    
    def __init__(self, scene, last_acted_object_ref):
        """
        Initialize the object resolver.
        
        Args:
            scene: The scene model containing objects
            last_acted_object_ref: Reference to the last acted upon object (list with single string)
        """
        self.scene = scene
        self.last_acted_object_ref = last_acted_object_ref
    
    def resolve_target_objects(self, vp: VerbPhrase) -> List[str]:
        """Resolve target objects for modification verbs."""
        target_objects = []
        
        # Check for pronouns (e.g., "move it")
        if vp.noun_phrase and vp.noun_phrase.pronoun:
            # Get the most recently acted upon object for "it", all objects for "them"/"they"
            if vp.noun_phrase.pronoun.lower() == "it":
                if self.last_acted_object_ref[0]:
                    target_objects.append(self.last_acted_object_ref[0])
                elif self.scene.objects:
                    # Fallback to most recently created object if no object has been acted upon
                    target_objects.append(self.scene.objects[-1].object_id)
            elif vp.noun_phrase.pronoun.lower() in ("them", "they"):
                target_objects.extend([obj.object_id for obj in self.scene.objects])
        
        # Check for specific noun phrases
        elif vp.noun_phrase:
            # Find objects by type/description
            objects = self.find_objects_by_description(vp.noun_phrase)
            target_objects.extend(objects)
        
        return target_objects
    
    def find_objects_by_description(self, np: NounPhrase) -> List[str]:
        """Find objects in the scene that match a noun phrase description using vector distance."""
        # Get all objects with their match scores
        object_scores = []
        
        for scene_obj in self.scene.objects:
            if self._object_matches_description(scene_obj, np):
                # Calculate distance for ranking (lower is better)
                distance = 0.0
                if hasattr(np, 'vector') and np.vector:
                    distance = self._calculate_vector_distance(scene_obj.vector, np.vector)
                
                object_scores.append((scene_obj.object_id, distance))
        
        # Sort by distance (best matches first) and return object IDs
        object_scores.sort(key=lambda x: x[1])
        return [obj_id for obj_id, _ in object_scores]
    
    def _object_matches_description(self, scene_obj: SceneObject, np: NounPhrase) -> bool:
        """Check if a scene object matches a noun phrase description using two-pass approach."""
        # First check if basic noun type matches
        if np.noun and np.noun != scene_obj.name:
            return False
        
        # If no vector information in the noun phrase, just match by type
        if not hasattr(np, 'vector') or not np.vector:
            return True
        
        # Check for clear feature mismatches (e.g., "red cube" vs blue cube)
        if self._has_feature_mismatch(scene_obj.vector, np.vector):
            return False
        
        # If we get here, the noun matches and there's no clear feature conflict
        # This is a valid candidate - the find_objects_by_description method will
        # use vector distance to rank multiple candidates
        return True
    
    def _has_feature_mismatch(self, obj_vector: VectorSpace, query_vector: VectorSpace) -> bool:
        """Check if there's a clear feature mismatch (e.g., color conflict)."""
        # Check for color mismatches - reject if query specifies a color that conflicts with object's color
        color_features = ['red', 'green', 'blue']
        
        for color in color_features:
            try:
                obj_value = obj_vector[color]
                query_value = query_vector[color]
                
                # If query strongly specifies this color (>0.5) but object doesn't have it (<=0.5)
                if query_value > 0.5 and obj_value <= 0.5:
                    return True
                    
            except (ValueError, IndexError):
                # Feature not found in vector, skip it
                continue
        
        # Also check for conflicting colors: if object has a strong color and query has a different strong color
        try:
            obj_colors = []
            query_colors = []
            
            # Find which colors the object has
            for color in color_features:
                if obj_vector[color] > 0.5:
                    obj_colors.append(color)
                if query_vector[color] > 0.5:
                    query_colors.append(color)
            
            # If both have colors and none overlap, it's a mismatch
            if obj_colors and query_colors and not any(c in obj_colors for c in query_colors):
                return True
                
        except (ValueError, IndexError):
            pass
        
        return False
    
    def _calculate_vector_distance(self, obj_vector: VectorSpace, query_vector: VectorSpace) -> float:
        """Calculate sophisticated vector distance between object and query."""
        # Define feature categories with different weights - ONLY semantic features, not POS features
        feature_weights = {
            # Color features (high weight - very specific)
            'red': 2.0, 'green': 2.0, 'blue': 2.0,
            # Size features (medium weight)
            'scaleX': 1.5, 'scaleY': 1.5, 'scaleZ': 1.5,
            # Position features (low weight - less relevant for matching)
            'locX': 0.5, 'locY': 0.5, 'locZ': 0.5,
            # Other semantic features
            'texture': 1.0, 'transparency': 1.0
        }
        
        total_distance = 0.0
        total_weight = 0.0
        
        # Calculate weighted distance for each feature
        for feature, weight in feature_weights.items():
            try:
                obj_value = obj_vector[feature]
                query_value = query_vector[feature]
                
                # Special handling for binary features (colors)
                if feature in ['red', 'green', 'blue']:
                    # For colors, check if both are "on" (>0.5) or both are "off" (<0.5)
                    obj_on = obj_value > 0.5
                    query_on = query_value > 0.5
                    
                    if obj_on == query_on:
                        # Perfect match for this color
                        distance = 0.0
                    else:
                        # Mismatch - one has color, other doesn't
                        distance = 1.0
                
                # Handle scale features
                elif feature.startswith('scale'):
                    # For scale, check if both indicate same size category
                    obj_size = self._categorize_scale(obj_value)
                    query_size = self._categorize_scale(query_value)
                    
                    if obj_size == query_size:
                        distance = 0.0
                    else:
                        distance = abs(obj_value - query_value)
                
                # Handle other features with simple distance
                else:
                    distance = abs(obj_value - query_value)
                
                total_distance += distance * weight
                total_weight += weight
                
            except (ValueError, IndexError):
                # Feature not found in vector, skip it
                continue
        
        # Return normalized distance (0.0 = perfect match, 1.0 = completely different)
        return total_distance / total_weight if total_weight > 0 else 1.0
    
    def _categorize_scale(self, scale_value: float) -> str:
        """Categorize a scale value into size categories."""
        if scale_value >= 1.5:
            return 'large'
        elif scale_value <= 0.75:
            return 'small' 
        else:
            return 'normal'
