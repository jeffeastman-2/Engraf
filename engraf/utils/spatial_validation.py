"""
Shared spatial validation utilities for PP attachment and object positioning.

This module factors out common spatial reasoning logic used by both:
1. Layer 3 semantic grounding for PP attachment validation
2. Object modifier for spatial positioning calculations

Key patterns factored out:
- Preposition vector interpretation (locX, locY, locZ)
- Object dimension calculations 
- Spatial relationship validation logic
- Position calculation based on object bounds
"""

from typing import Tuple, Optional, Union
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.visualizer.scene.scene_object import SceneObject


class SpatialValidator:
    """Shared spatial validation and calculation utilities."""
    
    @staticmethod
    def get_object_half_scale(obj) -> Tuple[float, float, float]:
        """Get the half-scale of an object based on its type and size.
        
        Args:
            obj: SceneObject with vector containing scale dimensions
            
        Returns:
            tuple: (half_height, half_breadth, half_depth) representing the object's half-dimensions
        """
        if hasattr(obj, 'name') and 'cube' in obj.name.lower():
            # For cubes, all scales represent edge lengths, so half-size is scale/2
            half_height = obj.vector['scaleY'] / 2.0
            half_breadth = obj.vector['scaleX'] / 2.0
            half_depth = obj.vector['scaleZ'] / 2.0
        elif hasattr(obj, 'name') and 'sphere' in obj.name.lower():
            # For spheres, all scales represent radius, so half-size equals the radius
            radius = max(obj.vector['scaleX'], obj.vector['scaleY'], obj.vector['scaleZ'])
            half_height = radius
            half_breadth = radius
            half_depth = radius
        else:
            # Default: assume scales represent full dimensions, so half-size is scale/2
            half_height = obj.vector['scaleY'] / 2.0
            half_breadth = obj.vector['scaleX'] / 2.0
            half_depth = obj.vector['scaleZ'] / 2.0
            
        return half_height, half_breadth, half_depth
    
    @staticmethod
    def extract_direction_factors(preposition_vector) -> Tuple[float, float, float]:
        """Extract directional factors from a preposition vector.
        
        Args:
            preposition_vector: VectorSpace object containing locX, locY, locZ values
            
        Returns:
            tuple: (x_factor, y_factor, z_factor) representing spatial direction
        """
        x_factor = preposition_vector['locX'] if 'locX' in preposition_vector and preposition_vector['locX'] != 0.0 else 0.0
        y_factor = preposition_vector['locY'] if 'locY' in preposition_vector and preposition_vector['locY'] != 0.0 else 0.0
        z_factor = preposition_vector['locZ'] if 'locZ' in preposition_vector and preposition_vector['locZ'] != 0.0 else 0.0
        
        return x_factor, y_factor, z_factor
    
    @staticmethod
    def calculate_spatial_position(moving_obj, ref_obj, preposition_vector) -> Tuple[float, float, float]:
        """Calculate the position for spatial relationships like 'above', 'below', 'beside', etc.
        
        Uses the preposition vector to determine spatial direction and object dimensions
        for proper spacing.
        
        Args:
            moving_obj: Object being positioned
            ref_obj: Reference object for spatial relationship
            preposition: Preposition string (for logging)
            preposition_vector: VectorSpace with locX, locY, locZ direction factors
            
        Returns:
            tuple: (new_x, new_y, new_z) position for the moving object
        """
        # Get direction factors from the preposition vector
        x_factor, y_factor, z_factor = SpatialValidator.extract_direction_factors(preposition_vector)
        
        # Get reference object's position and size
        ref_x = ref_obj.vector['locX']
        ref_y = ref_obj.vector['locY'] 
        ref_z = ref_obj.vector['locZ']
        
        # Calculate object dimensions for proper spacing
        refHeight, refBreadth, refDepth = SpatialValidator.get_object_half_scale(ref_obj)
        movingHeight, movingBreadth, movingDepth = SpatialValidator.get_object_half_scale(moving_obj)
        
        # Start with reference object's position as base
        new_x = ref_x
        new_y = ref_y
        new_z = ref_z

        # Calculate X position based on directional factor
        if x_factor > 0:
            # Place object to the positive X direction (right/beside)
            new_x = ref_x + refBreadth + movingBreadth + abs(x_factor)
        elif x_factor < 0:
            # Place object to the negative X direction (left)
            new_x = ref_x - refBreadth - movingBreadth - abs(x_factor)

        # Calculate Y position based on directional factor
        if y_factor > 0:
            # Place object in positive Y direction (above)
            new_y = ref_y + refHeight + movingHeight + abs(y_factor)
        elif y_factor < 0:
            # Place object in negative Y direction (below)
            new_y = ref_y - refHeight - movingHeight - abs(y_factor)

        # Calculate Z position based on directional factor
        if z_factor > 0:
            # Place object in positive Z direction (behind)
            new_z = ref_z + refDepth + movingDepth + abs(z_factor)
        elif z_factor < 0:
            # Place object in negative Z direction (in front)
            new_z = ref_z - refDepth - movingDepth - abs(z_factor)

        return new_x, new_y, new_z
    
    @staticmethod
    def validate_spatial_relationship(prep_vector, obj1, obj2) -> float:
        """Validate a spatial relationship between two objects using proper spatial calculations.
        
        Args:
            prep_vector: PP token containing spatial features (VectorSpace with locX, locY, locZ) or preposition string
            obj1: Reference object (obj2 is positioned relative to obj1)
            obj2: Object being positioned relative to obj1
            
        Returns:
            float: Validation score between 0.0 and 1.0
        """
        try:
            print(f"🔍 Prep vector for '{prep_vector.word}': locX={prep_vector['locX']}, locY={prep_vector['locY']}, locZ={prep_vector['locZ']}")
            
            # Calculate where obj2 should be positioned relative to obj1
            expected_x, expected_y, expected_z = SpatialValidator.calculate_spatial_position(
                obj2, obj1, prep_vector
            )
            print(f"🔍 Expected position for obj2 relative to obj1: [{expected_x}, {expected_y}, {expected_z}]")
            
            # Get actual position of obj2
            actual_x = obj2.vector['locX'] if hasattr(obj2, 'vector') else (obj2.position[0] if hasattr(obj2, 'position') else 0)
            actual_y = obj2.vector['locY'] if hasattr(obj2, 'vector') else (obj2.position[1] if hasattr(obj2, 'position') else 0)
            actual_z = obj2.vector['locZ'] if hasattr(obj2, 'vector') else (obj2.position[2] if hasattr(obj2, 'position') else 0)
            print(f"🔍 Actual position of obj2: [{actual_x}, {actual_y}, {actual_z}]")
            
            # Calculate distance between expected and actual positions
            dx = expected_x - actual_x
            dy = expected_y - actual_y
            dz = expected_z - actual_z
            distance = (dx*dx + dy*dy + dz*dz) ** 0.5
            print(f"🔍 Distance between expected and actual: {distance}")
            
            # Convert distance to validation score (closer = higher score)
            # Use object dimensions to determine reasonable tolerance
            obj1_half_height, obj1_half_breadth, obj1_half_depth = SpatialValidator.get_object_half_scale(obj1)
            obj2_half_height, obj2_half_breadth, obj2_half_depth = SpatialValidator.get_object_half_scale(obj2)
            
            # Tolerance based on object sizes
            tolerance = max(obj1_half_height, obj1_half_breadth, obj1_half_depth, 
                          obj2_half_height, obj2_half_breadth, obj2_half_depth) * 0.5
            print(f"🔍 Tolerance: {tolerance}")
            
            if distance <= tolerance:
                score = 1.0  # Perfect spatial relationship
            elif distance <= tolerance * 3:
                score = 0.8  # Good spatial relationship
            elif distance <= tolerance * 6:
                score = 0.5  # Acceptable spatial relationship
            else:
                score = 0.1  # Poor spatial relationship
                
            print(f"🔍 Final score from preposition vector validation: {score}")
            return score
                
        except Exception as e:
            print(f"🔍 Exception in preposition vector validation: {e}")
            # Fallback to position-based validation if vector approach fails
            return SpatialValidator._validate_using_positions(prep_vector, obj1, obj2)
    
    @staticmethod
    def _validate_using_positions(pp_token: VectorSpace, obj1: SceneObject, obj2: SceneObject) -> float:
        """Fallback validation using simple position comparison."""
        try:
            # Get positions 
            pos1 = obj1.position                
            pos2 = obj2.position
            
            dx, dy, dz = pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]
            px = pp_token['locX']
            py = pp_token['locY']  
            pz = pp_token['locZ']
            
            if pp_token.isa("spatial_location"):
                # spatial relationships: prepositions affecting object positioning
                dot = (dx * px + dy * py + dz * pz)
                return 1.0 if dot > 0 else 0.0
            elif pp_token.isa("spatial_proximity"):
                # proximity relationships: near (+), at (specific), in (containment)
                distance = (dx*dx + dy*dy + dz*dz) ** 0.5
                return 1.0 if distance < 1.0 else 0.0
            else: return 0.0
        except Exception:
            return 0.0
