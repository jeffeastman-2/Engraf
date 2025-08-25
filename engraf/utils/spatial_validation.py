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
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.noun_phrase import NounPhrase
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
    def validate_spatial_relationship(pp_token, obj1, obj2) -> float:
        """Validate a spatial relationship between two objects using proper spatial calculations.
        
        Args:
            pp_token: PP token containing spatial features (VectorSpace with locX, locY, locZ) or preposition string
            obj1: Reference object (obj2 is positioned relative to obj1)
            obj2: Object being positioned relative to obj1
            
        Returns:
            float: Validation score between 0.0 and 1.0
        """
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
