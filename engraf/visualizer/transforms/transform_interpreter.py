"""
Transform Interpreter for converting linguistic structures to transformation matrices.

This module bridges the gap between parsed natural language sentences and
3D transformations, converting verb phrases and prepositional phrases into
TransformMatrix objects.
"""

import re
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from engraf.visualizer.transforms.transform_matrix import TransformMatrix
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.lexer.vocabulary import get_from_vocabulary


class TransformInterpreter:
    """
    Interprets natural language structures and converts them to transformation matrices.
    
    Handles various transformation types:
    - Movement: "move up 2 units", "move to [1, 2, 3]"
    - Rotation: "rotate by [0, 90, 0]", "turn 45 degrees"
    - Scaling: "scale by [2, 1, 1]", "make it twice as big"
    """
    
    def __init__(self):
        """Initialize the transform interpreter."""
        pass
    
    def interpret_verb_phrase(self, verb_phrase: VerbPhrase) -> Optional[TransformMatrix]:
        """
        Interpret a verb phrase and return the corresponding transformation.
        
        Args:
            verb_phrase: The verb phrase to interpret
            
        Returns:
            TransformMatrix if the verb phrase represents a transformation, None otherwise
        """
        if not verb_phrase.verb:
            return None
        
        verb = verb_phrase.verb.lower()
        
        # Get semantic vector for the verb to check its category
        verb_vector = get_from_vocabulary(verb)
        
        # Check if this is a transformation verb using semantic categories
        if verb_vector and self._is_transform_verb(verb_vector):
            return self._interpret_transform_verb(verb_phrase, verb_vector)
        
        # Fall back to original string-based matching for compatibility
        # Check for movement verbs
        if verb in ["move", "translate", "place", "position"]:
            return self._interpret_movement(verb_phrase)
        
        # Check for rotation verbs
        elif verb in ["rotate", "turn", "spin", "roll", "xrotate", "yrotate", "zrotate"]:
            return self._interpret_rotation(verb_phrase)
        
        # Check for scaling verbs
        elif verb in ["scale", "resize", "enlarge", "shrink", "make"]:
            return self._interpret_scaling(verb_phrase)
        
        return None
    
    def _is_transform_verb(self, verb_vector) -> bool:
        """
        Check if a verb vector represents a transformation action.
        
        Args:
            verb_vector: The semantic vector for the verb
            
        Returns:
            True if the verb is a transformation verb, False otherwise
        """
        if not verb_vector:
            return False
        
        # Check for transform action category
        return (verb_vector["verb"] > 0 and 
                verb_vector["action"] > 0 and 
                verb_vector["transform"] > 0)
    
    def _interpret_transform_verb(self, verb_phrase: VerbPhrase, verb_vector) -> Optional[TransformMatrix]:
        """
        Interpret a transformation verb using semantic categories.
        
        Args:
            verb_phrase: The verb phrase to interpret
            verb_vector: The semantic vector for the verb
            
        Returns:
            TransformMatrix if interpretation succeeds, None otherwise
        """
        verb = verb_phrase.verb.lower()
        
        # Handle specific rotation verbs
        if verb == "xrotate":
            degrees = self._extract_degrees(verb_phrase)
            if degrees is not None:
                return TransformMatrix.rotation_x(degrees)
            return TransformMatrix.rotation_x(90)  # Default 90 degrees
        
        elif verb == "yrotate":
            degrees = self._extract_degrees(verb_phrase)
            if degrees is not None:
                return TransformMatrix.rotation_y(degrees)
            return TransformMatrix.rotation_y(90)  # Default 90 degrees
        
        elif verb == "zrotate":
            degrees = self._extract_degrees(verb_phrase)
            if degrees is not None:
                return TransformMatrix.rotation_z(degrees)
            return TransformMatrix.rotation_z(90)  # Default 90 degrees
        
        # Handle general transform verbs
        elif verb in ["move", "position"]:
            return self._interpret_movement(verb_phrase)
        
        elif verb in ["rotate"]:
            return self._interpret_rotation(verb_phrase)
        
        elif verb in ["scale"]:
            return self._interpret_scaling(verb_phrase)
        
        return None
    
    def interpret_verb_semantics(self, verb_phrase: VerbPhrase) -> Optional[Dict[str, Any]]:
        """
        Interpret a verb phrase and return its semantic category and properties.
        
        This method provides broader semantic analysis beyond just transformations,
        useful for the complete visualization system.
        
        Args:
            verb_phrase: The verb phrase to interpret
            
        Returns:
            Dictionary with semantic information or None if no semantic info available
        """
        if not verb_phrase.verb:
            return None
        
        verb = verb_phrase.verb.lower()
        verb_vector = get_from_vocabulary(verb)
        
        if not verb_vector:
            return None
        
        # Extract semantic categories
        semantic_info = {
            "verb": verb,
            "is_action": verb_vector["action"] > 0,
            "category": self._get_verb_category(verb_vector),
            "vector": verb_vector
        }
        
        # Add category-specific information
        if semantic_info["category"] == "create":
            semantic_info["creates_object"] = True
        elif semantic_info["category"] == "edit":
            semantic_info["modifies_object"] = True
        elif semantic_info["category"] == "organize":
            semantic_info["affects_layout"] = True
        elif semantic_info["category"] == "select":
            semantic_info["affects_selection"] = True
        elif semantic_info["category"] == "style":
            semantic_info["affects_appearance"] = True
        elif semantic_info["category"] == "transform":
            semantic_info["affects_position"] = True
        
        return semantic_info
    
    def _get_verb_category(self, verb_vector) -> Optional[str]:
        """
        Extract the semantic category from a verb vector.
        
        Args:
            verb_vector: The semantic vector for the verb
            
        Returns:
            String category name or None if no category found
        """
        if not verb_vector:
            return None
        
        # Check each semantic category
        categories = ["create", "edit", "organize", "select", "style", "transform"]
        
        for category in categories:
            if verb_vector[category] > 0:
                return category
        
        return None
    
    def _interpret_movement(self, verb_phrase: VerbPhrase) -> Optional[TransformMatrix]:
        """
        Interpret movement-related verb phrases.
        
        Examples:
        - "move up 2 units" -> translation(0, 2, 0)
        - "move to [1, 2, 3]" -> translation(1, 2, 3)
        - "place at [5, 0, 5]" -> translation(5, 0, 5)
        """
        # Check for coordinate vector in prepositional phrases
        for prep in verb_phrase.preps:
            if prep.preposition in ["to", "at"]:
                coords = self._extract_coordinates(prep)
                if coords:
                    return TransformMatrix.translation(*coords)
        
        # Check for directional movement
        direction_vector = self._extract_direction_and_amount(verb_phrase)
        if direction_vector:
            return TransformMatrix.translation(*direction_vector)
        
        return None
    
    def _interpret_rotation(self, verb_phrase: VerbPhrase) -> Optional[TransformMatrix]:
        """
        Interpret rotation-related verb phrases.
        
        Examples:
        - "rotate by [0, 90, 0]" -> rotation_xyz(0, 90, 0)
        - "turn 45 degrees" -> rotation_z(45)
        - "rotate around x by 90 degrees" -> rotation_x(90)
        """
        # Check for coordinate vector in prepositional phrases
        for prep in verb_phrase.preps:
            if prep.preposition == "by":
                coords = self._extract_coordinates(prep)
                if coords:
                    return TransformMatrix.rotation_xyz(*coords)
        
        # Check for axis-specific rotation
        axis_rotation = self._extract_axis_rotation(verb_phrase)
        if axis_rotation:
            axis, degrees = axis_rotation
            if axis == "x":
                return TransformMatrix.rotation_x(degrees)
            elif axis == "y":
                return TransformMatrix.rotation_y(degrees)
            elif axis == "z":
                return TransformMatrix.rotation_z(degrees)
        
        # Check for simple rotation amount (defaults to Z-axis)
        degrees = self._extract_degrees(verb_phrase)
        if degrees is not None:
            return TransformMatrix.rotation_z(degrees)
        
        return None
    
    def _interpret_scaling(self, verb_phrase: VerbPhrase) -> Optional[TransformMatrix]:
        """
        Interpret scaling-related verb phrases.
        
        Examples:
        - "scale by [2, 1, 1]" -> scale(2, 1, 1)
        - "make it twice as big" -> uniform_scale(2)
        - "resize to half" -> uniform_scale(0.5)
        """
        # Check for coordinate vector in prepositional phrases
        for prep in verb_phrase.preps:
            if prep.preposition == "by":
                coords = self._extract_coordinates(prep)
                if coords:
                    return TransformMatrix.scale(*coords)
        
        # Check for uniform scaling
        scale_factor = self._extract_scale_factor(verb_phrase)
        if scale_factor is not None:
            return TransformMatrix.uniform_scale(scale_factor)
        
        return None
    
    def _extract_coordinates(self, prep: PrepositionalPhrase) -> Optional[Tuple[float, float, float]]:
        """
        Extract coordinate vector from a prepositional phrase.
        
        Examples:
        - "to [1, 2, 3]" -> (1.0, 2.0, 3.0)
        - "by [0, 90, 0]" -> (0.0, 90.0, 0.0)
        """
        if not prep.noun_phrase:
            return None
        
        # Check if the noun phrase contains a vector
        if hasattr(prep.noun_phrase, 'vector') and prep.noun_phrase.vector:
            # Look for vector literal pattern in the vector representation
            vector_str = str(prep.noun_phrase.vector)
            return self._parse_vector_literal(vector_str)
        
        return None
    
    def _parse_vector_literal(self, text: str) -> Optional[Tuple[float, float, float]]:
        """
        Parse a vector literal string like "[1, 2, 3]" or "(1.5, 2.0, 3.5)".
        
        Args:
            text: String that might contain a vector literal
            
        Returns:
            Tuple of three floats if parsing succeeds, None otherwise
        """
        # Pattern to match vector literals: [x, y, z] or (x, y, z)
        pattern = r'[\[\(]\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*[\]\)]'
        match = re.search(pattern, text)
        
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                return (x, y, z)
            except ValueError:
                pass
        
        return None
    
    def _extract_direction_and_amount(self, verb_phrase: VerbPhrase) -> Optional[Tuple[float, float, float]]:
        """
        Extract directional movement with amount.
        
        Examples:
        - "move up 2 units" -> (0, 2, 0)
        - "move left 3" -> (-3, 0, 0)
        - "move forward 1.5" -> (0, 0, 1.5)
        """
        # Look for direction keywords in prepositional phrases
        direction_map = {
            "up": (0, 1, 0),
            "down": (0, -1, 0),
            "left": (-1, 0, 0),
            "right": (1, 0, 0),
            "forward": (0, 0, 1),
            "backward": (0, 0, -1),
            "back": (0, 0, -1)
        }
        
        for prep in verb_phrase.preps:
            if prep.preposition in direction_map:
                direction = direction_map[prep.preposition]
                amount = self._extract_numeric_amount(prep)
                if amount is not None:
                    return (direction[0] * amount, direction[1] * amount, direction[2] * amount)
        
        return None
    
    def _extract_axis_rotation(self, verb_phrase: VerbPhrase) -> Optional[Tuple[str, float]]:
        """
        Extract axis-specific rotation.
        
        Examples:
        - "rotate around x by 90 degrees" -> ("x", 90.0)
        - "turn around y-axis 45 degrees" -> ("y", 45.0)
        """
        # Look for axis keywords
        axis_keywords = {"x": "x", "y": "y", "z": "z", "x-axis": "x", "y-axis": "y", "z-axis": "z"}
        
        for prep in verb_phrase.preps:
            if prep.preposition in ["around", "about"]:
                if prep.noun_phrase and prep.noun_phrase.noun:
                    noun = prep.noun_phrase.noun.lower()
                    if noun in axis_keywords:
                        axis = axis_keywords[noun]
                        degrees = self._extract_degrees(verb_phrase)
                        if degrees is not None:
                            return (axis, degrees)
        
        return None
    
    def _extract_degrees(self, verb_phrase: VerbPhrase) -> Optional[float]:
        """
        Extract angle in degrees from a verb phrase.
        
        Examples:
        - "rotate 90 degrees" -> 90.0
        - "turn 45°" -> 45.0
        """
        # Look for numeric amounts followed by degree indicators
        for prep in verb_phrase.preps:
            amount = self._extract_numeric_amount(prep)
            if amount is not None:
                # Check if it's likely to be degrees
                if prep.noun_phrase and prep.noun_phrase.noun:
                    noun = prep.noun_phrase.noun.lower()
                    if noun in ["degrees", "degree", "°"]:
                        return amount
        
        # Also check the amount field directly
        if hasattr(verb_phrase, 'amount') and verb_phrase.amount:
            amount = self._extract_numeric_amount_from_np(verb_phrase.amount)
            if amount is not None:
                return amount
        
        return None
    
    def _extract_scale_factor(self, verb_phrase: VerbPhrase) -> Optional[float]:
        """
        Extract scale factor from a verb phrase.
        
        Examples:
        - "make it twice as big" -> 2.0
        - "scale by 1.5" -> 1.5
        - "resize to half" -> 0.5
        """
        # Look for scale-related keywords
        scale_keywords = {
            "twice": 2.0,
            "double": 2.0,
            "half": 0.5,
            "quarter": 0.25,
            "triple": 3.0
        }
        
        # Check adjective complements
        for adj in verb_phrase.adjective_complement:
            if adj.lower() in scale_keywords:
                return scale_keywords[adj.lower()]
        
        # Check for numeric amounts
        for prep in verb_phrase.preps:
            if prep.preposition == "by":
                amount = self._extract_numeric_amount(prep)
                if amount is not None:
                    return amount
        
        return None
    
    def _extract_numeric_amount(self, prep: PrepositionalPhrase) -> Optional[float]:
        """Extract numeric amount from a prepositional phrase."""
        if not prep.noun_phrase:
            return None
        
        return self._extract_numeric_amount_from_np(prep.noun_phrase)
    
    def _extract_numeric_amount_from_np(self, noun_phrase: NounPhrase) -> Optional[float]:
        """Extract numeric amount from a noun phrase."""
        if not noun_phrase:
            return None
        
        # Check if the noun phrase has a numeric determiner
        if hasattr(noun_phrase, 'determiner') and noun_phrase.determiner:
            try:
                return float(noun_phrase.determiner)
            except ValueError:
                pass
        
        # Check if the noun itself is numeric
        if hasattr(noun_phrase, 'noun') and noun_phrase.noun:
            try:
                return float(noun_phrase.noun)
            except ValueError:
                pass
        
        # Check vector space for numeric projections
        if hasattr(noun_phrase, 'vector') and noun_phrase.vector:
            # Look for number-related projections
            try:
                number_value = noun_phrase.vector["number"]
                if number_value > 0:
                    return number_value
            except (KeyError, ValueError):
                pass
        
        return None
    
    def interpret_spatial_relationship(self, prep: PrepositionalPhrase) -> Optional[TransformMatrix]:
        """
        Interpret spatial relationship prepositions.
        
        Examples:
        - "above the box" -> translation(0, height_of_box, 0)
        - "to the right of the sphere" -> translation(radius_of_sphere, 0, 0)
        - "behind the cube" -> translation(0, 0, -depth_of_cube)
        
        Note: This is a simplified implementation. In a full system, this would
        need to query the scene for object dimensions and positions.
        """
        if not prep.preposition:
            return None
        
        # Default spatial offsets (would be calculated from object dimensions in full system)
        spatial_offsets = {
            "above": (0, 2, 0),
            "below": (0, -2, 0),
            "left": (-2, 0, 0),
            "right": (2, 0, 0),
            "behind": (0, 0, -2),
            "in front of": (0, 0, 2),
            "front": (0, 0, 2),
            "next to": (2, 0, 0),
            "beside": (2, 0, 0)
        }
        
        prep_lower = prep.preposition.lower()
        if prep_lower in spatial_offsets:
            offset = spatial_offsets[prep_lower]
            return TransformMatrix.translation(*offset)
        
        return None
