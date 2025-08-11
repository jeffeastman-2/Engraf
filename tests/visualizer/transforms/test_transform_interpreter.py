"""
Tests for the transform interpreter module.

Tests the conversion of linguistic structures to transformation matrices,
including movement, rotation, scaling, and spatial relationships.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from engraf.visualizer.transforms.transform_interpreter import TransformInterpreter
from engraf.visualizer.transforms.transform_matrix import TransformMatrix
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.lexer.vocabulary_builder import get_from_vocabulary


class TestTransformInterpreter:
    """Test suite for the TransformInterpreter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interpreter = TransformInterpreter()
    
    def test_initialization(self):
        """Test that the transform interpreter initializes correctly."""
        interpreter = TransformInterpreter()
        assert interpreter is not None
    
    def test_interpret_verb_phrase_none_verb(self):
        """Test that verb phrases without verbs return None."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = None
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        assert result is None
    
    def test_interpret_verb_phrase_unknown_verb(self):
        """Test that unknown verbs return None."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "unknown"
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        assert result is None
    
    def test_interpret_movement_to_coordinates(self):
        """Test interpreting movement to specific coordinates."""
        # Create mock prepositional phrase with coordinates
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "to"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.vector = Mock()
        prep.noun_phrase.vector.__str__ = Mock(return_value="[1, 2, 3]")
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "move"
        verb_phrase.preps = [prep]
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Check that the translation is correct
        expected = TransformMatrix.translation(1, 2, 3)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_movement_directional(self):
        """Test interpreting directional movement."""
        # Create mock prepositional phrase for direction
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "up"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.determiner = "2"
        prep.noun_phrase.noun = "units"
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "move"
        verb_phrase.preps = [prep]
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Check that the translation is correct (up 2 units)
        expected = TransformMatrix.translation(0, 2, 0)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_rotation_by_coordinates(self):
        """Test interpreting rotation by coordinate vector."""
        # Create mock prepositional phrase with rotation coordinates
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "by"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.vector = Mock()
        prep.noun_phrase.vector.__str__ = Mock(return_value="[0, 90, 0]")
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "rotate"
        verb_phrase.preps = [prep]
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Check that the rotation is correct
        expected = TransformMatrix.rotation_xyz(0, 90, 0)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_rotation_around_axis(self):
        """Test interpreting rotation around specific axis."""
        # Create mock prepositional phrase for axis
        prep_axis = Mock(spec=PrepositionalPhrase)
        prep_axis.preposition = "around"
        prep_axis.noun_phrase = Mock(spec=NounPhrase)
        prep_axis.noun_phrase.noun = "x"
        
        # Create mock prepositional phrase for amount
        prep_amount = Mock(spec=PrepositionalPhrase)
        prep_amount.preposition = "by"
        prep_amount.noun_phrase = Mock(spec=NounPhrase)
        prep_amount.noun_phrase.determiner = "90"
        prep_amount.noun_phrase.noun = "degrees"
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "rotate"
        verb_phrase.preps = [prep_axis, prep_amount]
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Check that the rotation is correct
        expected = TransformMatrix.rotation_x(90)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_scaling_by_coordinates(self):
        """Test interpreting scaling by coordinate vector."""
        # Create mock prepositional phrase with scale coordinates
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "by"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.vector = Mock()
        prep.noun_phrase.vector.__str__ = Mock(return_value="[2, 1, 1]")
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "scale"
        verb_phrase.preps = [prep]
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Check that the scaling is correct
        expected = TransformMatrix.scale(2, 1, 1)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_parse_vector_literal_brackets(self):
        """Test parsing vector literals with brackets."""
        text = "[1.5, 2.0, 3.5]"
        result = self.interpreter._parse_vector_literal(text)
        
        assert result == (1.5, 2.0, 3.5)
    
    def test_parse_vector_literal_parentheses(self):
        """Test parsing vector literals with parentheses."""
        text = "(1, 2, 3)"
        result = self.interpreter._parse_vector_literal(text)
        
        assert result == (1.0, 2.0, 3.0)
    
    def test_parse_vector_literal_negative(self):
        """Test parsing vector literals with negative numbers."""
        text = "[-1.5, 2.0, -3.5]"
        result = self.interpreter._parse_vector_literal(text)
        
        assert result == (-1.5, 2.0, -3.5)
    
    def test_parse_vector_literal_invalid(self):
        """Test parsing invalid vector literals."""
        text = "not a vector"
        result = self.interpreter._parse_vector_literal(text)
        
        assert result is None
    
    def test_extract_direction_and_amount(self):
        """Test extracting direction and amount from verb phrase."""
        # Create mock prepositional phrase for direction
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "up"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.determiner = "3"
        prep.noun_phrase.noun = "units"
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.preps = [prep]
        
        result = self.interpreter._extract_direction_and_amount(verb_phrase)
        
        assert result == (0, 3, 0)
    
    def test_extract_direction_and_amount_left(self):
        """Test extracting leftward direction and amount."""
        # Create mock prepositional phrase for direction
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "left"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.determiner = "2"
        prep.noun_phrase.noun = "units"
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.preps = [prep]
        
        result = self.interpreter._extract_direction_and_amount(verb_phrase)
        
        assert result == (-2, 0, 0)
    
    def test_extract_axis_rotation(self):
        """Test extracting axis-specific rotation."""
        # Create mock prepositional phrase for axis
        prep_axis = Mock(spec=PrepositionalPhrase)
        prep_axis.preposition = "around"
        prep_axis.noun_phrase = Mock(spec=NounPhrase)
        prep_axis.noun_phrase.noun = "y"
        
        # Create mock prepositional phrase for amount
        prep_amount = Mock(spec=PrepositionalPhrase)
        prep_amount.preposition = "by"
        prep_amount.noun_phrase = Mock(spec=NounPhrase)
        prep_amount.noun_phrase.determiner = "45"
        prep_amount.noun_phrase.noun = "degrees"
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.preps = [prep_axis, prep_amount]
        
        result = self.interpreter._extract_axis_rotation(verb_phrase)
        
        assert result == ("y", 45.0)
    
    def test_extract_degrees(self):
        """Test extracting degrees from verb phrase."""
        # Create mock prepositional phrase with degrees
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "by"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.determiner = "90"
        prep.noun_phrase.noun = "degrees"
        
        # Create mock verb phrase
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.preps = [prep]
        
        result = self.interpreter._extract_degrees(verb_phrase)
        
        assert result == 90.0
    
    def test_extract_numeric_amount_from_np_determiner(self):
        """Test extracting numeric amount from noun phrase determiner."""
        noun_phrase = Mock(spec=NounPhrase)
        noun_phrase.determiner = "3.5"
        noun_phrase.noun = "units"
        
        result = self.interpreter._extract_numeric_amount_from_np(noun_phrase)
        
        assert result == 3.5
    
    def test_extract_numeric_amount_from_np_noun(self):
        """Test extracting numeric amount from noun phrase noun."""
        noun_phrase = Mock(spec=NounPhrase)
        noun_phrase.determiner = None
        noun_phrase.noun = "2"
        
        result = self.interpreter._extract_numeric_amount_from_np(noun_phrase)
        
        assert result == 2.0
    
    def test_extract_numeric_amount_from_np_vector(self):
        """Test extracting numeric amount from noun phrase vector."""
        noun_phrase = Mock(spec=NounPhrase)
        noun_phrase.determiner = None
        noun_phrase.noun = "units"
        noun_phrase.vector = Mock()
        noun_phrase.vector.__getitem__ = Mock(return_value=4.5)
        
        result = self.interpreter._extract_numeric_amount_from_np(noun_phrase)
        
        assert result == 4.5
    
    def test_extract_numeric_amount_from_np_none(self):
        """Test extracting numeric amount from None noun phrase."""
        result = self.interpreter._extract_numeric_amount_from_np(None)
        
        assert result is None
    
    def test_interpret_spatial_relationship_above(self):
        """Test interpreting spatial relationship 'above'."""
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "above"
        
        result = self.interpreter.interpret_spatial_relationship(prep)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        expected = TransformMatrix.translation(0, 2, 0)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_spatial_relationship_right(self):
        """Test interpreting spatial relationship 'right'."""
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "right"
        
        result = self.interpreter.interpret_spatial_relationship(prep)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        expected = TransformMatrix.translation(2, 0, 0)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_spatial_relationship_behind(self):
        """Test interpreting spatial relationship 'behind'."""
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "behind"
        
        result = self.interpreter.interpret_spatial_relationship(prep)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        expected = TransformMatrix.translation(0, 0, -2)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_spatial_relationship_unknown(self):
        """Test interpreting unknown spatial relationship."""
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "unknown"
        
        result = self.interpreter.interpret_spatial_relationship(prep)
        
        assert result is None
    
    def test_interpret_spatial_relationship_none_preposition(self):
        """Test interpreting spatial relationship with None preposition."""
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = None
        
        result = self.interpreter.interpret_spatial_relationship(prep)
        
        assert result is None
    
    def test_interpret_verb_semantics_transform(self):
        """Test semantic interpretation of transform verbs."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "move"
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is not None
        assert result["verb"] == "move"
        assert result["is_action"] == True
        assert result["category"] == "move"
        assert result["affects_position"] == True
        assert "vector" in result
    
    def test_interpret_verb_semantics_create(self):
        """Test semantic interpretation of create verbs."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "create"
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is not None
        assert result["verb"] == "create"
        assert result["is_action"] == True
        assert result["category"] == "create"
        assert result["creates_object"] == True
    
    def test_interpret_verb_semantics_style(self):
        """Test semantic interpretation of style verbs."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "color"
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is not None
        assert result["verb"] == "color"
        assert result["is_action"] == True
        assert result["category"] == "style"
        assert result["affects_appearance"] == True
    
    def test_interpret_verb_semantics_edit(self):
        """Test semantic interpretation of edit verbs."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "delete"
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is not None
        assert result["verb"] == "delete"
        assert result["is_action"] == True
        assert result["category"] == "edit"
        assert result["modifies_object"] == True
    
    def test_interpret_verb_semantics_organize(self):
        """Test semantic interpretation of organize verbs."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "align"
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is not None
        assert result["verb"] == "align"
        assert result["is_action"] == True
        assert result["category"] == "organize"
        assert result["affects_layout"] == True
    
    def test_interpret_verb_semantics_select(self):
        """Test semantic interpretation of select verbs."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "select"
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is not None
        assert result["verb"] == "select"
        assert result["is_action"] == True
        assert result["category"] == "select"
        assert result["affects_selection"] == True
    
    def test_interpret_verb_semantics_unknown(self):
        """Test semantic interpretation of unknown verbs."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "unknown_verb"
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is None
    
    def test_interpret_verb_semantics_none_verb(self):
        """Test semantic interpretation with None verb."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = None
        
        result = self.interpreter.interpret_verb_semantics(verb_phrase)
        
        assert result is None
    
    def test_interpret_xrotate_verb(self):
        """Test interpretation of xrotate verb."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "xrotate"
        verb_phrase.preps = []
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Should default to 90 degrees
        expected = TransformMatrix.rotation_x(90)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_yrotate_verb(self):
        """Test interpretation of yrotate verb."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "yrotate"
        verb_phrase.preps = []
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Should default to 90 degrees
        expected = TransformMatrix.rotation_y(90)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_zrotate_verb(self):
        """Test interpretation of zrotate verb."""
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "zrotate"
        verb_phrase.preps = []
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        # Should default to 90 degrees
        expected = TransformMatrix.rotation_z(90)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_interpret_xrotate_with_degrees(self):
        """Test interpretation of xrotate verb with specific degrees."""
        # Create mock prepositional phrase with degrees
        prep = Mock(spec=PrepositionalPhrase)
        prep.preposition = "by"
        prep.noun_phrase = Mock(spec=NounPhrase)
        prep.noun_phrase.determiner = "45"
        prep.noun_phrase.noun = "degrees"
        
        verb_phrase = Mock(spec=VerbPhrase)
        verb_phrase.verb = "xrotate"
        verb_phrase.preps = [prep]
        
        result = self.interpreter.interpret_verb_phrase(verb_phrase)
        
        assert result is not None
        assert isinstance(result, TransformMatrix)
        expected = TransformMatrix.rotation_x(45)
        assert np.allclose(result.matrix, expected.matrix)
    
    def test_is_transform_verb(self):
        """Test the _is_transform_verb method."""
        # Test with transform verb
        transform_vector = get_from_vocabulary("move")
        assert self.interpreter._is_transform_verb(transform_vector) == True
        
        # Test with non-transform verb
        create_vector = get_from_vocabulary("create")
        assert self.interpreter._is_transform_verb(create_vector) == False
        
        # Test with None
        assert self.interpreter._is_transform_verb(None) == False
    
    def test_get_verb_category(self):
        """Test the _get_verb_category method."""
        # Test move verb
        transform_vector = get_from_vocabulary("move")
        assert self.interpreter._get_verb_category(transform_vector) == "move"
        
        # Test create verb
        create_vector = get_from_vocabulary("create")
        assert self.interpreter._get_verb_category(create_vector) == "create"
        
        # Test style verb
        style_vector = get_from_vocabulary("color")
        assert self.interpreter._get_verb_category(style_vector) == "style"
        
        # Test edit verb
        edit_vector = get_from_vocabulary("delete")
        assert self.interpreter._get_verb_category(edit_vector) == "edit"
        
        # Test organize verb
        organize_vector = get_from_vocabulary("align")
        assert self.interpreter._get_verb_category(organize_vector) == "organize"
        
        # Test select verb
        select_vector = get_from_vocabulary("select")
        assert self.interpreter._get_verb_category(select_vector) == "select"
        
        # Test with None
        assert self.interpreter._get_verb_category(None) is None

    # ...existing tests...
