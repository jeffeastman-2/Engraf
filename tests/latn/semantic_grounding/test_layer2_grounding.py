#!/usr/bin/env python3
"""
Test LATN Layer 2 Semantic Grounding

Tests for the Layer 2 semantic grounding capabilities.
"""

import pytest
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder, Layer2GroundingResult
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features
from engraf.pos.noun_phrase import NounPhrase
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE


class TestLayer2SemanticGrounding:
    """Test Layer 2 semantic grounding functionality."""
    
    def setup_method(self):
        """Set up test scene with objects for grounding tests."""
        self.scene = SceneModel()
        
        # Create test objects
        red_box_vector = vector_from_features("noun", red=1.0)
        green_box_vector = vector_from_features("noun", green=1.0)
        blue_sphere_vector = vector_from_features("noun", blue=1.0)
        
        self.red_box = SceneObject("box", red_box_vector, object_id="red_box_1")
        self.green_box = SceneObject("box", green_box_vector, object_id="green_box_1")
        self.blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere_1")
        
        self.scene.add_object(self.red_box)
        self.scene.add_object(self.green_box)
        self.scene.add_object(self.blue_sphere)
        
        self.grounder = Layer2SemanticGrounder(self.scene)
    
    def test_ground_specific_np(self):
        """Test grounding a specific NP that should match one object."""
        # Create a red box NP
        np = NounPhrase("box")
        np.determiner = "the"
        np.vector = vector_from_features("noun", red=1.0)
        
        result = self.grounder.ground(np)
        
        assert result.success
        assert result.resolved_object is not None
        assert result.resolved_object.object_id == "red_box_1"
        assert result.confidence > 0.0
    
    def test_ground_ambiguous_np(self):
        """Test grounding an ambiguous NP that could match multiple objects."""
        # Create a generic box NP
        np = NounPhrase("box")
        np.determiner = "the"
        np.vector = vector_from_features("noun def")
        
        result = self.grounder.ground(np, return_all_matches=True)
        
        assert result.success
        assert result.resolved_object is not None
        assert len(result.alternative_matches) >= 1  # Should have alternative box matches
        
        # Should match a box (either red or green)
        assert result.resolved_object.name == "box"
    
    def test_ground_no_match(self):
        """Test grounding an NP that doesn't match any scene objects."""
        # Create a cylinder NP (not in scene)
        np = NounPhrase("cylinder")
        np.determiner = "the"
        np.vector = vector_from_features("noun")
        
        result = self.grounder.ground(np)
        
        assert not result.success
        assert result.resolved_object is None
        assert result.confidence == 0.0
    
    def test_ground_multiple_nps(self):
        """Test grounding multiple NPs at once."""
        # Create multiple NPs
        np1 = NounPhrase("box")
        np1.determiner = "the"
        np1.vector = vector_from_features("noun", red=1.0)
        
        np2 = NounPhrase("sphere")
        np2.determiner = "the"
        np2.vector = vector_from_features("noun", blue=1.0)
        
        results = self.grounder.ground_multiple([np1, np2])
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].resolved_object.object_id == "red_box_1"
        assert results[1].resolved_object.object_id == "blue_sphere_1"
    
    def test_invalid_input_type(self):
        """Test grounding with invalid input type."""
        result = self.grounder.ground("not a noun phrase")
        
        assert not result.success
        assert result.confidence == 0.0
        assert "Expected NounPhrase" in result.description
