#!/usr/bin/env python3
"""
Test LATN Layer 3 Semantic Grounding

Tests for the Layer 3 semantic grounding capabilities.
"""

import pytest
from engraf.lexer.semantic_grounding_layer3 import Layer3SemanticGrounder, Layer3GroundingResult
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE


class TestLayer3SemanticGrounding:
    """Test Layer 3 semantic grounding functionality."""
    
    def setup_method(self):
        """Set up test scene with objects for grounding tests."""
        self.scene = SceneModel()

        # Create test objects
        table_vector = vector_from_features("noun")
        self.table = SceneObject("table", table_vector, object_id="table_1")
        self.scene.add_object(self.table)

        self.grounder = Layer3SemanticGrounder()  # Layer 3 doesn't take scene_model        # Ensure vocabulary has prepositions
        self.original_vocab = {}
        test_words = {
            'on': vector_from_features("prep spatial_location", locY=0.5),
            'above': vector_from_features("prep spatial_location", locY=1.0),
            'the': vector_from_features("det", singular=1.0),
            'table': vector_from_features("noun")
        }
        
        for word, vector in test_words.items():
            self.original_vocab[word] = word in SEMANTIC_VECTOR_SPACE
            if not self.original_vocab[word]:
                SEMANTIC_VECTOR_SPACE[word] = vector
    
    def teardown_method(self):
        """Clean up test vocabulary."""
        for word, was_original in self.original_vocab.items():
            if not was_original and word in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE[word]
    
    def test_ground_vector_location(self):
        """Test grounding a PP with vector coordinates."""
        pp = PrepositionalPhrase()
        pp.preposition = "at"
        pp.vector_text = "[1,2,3]"
        pp.vector = vector_from_features("prep", locX=1.0, locY=2.0, locZ=3.0)
        
        result = self.grounder.ground(pp)
        
        assert result.success
        assert result.confidence == 1.0
        assert result.resolved_object is not None
        assert "[1,2,3]" in result.resolved_object.word
    
    def test_ground_spatial_relationship(self):
        """Test grounding a PP with spatial relationship to an object."""
        # Create NP for "the table"
        np = NounPhrase("table")
        np.determiner = "the"
        np.vector = vector_from_features("noun")
        
        # Create PP "on the table"
        pp = PrepositionalPhrase()
        pp.preposition = "on"
        pp.noun_phrase = np
        
        result = self.grounder.ground(pp)
        
        assert result.success
        assert result.resolved_object is not None
        assert hasattr(result.resolved_object, '_reference_object')
        assert result.resolved_object._reference_object.object_id == "table_1"
        assert result.resolved_object._preposition == "on"
    
    def test_ground_spatial_relationship_no_object(self):
        """Test grounding a PP when the referenced object doesn't exist."""
        # Create NP for "the chair" (not in scene)
        np = NounPhrase("chair")
        np.determiner = "the"
        np.vector = vector_from_features("noun")
        
        # Create PP "on the chair"
        pp = PrepositionalPhrase()
        pp.preposition = "on"
        pp.noun_phrase = np
        
        result = self.grounder.ground(pp)
        
        assert not result.success
        assert result.confidence == 0.0
        assert "Failed to ground NP within PP" in result.description
    
    def test_ground_invalid_pp(self):
        """Test grounding a PP without proper structure."""
        pp = PrepositionalPhrase()
        pp.preposition = "on"
        # No np or vector_text
        
        result = self.grounder.ground(pp)
        
        assert not result.success
        assert "missing vector or NP object" in result.description
    
    def test_ground_multiple_pps(self):
        """Test grounding multiple PPs at once."""
        # Create PP with vector location
        pp1 = PrepositionalPhrase()
        pp1.preposition = "at"
        pp1.vector_text = "[0,0,0]"
        pp1.vector = vector_from_features("prep", locX=0.0, locY=0.0, locZ=0.0)
        
        # Create PP with spatial relationship
        np = NounPhrase("table")
        np.determiner = "the"
        np.vector = vector_from_features("noun")
        
        pp2 = PrepositionalPhrase()
        pp2.preposition = "above"
        pp2.noun_phrase = np
        
        results = self.grounder.ground_multiple([pp1, pp2])
        
        assert len(results) == 2
        assert results[0].success  # Vector location should succeed
        assert results[1].success  # Spatial relationship should succeed
    
    def test_invalid_input_type(self):
        """Test grounding with invalid input type."""
        result = self.grounder.ground("not a prepositional phrase")
        
        assert not result.success
        assert result.confidence == 0.0
        assert "Expected PrepositionalPhrase" in result.description
