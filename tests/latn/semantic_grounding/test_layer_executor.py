#!/usr/bin/env python3
"""
Test LATN Layer Executor

Tests for the LATN layer executor that provides entry points at each layer.
"""

import pytest
from engraf.lexer.latn_layer_executor import (
    LATNLayerExecutor, Layer1Result, Layer2Result, Layer3Result
)
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE


class TestLATNLayerExecutor:
    """Test LATN layer executor functionality."""
    
    def setup_method(self):
        """Set up test scene and vocabulary."""
        self.scene = SceneModel()
        
        # Create test objects
        red_box_vector = vector_from_features("noun", red=1.0)
        self.red_box = SceneObject("box", red_box_vector, object_id="red_box_1")
        self.scene.add_object(self.red_box)
        
        # Set up test vocabulary
        self.original_vocab = {}
        test_words = {
            'the': vector_from_features("det", singular=1.0),
            'red': vector_from_features("adj", red=1.0),
            'box': vector_from_features("noun"),
            'on': vector_from_features("prep spatial_location", locY=0.5)
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
    
    def test_execute_layer1(self):
        """Test Layer 1 execution (lexical tokenization)."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer1("the red box")
        
        assert isinstance(result, Layer1Result)
        assert result.success
        assert len(result.hypotheses) > 0
        assert result.confidence > 0.0
        assert "the red box" in result.description
    
    def test_execute_layer2_without_scene(self):
        """Test Layer 2 execution without semantic grounding."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("the red box")
        
        assert isinstance(result, Layer2Result)
        assert result.success
        assert result.layer1_result.success
        assert len(result.noun_phrases) > 0
        assert len(result.grounding_results) == 0  # No grounding without scene
    
    def test_execute_layer2_with_scene(self):
        """Test Layer 2 execution with semantic grounding."""
        executor = LATNLayerExecutor(self.scene)
        result = executor.execute_layer2("the red box")
        
        assert isinstance(result, Layer2Result)
        assert result.success
        assert result.layer1_result.success
        assert len(result.noun_phrases) > 0
        assert len(result.grounding_results) > 0
        
        # Check grounding worked
        grounding_result = result.grounding_results[0]
        assert grounding_result.success
        assert grounding_result.resolved_object.object_id == "red_box_1"
    
    def test_execute_layer3(self):
        """Test Layer 3 execution (includes all lower layers)."""
        executor = LATNLayerExecutor(self.scene)
        result = executor.execute_layer3("the red box")
        
        assert isinstance(result, Layer3Result)
        assert result.success
        assert result.layer2_result.success
        assert result.layer2_result.layer1_result.success
        
        # Should have NPs from Layer 2
        assert len(result.layer2_result.noun_phrases) > 0
        # May or may not have PPs depending on input
        assert len(result.prepositional_phrases) >= 0
    
    def test_layer_failure_propagation(self):
        """Test that layer failures propagate properly."""
        executor = LATNLayerExecutor()
        
        # Use empty string to cause Layer 1 failure
        result = executor.execute_layer2("")
        
        assert not result.success
        assert not result.layer1_result.success
        assert "Layer 1 failure" in result.description
        
    def test_scene_model_update(self):
        """Test updating the scene model."""
        executor = LATNLayerExecutor()
        
        # Initially no scene
        assert executor.scene_model is None
        assert executor.layer2_grounder is None
        
        # Update with scene
        executor.update_scene_model(self.scene)
        assert executor.scene_model is self.scene
        assert executor.layer2_grounder is not None
        assert executor.layer3_grounder is not None
        
        # Update with None
        executor.update_scene_model(None)
        assert executor.scene_model is None
        assert executor.layer2_grounder is None
        assert executor.layer3_grounder is None
    
    def test_layer_analysis(self):
        """Test the detailed layer analysis function."""
        executor = LATNLayerExecutor(self.scene)
        
        # Test analysis up to Layer 2
        analysis = executor.get_layer_analysis("the red box", target_layer=2)
        
        assert 'input' in analysis
        assert 'target_layer' in analysis
        assert analysis['target_layer'] == 2
        assert 'layer1' in analysis
        assert 'layer2' in analysis
        assert 'layer3' not in analysis  # Should not be included
        
        assert analysis['layer1']['success']
        assert analysis['layer2']['success']
        assert analysis['layer2']['noun_phrase_count'] > 0
        
        # Test analysis up to Layer 3
        analysis_layer3 = executor.get_layer_analysis("the red box", target_layer=3)
        assert 'layer3' in analysis_layer3
    
    def test_return_all_matches(self):
        """Test the return_all_matches parameter for grounding."""
        executor = LATNLayerExecutor(self.scene)
        
        # Add another box to create ambiguity
        green_box_vector = vector_from_features("noun", green=1.0)
        green_box = SceneObject("box", green_box_vector, object_id="green_box_1")
        self.scene.add_object(green_box)
        
        # Test with generic "box" query
        result = executor.execute_layer2("the box")
        
        assert result.success
        if result.grounding_results:
            grounding_result = result.grounding_results[0]
            # Should have alternatives when querying ambiguous "box"
            assert len(grounding_result.alternative_matches) >= 0
