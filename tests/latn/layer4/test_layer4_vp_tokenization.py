#!/usr/bin/env python3
"""
Unit tests for LATN Layer 4 - Verb Phrase Tokenization

Tests the VP tokenization functionality that builds on Layers 1-3.
"""

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.latn_tokenizer_layer4 import latn_tokenize_layer4, VPTokenizationHypothesis
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.pos.verb_phrase import VerbPhrase


class TestLayer4VPTokenization:
    """Test Layer 4 VP tokenization functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.scene = SceneModel()
        self.executor = LATNLayerExecutor(self.scene)
    
    def test_simple_create_command(self):
        """Test: 'create a red box' -> VP token."""
        result = self.executor.execute_layer4("create a red box")
        
        assert result.success, "Layer 4 should succeed"
        assert len(result.hypotheses) > 0, "Should have VP hypotheses"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        print(f"VP replacements: {len(best.vp_replacements)}")
        
        # Should have VP replacements
        assert len(best.vp_replacements) > 0, "Should have VP replacements"
        
        # Should have VP tokens
        vp_tokens = [tok for tok in best.tokens if tok.isa("VP")]
        assert len(vp_tokens) > 0, "Should have VP tokens"
    
    def test_move_command(self):
        """Test: 'move the sphere' -> VP token."""
        result = self.executor.execute_layer4("move the sphere")
        
        assert result.success, "Layer 4 should succeed"
        assert len(result.hypotheses) > 0, "Should have VP hypotheses"
        
        best = result.hypotheses[0]
        vp_tokens = [tok for tok in best.tokens if tok.isa("VP")]
        assert len(vp_tokens) > 0, "Should have VP tokens"
        
        # Check that it identifies as a move action
        vp_token = vp_tokens[0]
        assert vp_token["move"] > 0, "Should be identified as move action"
    
    def test_verb_phrase_extraction(self):
        """Test extraction of VerbPhrase objects from tokenization."""
        result = self.executor.execute_layer4("create a blue cube")
        
        assert result.success, "Layer 4 should succeed"
        assert len(result.verb_phrases) > 0, "Should extract VerbPhrase objects"
        
        vp = result.verb_phrases[0]
        assert isinstance(vp, VerbPhrase), "Should be VerbPhrase object"
        assert vp.verb is not None, "Should have verb"
        assert vp.noun_phrase is not None, "Should have noun phrase"
    
    def test_action_verbs(self):
        """Test various action verbs are recognized."""
        action_commands = [
            ("create a box", "create"),
            ("make a sphere", "create"),
            ("build a cube", "create"),
            ("move the object", "move"),
            ("rotate the sphere", "rotate"),
            ("delete the box", "edit")  # delete operations use "edit" dimension
        ]
        
        for command, expected_action in action_commands:
            result = self.executor.execute_layer4(command)
            
            assert result.success, f"Command '{command}' should succeed"
            
            if result.hypotheses:
                best = result.hypotheses[0]
                vp_tokens = [tok for tok in best.tokens if tok.isa("VP")]
                
                if vp_tokens:
                    vp_token = vp_tokens[0]
                    assert vp_token[expected_action] > 0, f"Should be identified as {expected_action} action"
    
    def test_layer4_builds_on_layer3(self):
        """Test that Layer 4 properly uses Layer 3 results."""
        # First get Layer 3 result
        layer3_result = self.executor.execute_layer3("create a red box")
        assert layer3_result.success, "Layer 3 should succeed"
        
        # Then test Layer 4 with the same input
        layer4_result = self.executor.execute_layer4("create a red box")
        assert layer4_result.success, "Layer 4 should succeed"
        
        # Layer 4 should include Layer 3 results
        assert layer4_result.layer3_result.success, "Layer 4 should include successful Layer 3"
        assert len(layer4_result.layer3_result.hypotheses) > 0, "Should have Layer 3 hypotheses"


class TestLayer4ActionExecution:
    """Test Layer 4 action execution functionality."""
    
    def setup_method(self):
        """Set up test environment with scene."""
        self.scene = SceneModel()
        self.executor = LATNLayerExecutor(self.scene)
    
    def test_create_action_execution(self):
        """Test that create commands are properly tokenized and grounded, but do NOT create objects."""
        initial_count = len(self.scene.objects)
        
        result = self.executor.execute_layer4("create a red box")
        
        assert result.success, "Layer 4 should succeed"
        
        # Should NOT have created an object - Layer 4 only does semantic grounding
        final_count = len(self.scene.objects)
        assert final_count == initial_count, "Layer 4 should not create objects"
        
        # Should have identified the verb phrase semantics
        assert len(result.verb_phrases) > 0, "Should have extracted verb phrases"
        vp = result.verb_phrases[0]
        assert vp.verb.word == "create", "Should identify create verb"
        assert vp.noun_phrase is not None, "Should have noun phrase object"
    
    def test_multiple_create_commands(self):
        """Test tokenizing multiple commands, but not creating objects."""
        commands = [
            "create a red box",
            "make a blue sphere", 
            "build a green cube"
        ]
        
        initial_count = len(self.scene.objects)
        
        for command in commands:
            result = self.executor.execute_layer4(command)
            assert result.success, f"Command '{command}' should succeed"
            assert len(result.verb_phrases) > 0, f"Command '{command}' should extract verb phrases"
        
        # Should NOT have created objects - Layer 4 only does semantic grounding
        final_count = len(self.scene.objects)
        assert final_count == initial_count, "Layer 4 should not create objects"
    
    def test_action_execution_disabled(self):
        """Test Layer 4 semantic grounding without action execution (which is the default)."""
        initial_count = len(self.scene.objects)
        
        result = self.executor.execute_layer4("create a red box")
        
        assert result.success, "Layer 4 should succeed"
        
        # Should NOT have created an object - Layer 4 never creates objects
        final_count = len(self.scene.objects)
        assert final_count == initial_count, "Layer 4 should not create objects"
        
        # Should have identified verb phrase semantics
        assert len(result.verb_phrases) > 0, "Should have extracted verb phrases"
    
    def test_create_with_properties(self):
        """Test semantic grounding of verb phrases with specific properties."""
        test_cases = [
            ("create a large red sphere", "sphere", "red", "large"),
            ("make a small blue box", "box", "blue", "small"),
            ("build a green cube", "cube", "green", "normal")
        ]
        
        for command, expected_shape, expected_color, expected_size in test_cases:
            initial_count = len(self.scene.objects)
            
            result = self.executor.execute_layer4(command)
            assert result.success, f"Command '{command}' should succeed"
            
            # Should NOT have created objects - Layer 4 only does semantic grounding
            final_count = len(self.scene.objects)
            assert final_count == initial_count, f"Should not create objects for '{command}'"
            
            # Should have extracted verb phrases with semantic information
            assert len(result.verb_phrases) > 0, f"Should extract verb phrases for '{command}'"
            vp = result.verb_phrases[0]
            assert vp.verb.word in ["create", "make", "build"], f"Should identify action verb for '{command}'"
            assert vp.noun_phrase is not None, f"Should have noun phrase for '{command}'"


class TestLayer4Integration:
    """Test Layer 4 integration with other layers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.scene = SceneModel()
        self.executor = LATNLayerExecutor(self.scene)
    
    def test_layer4_with_layer2_grounding(self):
        """Test Layer 4 verb phrase extraction with Layer 2 grounding of existing objects."""
        # Add some objects to the scene first (not through Layer 4)
        from engraf.lexer.vector_space import vector_from_features
        from engraf.visualizer.scene.scene_object import SceneObject
        
        red_box_vector = vector_from_features("noun", red=1.0)
        blue_sphere_vector = vector_from_features("noun", blue=1.0) 
        
        red_box = SceneObject("box", red_box_vector, object_id="red_box_1")
        blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere_1")
        
        self.scene.add_object(red_box)
        self.scene.add_object(blue_sphere)
        
        # Test Layer 4 verb phrase extraction
        create_commands = [
            "create a red box", 
            "make a blue sphere"
        ]
        
        for command in create_commands:
            result = self.executor.execute_layer4(command)
            assert result.success, f"Layer 4 should succeed: {command}"
            assert len(result.verb_phrases) > 0, f"Should extract verb phrases: {command}"
        
        # Now test Layer 2 grounding against the existing objects
        grounding_phrases = [
            "the red box",
            "the blue sphere"
        ]
        
        for phrase in grounding_phrases:
            result = self.executor.execute_layer2(phrase)
            assert result.success, f"Layer 2 should succeed: {phrase}"
            
            # Should find grounding results
            assert len(result.grounding_results) > 0, f"Should have grounding results for: {phrase}"            # At least one should be successful
            successful_groundings = [gr for gr in result.grounding_results if gr.success]
            assert len(successful_groundings) > 0, f"Should have successful grounding for: {phrase}"
    
    def test_layer4_confidence_propagation(self):
        """Test that confidence scores propagate correctly through layers."""
        result = self.executor.execute_layer4("create a red box")
        
        assert result.success, "Layer 4 should succeed"
        assert 0.0 < result.confidence <= 1.0, "Should have valid confidence score"
        
        # Layer 4 confidence should be based on Layer 3 confidence
        assert result.layer3_result.success, "Layer 3 should succeed"
        assert result.layer3_result.confidence > 0.0, "Layer 3 should have confidence"
    
    def test_layer4_error_handling(self):
        """Test Layer 4 error handling with invalid input."""
        # Test with non-verb input
        result = self.executor.execute_layer4("red box green")
        
        # Should still succeed at tokenization level, but may not find VPs
        # The exact behavior depends on implementation - document what we expect
        print(f"Result for non-VP input: success={result.success}, VP count={len(result.verb_phrases)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
