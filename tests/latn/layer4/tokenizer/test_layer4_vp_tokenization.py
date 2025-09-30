#!/usr/bin/env python3
"""
Unit tests for LATN Layer 4 - Verb Phrase Tokenization

Tests the VP tokenization functionality that builds on Layers 1-3.
"""

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
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
        result = self.executor.execute_layer4("create a red box", tokenize_only=True, report=True)
        
        assert result.success, "Layer 4 should succeed"
        assert len(result.hypotheses) > 0, "Should have VP hypotheses"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        # Should have VP tokens
        vp_tokens = [tok for tok in best.tokens if tok.isa("VP")]
        assert len(vp_tokens) == 1, "Should have VP tokens"
    
    def test_move_command(self):
        """Test: 'move the sphere' -> VP token."""
        result = self.executor.execute_layer4("move the sphere", tokenize_only=True, report=True)
        
        assert result.success, "Layer 4 should succeed"
        assert len(result.hypotheses) > 0, "Should have VP hypotheses"
        
        best = result.hypotheses[0]
        vp_tokens = [tok for tok in best.tokens if tok.isa("VP")]
        assert len(vp_tokens) == 1, "Should have VP tokens"
        
        # Check that it identifies as a move action
        vp_token = vp_tokens[0]
        assert vp_token["move"] > 0, "Should be identified as move action"
    
    def test_verb_phrase_extraction(self):
        """Test extraction of VerbPhrase objects from tokenization."""
        result = self.executor.execute_layer4("create a blue cube", tokenize_only=True, report=True)
        
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
        assert vp.verb == "create", "Should identify create verb"
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
            result = self.executor.execute_layer4(command, tokenize_only=True, report=True)
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
            
            result = self.executor.execute_layer4(command, tokenize_only=True, report=True)
            assert result.success, f"Command '{command}' should succeed"
            
            # Should NOT have created objects - Layer 4 only does semantic grounding
            final_count = len(self.scene.objects)
            assert final_count == initial_count, f"Should not create objects for '{command}'"
            
            # Should have extracted verb phrases with semantic information
            assert len(result.verb_phrases) > 0, f"Should extract verb phrases for '{command}'"
            vp = result.verb_phrases[0]
            assert vp.verb in ["create", "make", "build"], f"Should identify action verb for '{command}'"
            assert vp.noun_phrase is not None, f"Should have noun phrase for '{command}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
