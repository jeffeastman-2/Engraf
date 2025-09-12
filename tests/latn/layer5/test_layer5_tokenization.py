#!/usr/bin/env python3
"""
Unit tests for LATN Layer 5 - Sentence Phrase Tokenization

Tests the Sentence tokenization functionality that builds on Layers 1-4.
"""

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.pos.verb_phrase import VerbPhrase


class TestLayer5VPTokenization:
    """Test Layer 5 VP tokenization functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.scene = SceneModel()
        self.executor = LATNLayerExecutor()
    
    def test_simple_create_command(self):
        """Test: 'create a red box' -> VP token."""
        result = self.executor.execute_layer5("create a red box")

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have VP hypotheses"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
    
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
        """Test extraction of Sentence objects from tokenization."""
        result = self.executor.execute_layer5("create a blue cube")

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 1, "Should extract Sentence objects"

        hyp = result.hypotheses[0]
        sent = hyp.tokens[0]._original_sp
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is None, "Should not have subject"
        assert sent.predicate is not None, "Should have predicate"


    def test_layer5_builds_on_layer4(self):
        """Test that Layer 5 properly uses Layer 4 results."""
        # First get Layer 4 result
        layer4_result = self.executor.execute_layer4("create a red box")
        assert layer4_result.success, "Layer 4 should succeed"

        # Then test Layer 5 with the same input
        layer5_result = self.executor.execute_layer5("create a red box")
        assert layer5_result.success, "Layer 5 should succeed"

        # Layer 5 should include Layer 4 results
        assert layer5_result.layer4_result.success, "Layer 5 should include successful Layer 4"
        assert len(layer4_result.layer3_result.hypotheses) > 0, "Should have Layer 3 hypotheses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
