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
from tests.latn.dummy_test_scene import DummyTestScene


class TestLayer5VPTokenization:
    """Test Layer 5 VP tokenization functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.executor = LATNLayerExecutor(DummyTestScene().get_scene1())
    
    def test_simple_create_command(self):
        """Test: 'create a red box at [2,2,2]' -> SP token."""
        result = self.executor.execute_layer5("create a red box at [2,2,2]", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have SP hypotheses"
        
        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is None, "Should not have subject"
        assert sent.predicate is not None, "Should have predicate"
        assert isinstance(sent.predicate, VerbPhrase), "Predicate should be VerbPhrase"

    def test_move_command(self):
        """Test: 'move the sphere above the box above the table' -> SP token."""
        result = self.executor.execute_layer5("move the sphere above the box above the table", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have SP hypotheses"
        
        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is None, "Should not have subject"
        assert sent.predicate is not None, "Should have predicate"
        assert isinstance(sent.predicate, VerbPhrase), "Predicate should be VerbPhrase"        
    
    def test_verb_phrase_extraction(self):
        """Test extraction of Sentence objects from tokenization."""
        result = self.executor.execute_layer5("create a blue cube", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 1, "Should extract Sentence objects"

        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is None, "Should not have subject"
        assert sent.predicate is not None, "Should have predicate"

    def test_declarative_sentence(self):
        """Test: 'the box is on the table' -> SP token."""
        result = self.executor.execute_layer5("a box is on the table", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have SP hypotheses"
        
        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is not None, "Should have subject"
        assert sent.predicate is not None, "Should have predicate"
        assert isinstance(sent.predicate, VerbPhrase), "Predicate should be VerbPhrase" 
        assert sent.predicate.vector.isa("tobe"), "Predicate should include negation"
        
    def test_declarative_sentence2(self):
        """Test: 'the box is not on the table' -> SP token."""
        result = self.executor.execute_layer5("a box is not on the table", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have SP hypotheses"
        
        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is not None, "Should have subject"
        assert sent.predicate is not None, "Should have predicate"
        assert isinstance(sent.predicate, VerbPhrase), "Predicate should be VerbPhrase" 
        assert sent.predicate.vector.isa("neg"), "Predicate should include negation"

    def test_interrogative_sentence(self):   
        """Test: 'is a box on the table' -> SP token."""
        result = self.executor.execute_layer5("is a box on the table", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have SP hypotheses"
        
        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        vp = sent.predicate
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is None, "Should have subject"
        assert vp is not None, "Should have predicate"
        assert isinstance(vp, VerbPhrase), "Predicate should be VerbPhrase" 
        assert vp.vector.isa("tobe"), "Predicate should include 'tobe'"      
        assert "box" in vp.noun_phrase.noun, "VP NP should be 'box'"
        assert len(vp.prepositions) == 1, "VP should have 1 PP"
        assert vp.prepositions[0].preposition == "on", "PP should be 'on'"
        assert vp.prepositions[0].noun_phrase is not None, "PP should have NP"
        assert "table" in vp.prepositions[0].noun_phrase.noun, "PP NP should be 'table'"

    def test_interrogative_sentence_negation(self):   
        """Test: 'is not a box on the table' -> SP token."""
        result = self.executor.execute_layer5("is not a box on the table", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have SP hypotheses"
        
        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        vp = sent.predicate
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is None, "Should have subject"
        assert vp is not None, "Should have predicate"
        assert isinstance(vp, VerbPhrase), "Predicate should be VerbPhrase" 
        assert vp.vector.isa("tobe"), "Predicate should include 'tobe'"      
        assert "box" in vp.noun_phrase.noun, "VP NP should be 'box'"
        assert len(vp.prepositions) == 1, "VP should have 1 PP"
        assert vp.prepositions[0].preposition == "on", "PP should be 'on'"
        assert vp.prepositions[0].noun_phrase is not None, "PP should have NP"
        assert "table" in vp.prepositions[0].noun_phrase.noun, "PP NP should be 'table'"

    def test_interrogative_sentence_negation2(self):   
        """Test: 'is a box not on the table' -> SP token."""
        result = self.executor.execute_layer5("is a box not on the table", report=True, tokenize_only=True)

        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should have SP hypotheses"
        
        hyp = result.hypotheses[0]
        sent = hyp.tokens[0].phrase
        vp = sent.predicate
        assert isinstance(sent, SentencePhrase), "Should be SentencePhrase object"
        assert sent.subject is None, "Should have subject"
        assert vp is not None, "Should have predicate"
        assert isinstance(vp, VerbPhrase), "Predicate should be VerbPhrase" 
        assert vp.vector.isa("tobe"), "Predicate should include 'tobe'"      
        assert "box" in vp.noun_phrase.noun, "VP NP should be 'box'"
        assert len(vp.prepositions) == 1, "VP should have 1 PP"
        assert vp.prepositions[0].preposition == "not on", "PP should be 'on'"
        assert vp.prepositions[0].noun_phrase is not None, "PP should have NP"
        assert "table" in vp.prepositions[0].noun_phrase.noun, "PP NP should be 'table'"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
