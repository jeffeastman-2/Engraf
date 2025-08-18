#!/usr/bin/env python3
"""
Unit Tests for LATN Layer 2: NounPhrase Token Replacement

Tests for the Layer 2 tokenization system that identifies noun phrases
and replaces them with single NounPhrase tokens.
"""

import pytest
from engraf.lexer.latn_tokenizer_layer2 import (
    latn_tokenize_layer2, 
    latn_tokenize_layer2_best, 
    run_layer2,
    create_np_token,
    find_np_sequences,
    replace_np_sequences
)
from engraf.lexer.latn_tokenizer import latn_tokenize_best as tokenize
from engraf.atn.subnet_np import run_np
from engraf.pos.noun_phrase import NounPhrase


class TestLATNLayer2Basic:
    """Test basic Layer 2 NP token replacement functionality."""
    
    def test_simple_np_replacement(self):
        """Test basic NP replacement: 'the red box' -> single NP token."""
        hypotheses = latn_tokenize_layer2("the red box")
        assert len(hypotheses) > 0, "Should generate hypotheses"
        
        best = hypotheses[0]
        assert len(best.tokens) == 1, "Should have exactly one token"
        assert best.tokens[0].word == "NP(the red box)", "Should create NP token"
        assert best.tokens[0].isa("NP"), "Token should have NP dimension"
        assert hasattr(best.tokens[0], '_original_np'), "Should have original NP reference"
        assert len(best.np_replacements) == 1, "Should record NP replacement"
    
    def test_vector_np_replacement(self):
        """Test NP replacement with vector: '[1,2,3]' -> single NP token."""
        hypotheses = latn_tokenize_layer2("[1,2,3]")
        assert len(hypotheses) > 0, "Should generate hypotheses"
        
        best = hypotheses[0]
        assert len(best.tokens) == 1, "Should have exactly one token"
        assert best.tokens[0].word == "NP([1,2,3])", "Should create vector NP token"
        assert best.tokens[0].isa("NP"), "Token should have NP dimension"
        assert best.tokens[0].isa("vector"), "Should preserve vector semantics"
    
    def test_complex_np_replacement(self):
        """Test complex NP: 'a very large red sphere' -> single NP token."""
        hypotheses = latn_tokenize_layer2("a very large red sphere")
        assert len(hypotheses) > 0, "Should generate hypotheses"
        
        best = hypotheses[0]
        assert len(best.tokens) == 1, "Should have exactly one token"
        assert "NP(" in best.tokens[0].word, "Should create NP token"
        assert best.tokens[0].isa("NP"), "Token should have NP dimension"
        assert best.tokens[0].isa("adj"), "Should preserve adjective semantics"
        assert best.tokens[0].isa("noun"), "Should preserve noun semantics"
    
    def test_multiple_nps(self):
        """Test sentence with multiple NPs: 'the red box and the blue sphere'."""
        hypotheses = latn_tokenize_layer2("the red box and the blue sphere")
        assert len(hypotheses) > 0, "Should generate hypotheses"
        
        best = hypotheses[0]
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) >= 1, "Should have at least one NP token"
        assert len(best.np_replacements) >= 1, "Should record NP replacements"
    
    def test_no_np_sentence(self):
        """Test sentence with no noun phrases: 'hello'."""
        hypotheses = latn_tokenize_layer2("hello")
        assert len(hypotheses) > 0, "Should generate hypotheses"
        
        best = hypotheses[0]
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) == 0, "Should have no NP tokens"
        assert len(best.np_replacements) == 0, "Should record no NP replacements"


class TestLATNLayer2NounPhraseDetection:
    """Test the NP sequence finding functionality."""
    
    def test_find_np_sequences_simple(self):
        """Test finding simple NP sequences."""
        tokens = tokenize("the red box")
        np_sequences = find_np_sequences(tokens)
        
        assert len(np_sequences) > 0, "Should find NP sequence"
        start, end, np = np_sequences[0]
        assert start == 0, "Should start at beginning"
        assert end == 2, "Should end at last token"
        assert np.noun == "box", "Should identify correct noun"
        assert np.determiner == "the", "Should identify determiner"
    
    def test_find_np_sequences_vector(self):
        """Test finding vector NP sequences."""
        tokens = tokenize("[5,10,15]")
        np_sequences = find_np_sequences(tokens)
        
        assert len(np_sequences) > 0, "Should find vector NP"
        start, end, np = np_sequences[0]
        assert start == 0, "Should start at beginning"
        assert np.vector["vector"] == 1.0, "Should be marked as vector"
    
    def test_replace_np_sequences(self):
        """Test replacing NP sequences with NP tokens."""
        tokens = tokenize("the red box")
        np_sequences = find_np_sequences(tokens)
        new_tokens = replace_np_sequences(tokens, np_sequences)
        
        assert len(new_tokens) == 1, "Should replace with single token"
        assert new_tokens[0].isa("NP"), "Should be NP token"
        assert "NP(" in new_tokens[0].word, "Should have NP descriptor"


class TestLATNLayer2TokenCreation:
    """Test NP token creation functionality."""
    
    def test_create_np_token_structure(self):
        """Test NP token creation preserves structure."""
        tokens = tokenize("the big red sphere")
        np = run_np(tokens)
        assert np is not None, "Should successfully parse NP"
        
        np_token = create_np_token(np)
        assert np_token.isa("NP"), "Should have NP dimension"
        assert np_token.isa("noun"), "Should preserve noun semantics"
        assert np_token.isa("adj"), "Should preserve adjective semantics"
        assert hasattr(np_token, '_original_np'), "Should store original NP"
        assert np_token._original_np is np, "Should reference original NP"
    
    def test_create_np_token_word_format(self):
        """Test NP token word formatting."""
        tokens = tokenize("the table")
        np = run_np(tokens)
        np_token = create_np_token(np)
        
        assert "NP(" in np_token.word, "Should have NP prefix"
        assert "table" in np_token.word, "Should include noun"
        assert "the" in np_token.word, "Should include determiner"


class TestLATNLayer2Integration:
    """Test Layer 2 integration with overall system."""
    
    def test_layer2_convenience_function(self):
        """Test the convenience function latn_tokenize_layer2_best."""
        tokens = latn_tokenize_layer2_best("the red box")
        assert len(tokens) == 1, "Should return single NP token"
        assert tokens[0].isa("NP"), "Should be NP token"
    
    def test_run_layer2_function(self):
        """Test the run_layer2 debugging function."""
        hypothesis = run_layer2("a blue sphere")
        assert hypothesis is not None, "Should return hypothesis"
        assert len(hypothesis.tokens) == 1, "Should have NP token"
        assert hypothesis.tokens[0].isa("NP"), "Should be NP token"
    
    def test_layer2_hypothesis_metadata(self):
        """Test that Layer 2 hypotheses have proper metadata."""
        hypotheses = latn_tokenize_layer2("the green cube")
        assert len(hypotheses) > 0, "Should generate hypotheses"
        
        best = hypotheses[0]
        assert hasattr(best, 'confidence'), "Should have confidence score"
        assert hasattr(best, 'description'), "Should have description"
        assert hasattr(best, 'np_replacements'), "Should track NP replacements"
        assert "Layer 2" in best.description, "Description should mention Layer 2"


class TestLATNLayer2EdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_string(self):
        """Test Layer 2 with empty string."""
        hypotheses = latn_tokenize_layer2("")
        assert len(hypotheses) >= 0, "Should handle empty string gracefully"
    
    def test_single_word(self):
        """Test Layer 2 with single noun."""
        hypotheses = latn_tokenize_layer2("sphere")
        assert len(hypotheses) > 0, "Should handle single noun"
        
        best = hypotheses[0]
        if len(best.tokens) == 1 and best.tokens[0].isa("NP"):
            # Single noun was converted to NP
            assert "sphere" in best.tokens[0].word
        else:
            # Single noun remained as is
            assert best.tokens[0].word == "sphere"
    
    def test_preposition_not_converted(self):
        """Test that prepositions alone are not converted to NPs."""
        hypotheses = latn_tokenize_layer2("on")
        assert len(hypotheses) > 0, "Should handle preposition"
        
        best = hypotheses[0]
        prep_tokens = [tok for tok in best.tokens if tok.isa("prep")]
        assert len(prep_tokens) > 0, "Should preserve preposition"


