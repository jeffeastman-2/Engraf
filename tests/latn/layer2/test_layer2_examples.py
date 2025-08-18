#!/usr/bin/env python3
"""
LATN Layer 2 Integration Tests

Simple integration tests that verify Layer 2 works with the examples
that were previously in the __main__ section.
"""

import pytest
from engraf.lexer.latn_tokenizer_layer2 import latn_tokenize_layer2


class TestLATNLayer2Examples:
    """Test Layer 2 with the original test examples."""
    
    def test_the_red_box(self):
        """Test: 'the red box' -> NP token."""
        hypotheses = latn_tokenize_layer2("the red box")
        assert len(hypotheses) > 0
        
        best = hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        print(f"NP replacements: {len(best.np_replacements)}")
        
        assert len(best.np_replacements) > 0, "Should have NP replacements"
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) > 0, "Should have NP tokens"
    
    def test_very_large_sphere(self):
        """Test: 'a very large sphere' -> NP token."""
        hypotheses = latn_tokenize_layer2("a very large sphere")
        assert len(hypotheses) > 0
        
        best = hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) > 0, "Should have NP tokens"
    
    def test_vector_coordinate(self):
        """Test: '[1,2,3]' -> NP token."""
        hypotheses = latn_tokenize_layer2("[1,2,3]")
        assert len(hypotheses) > 0
        
        best = hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) > 0, "Should have NP tokens"
        assert best.tokens[0].isa("vector"), "Should preserve vector semantics"
    
    def test_big_red_sphere(self):
        """Test: 'the big red sphere' -> NP token."""
        hypotheses = latn_tokenize_layer2("the big red sphere")
        assert len(hypotheses) > 0
        
        best = hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) > 0, "Should have NP tokens"
    
    def test_complex_with_pp(self):
        """Test: 'a small green box on the table' -> multiple tokens."""
        hypotheses = latn_tokenize_layer2("a small green box on the table")
        assert len(hypotheses) > 0
        
        best = hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        print(f"NP replacements: {len(best.np_replacements)}")
        
        # Should have at least one NP token (and possibly a preposition + another NP)
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) >= 1, "Should have at least one NP token"
