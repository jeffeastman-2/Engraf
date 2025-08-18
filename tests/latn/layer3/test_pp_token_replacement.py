#!/usr/bin/env python3
"""
LATN Layer 3: Prepositional    def test_pp_with_simple_np(self):
        """Test PP replacement with simple NP: 'on the table'."""
        hypothesis = run_layer3("on the table")
        assert hypothesis is not None, "Layer 3 should process PP with simple NP"
        
        tokens = [tok.word for tok in hypothesis.tokens]
        print(f"Simple NP PP tokens: {tokens}")
        
        # Should have PP replacement
        assert any("PP(" in token for token in tokens), "Should have PP token"
        assert len(hypothesis.pp_replacements) > 0, "Should have PP replacements"
        
        # Check for PP tokens
        pp_tokens = [tok for tok in hypothesis.tokens if tok.isa("prep_phrase")]
        assert len(pp_tokens) > 0, "Should have PrepositionalPhrase tokens"ts

Tests for the Layered Augmented Transition Network Layer 3 implementation.
Layer 3 replaces prepositional phrase constructions with single PrepositionalPhrase tokens.

The beauty of Layer 3 is that the PP ATN becomes incredibly simple:
- PREP -> NounPhrase token -> END
- No more complex NP subnetwork parsing!
"""

import pytest
from engraf.lexer.latn_tokenizer_layer3 import latn_tokenize_layer3, latn_tokenize_layer3_best, create_pp_token
from engraf.lexer.vector_space import VectorSpace, vector_from_features


def run_layer3(text):
    """Process text through all LATN layers up to Layer 3."""
    hypotheses = latn_tokenize_layer3(text)
    if not hypotheses:
        return None
    return hypotheses[0]  # Return best hypothesis


def create_mock_np_token(np_object):
    """Create a mock NounPhrase token for testing."""
    token = VectorSpace()
    token["noun_phrase"] = 1.0
    token.word = f"NP({np_object.get_original_text() if hasattr(np_object, 'get_original_text') else str(np_object)})"
    token._original_np = np_object
    return token


class TestLATNLayer3BasicPP:
    """Test basic PP token replacement functionality."""
    
    def test_simple_pp_replacement(self):
        """Test basic PP replacement: 'at [1,2,3]' -> single PP token."""
        # Use Layer 3 processing pipeline
        hypothesis = run_layer3("at [1,2,3]")
        assert hypothesis is not None, "Layer 3 should process simple PP"
        
        tokens = [tok.word for tok in hypothesis.tokens]
        print(f"Layer 3 tokens: {tokens}")
        
        # Should have PP replacement
        assert any("PP(" in token for token in tokens), "Should have PP token"
        assert len(hypothesis.pp_replacements) > 0, "Should have PP replacements"
        
        # Check for PP tokens
        pp_tokens = [tok for tok in hypothesis.tokens if tok.isa("prep_phrase")]
        assert len(pp_tokens) > 0, "Should have PrepositionalPhrase tokens"
    
    def test_pp_with_simple_np(self):
        """Test PP replacement with simple NP: 'on the table'."""
        tokens = tokenize("on the table")
        pp = run_pp(tokens)
        
        assert pp is not None
        assert pp.preposition == "on"
        assert pp.noun_phrase is not None
        assert pp.noun_phrase.noun == "table"
        assert pp.noun_phrase.determiner == "the"
        
        # Test PP token creation
        pp_token = create_pp_token(pp)
        assert pp_token["prep_phrase"] == 1.0
        assert "on" in pp_token.word
        assert pp_token._original_pp == pp
    
    def test_pp_with_complex_np(self):
        """Test PP replacement with complex NP: 'above the big red sphere'."""
        tokens = tokenize("above the big red sphere")
        pp = run_pp(tokens)
        
        assert pp is not None
        assert pp.preposition == "above"
        assert pp.noun_phrase is not None
        assert pp.noun_phrase.noun == "sphere"
        assert pp.noun_phrase.vector["red"] > 0
        assert pp.noun_phrase.vector["scaleX"] > 1  # big
        
        # Test PP token creation  
        pp_token = create_pp_token(pp)
        assert pp_token["prep_phrase"] == 1.0
        assert "above" in pp_token.word


class TestLATNLayer3Integration:
    """Test Layer 3 integration with the full LATN system."""
    
    def test_layer3_tokenization_simple_pp(self):
        """Test Layer 3 tokenization of simple prepositional phrases."""
        sentence = "at [1,2,3]"
        
        hypotheses = latn_tokenize_layer3(sentence)
        assert len(hypotheses) > 0
        
        # Check the best hypothesis
        best_hyp = hypotheses[0]
        tokens = [tok.word for tok in best_hyp.tokens]
        
        print(f"Layer 3 tokens: {tokens}")
        
        # Should have PP replacement
        assert any("PP(" in token for token in tokens), "Should have PP token"
        assert len(best_hyp.pp_replacements) > 0, "Should have PP replacements"
        
        # Test the convenience function
        best_tokens = latn_tokenize_layer3_best(sentence)
        pp_tokens = [tok for tok in best_tokens if tok.isa("prep_phrase")]
        assert len(pp_tokens) > 0, "Should have PP tokens"
    
    def test_layer3_complex_pp(self):
        """Test Layer 3 with complex prepositional phrase."""
        sentence = "above the very large red sphere"
        
        hypotheses = latn_tokenize_layer3(sentence)
        assert len(hypotheses) > 0
        
        best_hyp = hypotheses[0]
        tokens = [tok.word for tok in best_hyp.tokens]
        
        print(f"Complex PP tokens: {tokens}")
        
        # Should have PP replacement
        pp_count = sum(1 for token in tokens if "PP(" in token)
        assert pp_count >= 1, f"Should have at least 1 PP token, got {pp_count}"
        assert len(best_hyp.pp_replacements) >= 1, "Should have PP replacements"


class TestLATNLayer3Ambiguity:
    """Test Layer 3 with NP ambiguity that creates PP ambiguity."""
    
    def test_pp_ambiguity_from_np_ambiguity(self):
        """Test how NP ambiguity affects PP parsing: 'near the red box'."""
        # This tests the scenario where NP ambiguity propagates to PP ambiguity
        # If there are multiple ways to parse "the red box", there are multiple ways to parse "near the red box"
        
        sentence = "near the red box"
        
        hypotheses = latn_tokenize_layer3(sentence)
        assert len(hypotheses) > 0
        
        print(f"\nPP ambiguity test: '{sentence}'")
        for i, hyp in enumerate(hypotheses[:3], 1):  # Show first 3
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.3f}): {tokens}")
            print(f"    PP replacements: {len(hyp.pp_replacements)}")
        
        # Check that we get PP tokens in hypotheses
        best_hyp = hypotheses[0]
        pp_tokens = [tok for tok in best_hyp.tokens if tok.isa("prep_phrase")]
        assert len(pp_tokens) > 0, "Should have PP tokens"
    
    def test_compound_ambiguity_in_pp(self):
        """Test compound word ambiguity affecting PP: 'above the light house'."""
        from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
        
        # Set up compound ambiguity
        original_entries = {}
        test_entries = {
            'light': vector_from_features('adj', transparency=0.8),
            'house': vector_from_features('noun'),
            'light house': vector_from_features('noun')  # compound
        }
        
        for word, vector in test_entries.items():
            original_entries[word] = word in SEMANTIC_VECTOR_SPACE
            if not original_entries[word]:
                SEMANTIC_VECTOR_SPACE[word] = vector
        
        try:
            sentence = "above the light house"
            
            hypotheses = latn_tokenize_layer3(sentence)
            assert len(hypotheses) > 0
            
            print(f"\nCompound ambiguity in PP: '{sentence}'")
            for i, hyp in enumerate(hypotheses[:3], 1):
                tokens = [tok.word for tok in hyp.tokens]
                print(f"  Hypothesis {i} (conf={hyp.confidence:.3f}): {tokens}")
                print(f"    PP replacements: {len(hyp.pp_replacements)}")
            
            # Should have different interpretations
            # Some with "light house" (compound) and some with "light" + "house"
            # This creates different PP structures
            assert len(hypotheses) >= 2, "Should have multiple hypotheses for compound ambiguity"
            
        finally:
            # Clean up
            for word, was_original in original_entries.items():
                if not was_original and word in SEMANTIC_VECTOR_SPACE:
                    del SEMANTIC_VECTOR_SPACE[word]
    
    def test_complex_ambiguous_pp_chain(self):
        """Test multiple ambiguous PPs: 'from the light house to the green sphere'."""
        sentence = "from the light house to the green sphere"
        
        hypotheses = latn_tokenize_layer3(sentence)
        assert len(hypotheses) > 0
        
        print(f"\nComplex PP chain: '{sentence}'")
        for i, hyp in enumerate(hypotheses[:3], 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.3f}): {tokens}")
            print(f"    PP replacements: {len(hyp.pp_replacements)}")
            
            # Count PP tokens
            pp_count = sum(1 for tok in hyp.tokens if tok.isa("prep_phrase"))
            print(f"    PP tokens: {pp_count}")
        
        # Should have multiple PPs in the best hypothesis
        best_hyp = hypotheses[0]
        pp_tokens = [tok for tok in best_hyp.tokens if tok.isa("prep_phrase")]
        assert len(pp_tokens) >= 2, f"Should have at least 2 PP tokens, got {len(pp_tokens)}"


class TestLATNLayer3PPTokenStructure:
    """Test the structure and properties of PP tokens."""
    
    def test_pp_token_semantic_content(self):
        """Test that PP tokens preserve semantic content."""
        tokens = tokenize("at [5,10,15]")
        pp = run_pp(tokens)
        pp_token = create_pp_token(pp)
        
        # PP token should have the location information
        assert pp_token["locX"] == 5.0
        assert pp_token["locY"] == 10.0
        assert pp_token["locZ"] == 15.0
        assert pp_token["prep_phrase"] == 1.0
        
        # Should preserve the original PP object
        assert hasattr(pp_token, '_original_pp')
        assert pp_token._original_pp == pp
        assert pp_token._original_pp.preposition == "at"
    
    def test_pp_token_with_adjective_np(self):
        """Test PP token with adjectival NP content."""
        tokens = tokenize("under the very large blue box")
        pp = run_pp(tokens)
        pp_token = create_pp_token(pp)
        
        # Should have the adjective content from the NP
        assert pp_token["blue"] > 0
        assert pp_token["scaleX"] > 2  # very large
        assert pp_token["prep_phrase"] == 1.0
        
        # Original PP should be preserved
        original_pp = pp_token._original_pp
        assert original_pp.preposition == "under"
        assert original_pp.noun_phrase.noun == "box"
    
    def test_pp_token_descriptive_word(self):
        """Test that PP tokens have descriptive word representations."""
        test_cases = [
            ("at [1,2,3]", "PP(at"),
            ("on the table", "PP(on"),
            ("above the red sphere", "PP(above")
        ]
        
        for sentence, expected_start in test_cases:
            tokens = tokenize(sentence)
            pp = run_pp(tokens)
            pp_token = create_pp_token(pp)
            
            assert pp_token.word.startswith(expected_start), f"Expected '{pp_token.word}' to start with '{expected_start}'"
            assert "PP(" in pp_token.word, "Should have PP( prefix"

