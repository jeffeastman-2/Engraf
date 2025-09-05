#!/usr/bin/env python3
"""
LATN Layer 3 Integration Tests

Integration tests that verify Layer 3 works with complex examples,
including those that were previously in the __main__ section.
"""

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase


class TestLATNLayer3ComplexExamples:
    """Test Layer 3 with complex prepositional phrase examples."""
    
    def test_at_vector_coordinates(self):
        """Test: 'at [1,2,3]' -> single PP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("at [1,2,3]",)
        assert result.success, "Layer 3 should succeed"
        
        print(f"Generated {len(result.hypotheses)} Layer 3 hypotheses")
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP replacements: {len(best.replacements)}")
        
        # Should have exactly one PP token for "at [1,2,3]"
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) == 1, "Should have exactly one PP token"
        assert len(best.tokens) == 1, "Should be a single token"
    
    def test_above_complex_np(self):
        """Test: 'above the table' -> single PP token with NP inside."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("above the table")
        assert result.success, "Layer 3 should succeed"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP replacements: {len(best.replacements)}")
        
        # Should have PP token for "above the table"
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) == 1, "Should have exactly one PP token"
        assert len(best.tokens) == 1, "Should be a single token"
    
    def test_on_complex_np(self):
        """Test: 'on the red sphere' -> single PP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("on the red sphere")
        assert result.success, "Layer 3 should succeed"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        
        # Should have PP token for "on the red sphere"
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) == 1, "Should have exactly one PP token"
        assert len(best.tokens) == 1, "Should be a single token"
    
    def test_multiple_prep_phrases(self):
        """Test: 'from the table to the sphere' -> two PP tokens."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("from the table to the sphere")
        assert result.success, "Layer 3 should succeed"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP replacements: {len(best.replacements)}")
        
        # Should have multiple PP tokens
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) >= 1, "Should have at least one PP token"


class TestLATNLayer3SimpleExamples:
    """Test Layer 3 with simple PP-only examples (Layer 3 scope)."""
    
    def test_at_vector_only(self):
        """Test: 'at [1,2,3]' -> single PP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("at [1,2,3]")
        assert result.success, "Layer 3 should succeed"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        
        assert len(best.tokens) == 1, "Should have single token"
        assert best.tokens[0].isa("PP"), "Should be PP token"
        assert "at" in best.tokens[0].word, "Should include preposition"
    
    def test_on_the_table_only(self):
        """Test: 'on the table' -> single PP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("on the table")
        assert result.success, "Layer 3 should succeed"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) > 0, "Should have PP tokens"
    
    def test_above_red_sphere_only(self):
        """Test: 'above the red sphere' -> single PP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("above the red sphere")
        assert result.success, "Layer 3 should succeed"
        
        best = result.hypotheses[0]
        print(f"Tokens: {[tok.word for tok in best.tokens]}")
        
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) > 0, "Should have PP tokens"


class TestLATNLayer3LayerProgression:
    """Test the layer progression: Layer 1 -> Layer 2 -> Layer 3."""
    
    def test_layer_progression_demonstration(self):
        """Demonstrate the beautiful layer progression."""
        sentence = "at [1,2,3]"
        
        print(f"\n🔄 LATN Layer Progression for: '{sentence}'")
        print("=" * 60)
        
        # Use the LayerExecutor for proper pipeline testing
        executor = LATNLayerExecutor()
        
        # Layer 1: Multi-hypothesis tokenization
        layer1_result = executor.execute_layer1(sentence)
        print(f"Layer 1: {len(layer1_result.hypotheses)} tokenization hypotheses")
        print(f"  Best: {[t.word for t in layer1_result.hypotheses[0].tokens]}")
        
        # Layer 2: NP token replacement
        layer2_result = executor.execute_layer2(sentence)
        print(f"Layer 2: {len(layer2_result.hypotheses)} NP-enhanced hypotheses")
        print(f"  Best: {[t.word for t in layer2_result.hypotheses[0].tokens]}")
        
        # Layer 3: PP token replacement  
        layer3_result = executor.execute_layer3(sentence)
        print(f"Layer 3: {len(layer3_result.hypotheses)} PP-enhanced hypotheses")
        print(f"  Best: {[t.word for t in layer3_result.hypotheses[0].tokens]}")
        
        # Verify progression
        assert layer1_result.success, "Layer 1 should work"
        assert layer2_result.success, "Layer 2 should work"
        assert layer3_result.success, "Layer 3 should work"
        
        # Layer 3 should have the most compact representation
        layer3_best = layer3_result.hypotheses[0]
        assert len(layer3_best.tokens) == 1, "Layer 3 should create single PP token"
        assert layer3_best.tokens[0].isa("PP"), "Should be PP token"

def test_coordinated_pp():
    """Test Layer 3 PP tokenization with coordinated prepositional phrases """
    executor = LATNLayerExecutor()

    # Test coordinated PPs: "above the red box and below the blue circle and behind the octahedron"
    result = executor.execute_layer3('above the red box and below the blue circle and behind the octahedron')

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    main_hyp = result.hypotheses[1]
    # Should have exactly 3 PP tokens (one for each prepositional phrase)
    assert len(main_hyp.tokens) == 5, f"Should have exactly 5 tokens, got {len(main_hyp.tokens)}"
    main_pp = main_hyp.tokens[0].word
    assert main_pp.startswith("PP("), "First token should be a PP"
    conj_pp = main_hyp.tokens[1].word
    assert conj_pp.startswith("and"), "Second token should be a conjunction"
    other_pp = main_hyp.tokens[2].word
    assert other_pp.startswith("PP("), "Third token should be a PP"
    conj_pp = main_hyp.tokens[3].word
    assert conj_pp.startswith("and"), "Fourth token should be a conjunction"
    other_pp = main_hyp.tokens[4].word
    assert other_pp.startswith("PP("), "Fifth token should be a PP"

    conj_hyp = result.hypotheses[0]    
    # Should have exactly 2 NP tokens (one for each noun phrase)
    assert len(conj_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(conj_hyp.tokens)}"
    conj_np = conj_hyp.tokens[0].word
    assert conj_np.startswith("CONJ-PP"), "First token should be a conjunction PP"
    conj = conj_hyp.tokens[0]
    assert conj is not None
    original_pp = conj._original_pp
    assert isinstance(original_pp, ConjunctionPhrase), f"Expected ConjunctionPhrase, got {type(original_pp)}"
    assert isinstance(original_pp.right, PrepositionalPhrase), f"Expected PrepositionalPhrase, got {type(original_pp.right)}"
    assert isinstance(original_pp.left, ConjunctionPhrase), f"Expected ConjunctionPhrase, got {type(original_pp.left)}"
    assert isinstance(original_pp.left.left, PrepositionalPhrase), f"Expected PrepositionalPhrase, got {type(original_pp.left.left)}"
    assert isinstance(original_pp.left.right, PrepositionalPhrase), f"Expected PrepositionalPhrase, got {type(original_pp.left.right)}"

def test_coordinated_pp_with_nps():

    executor = LATNLayerExecutor()

    # Test coordinated PPs: "color the red cube above the table and the blue sphere below the cylinder green"
    # Hypothesis-1: "color the red cube (above the table) and (the blue sphere) below the cylinder green"
    # Hypothesis-2: "color the red cube (above the table and the blue sphere) (below the cylinder) green"
    result = executor.execute_layer3('color the red cube above the table and the blue sphere below the cylinder green',report=True)

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 5, f"Should have exactly 5 tokens, got {len(hyp.tokens)}"
    hyp = result.hypotheses[1]
    assert len(hyp.tokens) == 7, f"Should have exactly 2 tokens, got {len(hyp.tokens)}"

