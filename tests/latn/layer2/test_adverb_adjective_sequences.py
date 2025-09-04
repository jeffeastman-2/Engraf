import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor


def test_adverb_adjective_in_coordinated_np():
    """Test Layer 2 NP tokenization with coordinated noun phrases containing adverb-adjective sequences"""
    executor = LATNLayerExecutor()
    
    # Test coordinated NPs: "a tall red box and a small bright blue circle"
    result = executor.execute_layer2('a tall red box and a small bright blue circle')
    
    assert result.success, "Failed to tokenize coordinated NPs with adverb-adjective sequences"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    main_hyp = result.hypotheses[1]
    # Should have exactly 2 NP tokens (one for each noun phrase)
    assert len(main_hyp.tokens) == 3, f"Should have exactly 3 tokens, got {len(main_hyp.tokens)}"
    main_np = main_hyp.tokens[0].word
    assert main_np.startswith("NP("), "First token should be an NP"
    conj_np = main_hyp.tokens[1].word
    assert conj_np.startswith("and"), "Second token should be a conjunction"
    other_np = main_hyp.tokens[2].word
    assert other_np.startswith("NP("), "Third token should be an NP"

    conj_hyp = result.hypotheses[0]    
    # Should have exactly 2 NP tokens (one for each noun phrase)
    assert len(conj_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(conj_hyp.tokens)}"
    conj_np = conj_hyp.tokens[0].word
    assert conj_np.startswith("CONJ-NP"), "First token should be a conjunction NP"

def test_simple_adverb_adjective_sequence():
    """Test Layer 2 NP tokenization with simple adverb-adjective sequence: 'a very tall cylinder'"""
    executor = LATNLayerExecutor()
    
    result = executor.execute_layer2('a very tall cylinder')
    
    assert result.success, "Failed to tokenize simple adverb-adjective sequence"
    assert len(result.hypotheses) > 0, "Should generate hypotheses"
    
    best = result.hypotheses[0]
    
    # Should have NP token in the result
    np_tokens = [token for token in best.tokens if hasattr(token, 'word') and token.word.startswith('NP(')]
    assert len(np_tokens) >= 1, "Should have at least 1 NP token"
    
    # Verify that adverb-adjective sequence is properly handled
    tokens_str = ' '.join([token.word for token in best.tokens])
    assert 'very' in tokens_str or 'NP(' in tokens_str, "Should handle 'very' adverb"
    assert 'tall' in tokens_str or 'NP(' in tokens_str, "Should handle 'tall' adjective"
    assert 'cylinder' in tokens_str or 'NP(' in tokens_str, "Should handle 'cylinder' noun"


def test_multiple_adverb_adjective_sequences():
    """Test Layer 2 NP tokenization with multiple adjectives: 'a very tall red box'"""
    executor = LATNLayerExecutor()
    
    result = executor.execute_layer2('a very tall red box')
    
    assert result.success, "Failed to tokenize multiple adverb-adjective sequences"
    assert len(result.hypotheses) > 0, "Should generate hypotheses"
    
    best = result.hypotheses[0]
    
    # Should have NP token in the result
    np_tokens = [token for token in best.tokens if hasattr(token, 'word') and token.word.startswith('NP(')]
    assert len(np_tokens) >= 1, "Should have at least 1 NP token"
    
    # Verify that multiple adjectives are properly handled
    tokens_str = ' '.join([token.word for token in best.tokens])
    assert 'very' in tokens_str or 'NP(' in tokens_str, "Should handle 'very' adverb"
    assert 'tall' in tokens_str or 'NP(' in tokens_str, "Should handle 'tall' adjective"
    assert 'red' in tokens_str or 'NP(' in tokens_str, "Should handle 'red' adjective"
    assert 'box' in tokens_str or 'NP(' in tokens_str, "Should handle 'box' noun"
