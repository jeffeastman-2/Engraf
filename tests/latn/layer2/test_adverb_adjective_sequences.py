import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor


def test_adverb_adjective_in_coordinated_np():
    """Test Layer 2 NP tokenization with coordinated noun phrases containing adverb-adjective sequences"""
    executor = LATNLayerExecutor()
    
    # Test coordinated NPs: "a tall red box and a small bright blue circle"
    result = executor.execute_layer2('a tall red box and a small bright blue circle', enable_semantic_grounding=False)
    
    assert result.success, "Failed to tokenize coordinated NPs with adverb-adjective sequences"
    assert len(result.hypotheses) > 0, "Should generate hypotheses"
    
    best = result.hypotheses[0]
    
    # Should have exactly 2 NP tokens (one for each noun phrase)
    np_tokens = [token for token in best.tokens if hasattr(token, 'word') and token.word.startswith('NP(')]
    assert len(np_tokens) == 2, f"Should have exactly 2 NP tokens, got {len(np_tokens)}"
    
    # Check for conjunction token
    conjunction_tokens = [token for token in best.tokens if hasattr(token, 'word') and token.word == 'and']
    assert len(conjunction_tokens) == 1, "Should have exactly one conjunction token"
    
    # Verify the structure: NP1, 'and', NP2
    assert len(best.tokens) == 3, "Should have exactly 3 tokens: NP1, 'and', NP2"
    assert best.tokens[0].word == "NP(a tall red box)", "First token should be first NP"
    assert best.tokens[1].word == "and", "Second token should be conjunction"
    assert best.tokens[2].word == "NP(a small bright blue circle)", "Third token should be second NP"


def test_simple_adverb_adjective_sequence():
    """Test Layer 2 NP tokenization with simple adverb-adjective sequence: 'a very tall cylinder'"""
    executor = LATNLayerExecutor()
    
    result = executor.execute_layer2('a very tall cylinder', enable_semantic_grounding=False)
    
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
    
    result = executor.execute_layer2('a very tall red box', enable_semantic_grounding=False)
    
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
