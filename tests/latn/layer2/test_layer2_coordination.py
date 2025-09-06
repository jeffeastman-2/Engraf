import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor

def test_adverb_adjective_in_coordinated_np():
    """Test Layer 2 NP tokenization with coordinated noun phrases containing adverb-adjective sequences"""
    executor = LATNLayerExecutor()
    
    # Test coordinated NPs: "a tall red box and a small bright blue circle"
    result = executor.execute_layer2('a tall red box and a small bright blue circle',report=True)
    
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

