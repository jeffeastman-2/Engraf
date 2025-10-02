import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from tests.latn.dummy_test_scene import DummyTestScene

def test_adverb_adjective_in_coordinated_np():
    """Test Layer 2 NP tokenization with coordinated noun phrases containing adverb-adjective sequences"""
    executor = LATNLayerExecutor()
    
    # Test coordinated NPs: "a tall red box and a small bright blue circle"
    result = executor.execute_layer2('a tall red box and a small bright blue circle',report=True)
    
    assert result.success, "Failed to tokenize coordinated NPs with adverb-adjective sequences"
    assert len(result.hypotheses) == 1, "Should generate 1 hypothesis0"

    conj_hyp = result.hypotheses[0]    
    # Should have exactly 1 CONJ-NP token
    assert len(conj_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(conj_hyp.tokens)}"
    conj_np = conj_hyp.tokens[0].word
    assert conj_np.startswith("CONJ-NP"), "First token should be a conjunction NP"

def test_coordinated_np_and_vp():
    """Test Layer 2 NP tokenization with coordinated noun phrases containing VPs"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene1()) 
    
    # Test coordinated NPs: "delete the sphere and the table and move the box to [3,3,3]"
    result = executor.execute_layer2('delete the sphere and the table and move the box to [3,3,3]',report=True)

    assert result.success, "Failed to tokenize coordinated NPs with VPs"
    assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

    main_hyp = result.hypotheses[0]
    # Should have exactly 2 NP tokens (one for each noun phrase)
    assert len(main_hyp.tokens) == 7, f"Should have exactly 7 tokens, got {len(main_hyp.tokens)}"
    main_np = main_hyp.tokens[0].word
    assert main_np.startswith("delete"), "First token should be a 'delete'"
    conj_np = main_hyp.tokens[1].word
    assert conj_np.startswith("CONJ-NP"), "Second token should be a NP conjunction"
    other_vp = main_hyp.tokens[2].word
    assert other_vp.startswith("and"), "Third token should be 'and'"
