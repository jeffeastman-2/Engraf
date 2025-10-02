from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from tests.latn.dummy_test_scene import DummyTestScene

def test_boxes_grounding():
    """Test Layer 2 NP tokenization with plural noun phrases"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene2()) 
    
    # Test plural NPs: "the red boxes"
    result = executor.execute_layer2('the boxes',report=True)
    
    assert result.success, "Failed to tokenize plural NPs"
    assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

    main_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    token = main_hyp.tokens[0]
    assert token.isa("NP"), "First token should be a noun phrase NP"
    np = token.phrase
    assert token.isa("plural"), "NP should be marked as plural"
    assert hasattr(np, 'grounding'), "NP token should have grounding info"
    grounding_info = np.grounding.get('scene_objects')
    assert len(grounding_info) == 5, f"NP grounding should reference 5 objects, got {len(grounding_info)}"

