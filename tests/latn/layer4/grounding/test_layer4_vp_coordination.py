import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from tests.latn.dummy_test_scene import DummyTestScene



def test_coordinated_vp():
    """Test Layer 4 VP tokenization with coordinated verb phrases """
    executor = LATNLayerExecutor(SceneModel()) # force grounding L2&3

    # Test coordinated VPs: "draw a red box and move the cube to [3,3,3]"
    result = executor.execute_layer4('draw a red box and draw a blue cube',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

    hyp0 = result.hypotheses[0]    
    # Should have exactly 1 VP token
    assert len(hyp0.tokens) == 1, f"Should have exactly 1 token, got {len(hyp0.tokens)}"
    conj_vp = hyp0.tokens[0].word
    assert conj_vp.startswith("CONJ-VP"), "First token should be a conjunction VP"
    conj = hyp0.tokens[0]
    assert conj is not None

def test_invalid_coordinated_vp():
    """Test Layer 4 VP tokenization with coordinated verb phrases, one invalid """
    executor = LATNLayerExecutor(SceneModel()) # force grounding 

    # Test coordinated VPs: "draw a red box and delete a blue cube"
    result = executor.execute_layer4('draw a red box and delete a blue cube',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) == 0 , "Should generate 0 hypotheses"

def test_invalid_coordinated_vp2():
    """Test Layer 4 VP tokenization with coordinated verb phrases, one invalid """
    executor = LATNLayerExecutor(SceneModel()) # force grounding 

    # Test coordinated VPs: "draw the red box and delete a blue cube"
    result = executor.execute_layer4('draw the red box and delete a blue cube',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) == 0 , "Should generate 0 hypotheses"

def test_coordinated_vp_with_pps():
    """Set up a dummy scene for spatial validation tests.
    
    Scene contains:
    - box (above table)
    - table (reference object)
    - pyramid (left of table)
    - sphere (above pyramid)    
    """
    executor = LATNLayerExecutor(DummyTestScene().scene) # force grounding 

    # Test coordinated VPs: "draw a red cube above the table and color the sphere above the pyramid green'"
    result = executor.execute_layer4('draw a red cube above the table and color the sphere above the pyramid green',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"

def test_coordinated_vp_with_coordinated_pps():
    """Set up a dummy scene for spatial validation tests.
    
    Scene contains:
    - box (above table)
    - table (reference object)
    - pyramid (left of table)
    - sphere (above pyramid)    
    """
    #assert False
    executor = LATNLayerExecutor(DummyTestScene().scene) # force grounding
    # Test coordinated VPs: "draw a red cube above the table and the sphere and color the sphere above the pyramid green'"
    result = executor.execute_layer4('draw a red cube above the table and color the sphere above the pyramid green',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) > 0, "Should generate 2 hypotheses"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
    assert hyp.tokens[0].word.startswith("CONJ-VP"), "First token should be a CONJ-VP"

def test_coordinated_vp_with_coordinated_pps_and_nps():
    """Set up a dummy scene for spatial validation tests.
    
    Scene contains:
    - box (above table)
    - table (reference object)
    - pyramid (left of table)
    - sphere (above pyramid)    
    """
    #assert False
    executor = LATNLayerExecutor(DummyTestScene().scene) # force grounding
    # Test coordinated VPs: "draw a red cube and a green cube above the table and the sphere and color the sphere above the pyramid green'"
    result = executor.execute_layer4('draw a red cube and a green cube above the table and the sphere and color the sphere above the pyramid green',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) > 0, "Should generate 6 hypotheses"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
    assert hyp.tokens[0].word.startswith("CONJ-VP"), "First token should be a CONJ-VP"

def test_coordinated_vp_with_coordinated_pps_and_location():                    
    #assert False
    executor = LATNLayerExecutor(DummyTestScene().scene) # force grounding

    # Test coordinated VPs: "draw a very tall box at [0,1,0] and rotate the table by 90 degrees"
    result = executor.execute_layer4('draw a very tall box at [3,1,0] and rotate the table by 90 degrees',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
#    assert False, "Forcing failure to test error handling"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
