import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from tests.latn.dummy_test_scene import DummyTestScene

class TestLATNLayer4VPCoordination:

    scene = DummyTestScene().get_scene1()

    """Set up a dummy scene for spatial validation tests.
    
    Scene contains:
    - box (above table)
    - table (reference object)
    - pyramid (left of table)
    - sphere (above pyramid)    
        """


    def test_create_vp(self):
        """Test Layer 4 VP tokenization with a simple 'create' verb phrase."""
        executor = LATNLayerExecutor(self.scene) # force grounding

        # Test simple VP: "draw a red box"
        result = executor.execute_layer4('draw a red box',report=True)

        assert result.success, "Failed to tokenize simple VP in Layer 4"
        assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

        hyp = result.hypotheses[0]    
        # Should have exactly 1 VP token
        assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
        vp = hyp.tokens[0]
        assert vp.word.startswith("VP"), "First token should be a VP"
        assert "draw" in vp.word, "VP should contain the verb 'draw'"
        assert "box" in vp.phrase.noun_phrase.noun, "VP should contain the noun 'box'"

    def test_create_with_pp(self):
        """Test Layer 4 VP tokenization with a 'create' verb phrase containing a PP."""
        executor = LATNLayerExecutor(self.scene) # force grounding  
        # Test VP with PP: "draw a red cube above the table"
        result = executor.execute_layer4('draw a red cube above the table',report=True)
        assert result.success, "Failed to tokenize VP with PP in Layer 4"
        assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

    def test_copy_vp(self):
        """Test Layer 4 VP tokenization with a simple 'create' verb phrase."""
        executor = LATNLayerExecutor(self.scene) # force grounding

        # Test simple VP: "copy the box"
        result = executor.execute_layer4('copy the box',report=True)

        assert result.success, "Failed to tokenize simple VP in Layer 4"
        assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

        hyp = result.hypotheses[0]    
        # Should have exactly 1 VP token
        assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
        vp = hyp.tokens[0]
        assert vp.word.startswith("VP"), "First token should be a VP"
        assert "copy" in vp.word, "VP should contain the verb 'copy'"
        assert "box" in vp.phrase.noun_phrase.noun, "VP should contain the noun 'box'"

    def test_copy_vp_ungrounded_np(self):
        """Test Layer 4 VP tokenization with a simple 'create' verb phrase."""
        executor = LATNLayerExecutor(self.scene) # force grounding

        # Test simple VP: "copy a red cube"
        result = executor.execute_layer4('copy a red cube',report=True)

        assert result.success, "Failed to tokenize simple VP in Layer 4"
        assert len(result.hypotheses) == 0, "Should generate 0 hypotheses"
        # No valid hypotheses because "copy" requires a grounded NP

    def test_copy_with_pp(self):
        """Test Layer 4 VP tokenization with a 'copy' verb phrase containing a PP."""
        executor = LATNLayerExecutor(self.scene) # force grounding  
        # Test VP with PP: "copy the box above the table"
        result = executor.execute_layer4('copy the box above the table',report=True)
        assert result.success, "Failed to tokenize VP with PP in Layer 4"
        assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"
    
    def test_copy_with_pp2(self):
        """Test Layer 4 VP tokenization with a 'copy' verb phrase containing a PP."""
        executor = LATNLayerExecutor(self.scene) # force grounding  
        # Test VP with PP: "copy the box below the table"
        result = executor.execute_layer4('copy the box below the table',report=True)
        assert result.success, "Failed to tokenize VP with PP in Layer 4"
        assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"        