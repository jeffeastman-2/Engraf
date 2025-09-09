import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel

class TestLayer5Integration:
    """Test Layer 5 integration with other layers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.scene = SceneModel()
        self.executor = LATNLayerExecutor(self.scene)

    def test_layer5_with_layer2_grounding(self):
        """Test Layer 5 sentence phrase extraction with Layer 2 grounding of existing objects."""
        # Add some objects to the scene first (not through Layer 5)
        from engraf.lexer.vector_space import vector_from_features
        from engraf.visualizer.scene.scene_object import SceneObject
        
        red_box_vector = vector_from_features("noun", red=1.0)
        blue_sphere_vector = vector_from_features("noun", blue=1.0) 
        
        red_box = SceneObject("box", red_box_vector, object_id="red_box_1")
        blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere_1")
        
        self.scene.add_object(red_box)
        self.scene.add_object(blue_sphere)

        # Test Layer 5 sentence phrase extraction
        create_commands = [
            "create a red box", 
            "make a blue sphere"
        ]
        
        for command in create_commands:
            result = self.executor.execute_layer5(command)
            assert result.success, f"Layer 5 should succeed: {command}"
            assert len(result.sentence_phrases) > 0, f"Should extract sentence phrases: {command}"

        # Now test Layer 2 grounding against the existing objects
        grounding_phrases = [
            "the red box",
            "the blue sphere"
        ]
        
        for phrase in grounding_phrases:
            result = self.executor.execute_layer2(phrase)
            assert result.success, f"Layer 2 should succeed: {phrase}"
            
            # Should find grounding results
            assert len(result.grounding_results) > 0, f"Should have grounding results for: {phrase}"            # At least one should be successful
            successful_groundings = [gr for gr in result.grounding_results if gr.success]
            assert len(successful_groundings) > 0, f"Should have successful grounding for: {phrase}"

    def test_layer5_confidence_propagation(self):
        """Test that confidence scores propagate correctly through layers."""
        result = self.executor.execute_layer5("create a red box")

        assert result.success, "Layer 5 should succeed"
        assert 0.0 < result.confidence <= 1.0, "Should have valid confidence score"
        
        # Layer 5 confidence should be based on Layer 4 confidence
        assert result.layer4_result.success, "Layer 4 should succeed"
        assert result.layer4_result.confidence > 0.0, "Layer 4 should have confidence"

    def test_layer5_error_handling(self):
        """Test Layer 5 error handling with invalid input."""
        # Test with non-verb input
        result = self.executor.execute_layer5("red box green")

        # Should still succeed at tokenization level, but may not find VPs
        # The exact behavior depends on implementation - document what we expect
        print(f"Result for non-VP input: success={result.success}, Sentence count={len(result.sentence_phrases)}")
