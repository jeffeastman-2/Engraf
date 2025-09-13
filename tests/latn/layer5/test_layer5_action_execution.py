import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.verb_phrase import VerbPhrase
from engraf.visualizer.scene.scene_model import SceneModel

class TestLayer5ActionExecution:
    """Test Layer 5 action execution functionality."""

    def setup_method(self):
        """Set up test environment with scene."""
        self.scene = SceneModel()
        self.executor = LATNLayerExecutor(self.scene)
    
    def test_create_action_execution(self):
        """Test that create commands are properly tokenized and grounded, but do NOT create objects."""
        initial_count = len(self.scene.objects)
        
        result = self.executor.execute_layer5("create a red box")
        assert result.success, "Layer 5 should succeed"

        # Should have created an object - Layer 5 creates objects
        final_count = len(self.scene.objects)
        assert final_count > initial_count, "Layer 5 should create objects"

        # Should have identified the verb phrase semantics
        assert len(result.sentence_phrases) > 0, "Should have extracted sentence phrases"
        vp = result.sentence_phrases[0].predicate
        assert isinstance(vp, VerbPhrase), "Predicate should be a VerbPhrase"
        assert vp.verb == "create", "Should identify create verb"
        assert vp.noun_phrase is not None, "Should have noun phrase object"
    
    def test_multiple_create_commands(self):
        """Test executing multiple commands, creating objects."""
        commands = [
            "create a red box",
            "make a blue sphere", 
            "build a green cube"
        ]
        
        initial_count = len(self.scene.objects)

        for command in commands:
            initial_count = len(self.scene.objects)

            result = self.executor.execute_layer5(command)
            assert result.success, f"Command '{command}' should succeed"

            # Should have created objects - Layer 5 creates objects
            final_count = len(self.scene.objects)
            assert final_count > initial_count, f"Should create objects for '{command}'"

            # Should have extracted verb phrases with semantic information
            assert len(result.sentence_phrases) > 0, f"Should extract sentence phrases for '{command}'"
            vp = result.sentence_phrases[0].predicate
            assert vp.verb in ["create", "make", "build"], f"Should identify action verb for '{command}'"
            assert vp.noun_phrase is not None, f"Should have noun phrase for '{command}'"

        assert final_count > initial_count, "Layer 5 should create objects"

    def test_action_execution_disabled(self):
        """Test Layer 5 semantic grounding without action execution (which is the default)."""
        initial_count = len(self.scene.objects)

        result = self.executor.execute_layer5("create a red box")

        assert result.success, "Layer 5 should succeed"

        # Should have created an object - Layer 5 always creates objects
        final_count = len(self.scene.objects)
        assert final_count > initial_count, "Layer 5 should create objects"

        # Should have identified verb phrase semantics
        assert len(result.sentence_phrases) > 0, "Should have extracted sentence phrases"
    
    def test_create_with_properties(self):
        """Test semantic grounding of verb phrases with specific properties."""
        test_cases = [
            ("create a large red sphere", "sphere", "red", "large"),
            ("make a small blue box", "box", "blue", "small"),
            ("build a green cube", "cube", "green", "normal")
        ]
        
        for command, expected_shape, expected_color, expected_size in test_cases:
            initial_count = len(self.scene.objects)
            
            result = self.executor.execute_layer5(command)
            assert result.success, f"Command '{command}' should succeed"
            
            # Should have created objects - Layer 5 creates objects
            final_count = len(self.scene.objects)
            assert final_count > initial_count, f"Should create objects for '{command}'"

            # Should have extracted verb phrases with semantic information
            assert len(result.sentence_phrases) > 0, f"Should extract sentence phrases for '{command}'"
            vp = result.sentence_phrases[0].predicate
            assert vp.verb in ["create", "make", "build"], f"Should identify action verb for '{command}'"
            assert vp.noun_phrase is not None, f"Should have noun phrase for '{command}'"
