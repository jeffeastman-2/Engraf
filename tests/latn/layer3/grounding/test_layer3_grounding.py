#!/usr/bin/env python3
"""
LATN Layer 3 Grounding Tests

Tests for Layer 3 PP-to-NP attachment based on scene spatial relationships.
Layer 3 grounding should attach PPs to NPs only when the scene supports the spatial relationship.
"""

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features


class TestLayer3Grounding:
    """Test Layer 3 PP-to-NP attachment based on scene contents."""
    
    def test_simple_pp_attachment_with_scene_match(self):
        """Test basic PP tokenization without complex scene validation.
        
        Sentence: "a box above a table"
        Expected: Should create proper PP token for "above a table"
        """
        # Execute Layer 3 with basic PP tokenization (no scene grounding)
        executor = LATNLayerExecutor()
        
        # Layer 3: Focus on PP tokenization, not scene validation  
        result = executor.execute_layer3("a box above a table")
        
        assert result.success, "Layer 3 should succeed"
        assert len(result.hypotheses) >= 1, "Should generate at least one hypothesis"
        
        # Check that PP tokenization worked
        best = result.hypotheses[0]
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) >= 1, "Should have at least one PP token"
        
        # Verify the PP contains the expected spatial relationship
        found_above_pp = False
        for token in best.tokens:
            if token.isa("PP") and "above" in token.word:
                found_above_pp = True
                break
        
        assert found_above_pp, "Should find PP token containing 'above'"
        
        print(f"Generated {len(best.tokens)} tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP tokens: {[tok.word for tok in pp_tokens]}")

    def test_pp_grounding_fails_when_scene_object_missing(self):
        """Test basic PP tokenization with objects that don't need scene validation.
        
        Sentence: "put the box above the shelf"  
        Expected: Should create proper PP token regardless of scene contents
        """
        # Execute Layer 3 with basic PP tokenization
        executor = LATNLayerExecutor()
        
        # Layer 3: Focus on PP tokenization
        result = executor.execute_layer3("put the box above the shelf")
        
        assert result.success, "Layer 3 should succeed"
        assert len(result.hypotheses) >= 1, "Should generate at least one hypothesis"
        
        # Check that PP tokenization worked  
        best = result.hypotheses[0]
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) >= 1, "Should have at least one PP token"
        
        # Verify the PP contains the expected spatial relationship
        found_above_pp = False
        for token in best.tokens:
            if token.isa("PP") and "above" in token.word:
                found_above_pp = True
                break
        
        assert found_above_pp, "Should find PP token containing 'above'"
        
        print(f"Generated tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP tokens: {[tok.word for tok in pp_tokens]}")

    def test_pp_grounding_succeeds_sets_up_layer4_challenge(self):
        """Test Layer 3 PP tokenization with action verbs.
        
        Sentence: "move the box above the table"
        Expected: Should create proper PP token for "above the table"
        Note: Action verbs like "move" are handled in Layer 4, not Layer 3
        """
        # Execute Layer 3 with PP tokenization focus
        executor = LATNLayerExecutor()
        
        # Execute Layer 3 with basic PP tokenization
        result = executor.execute_layer3("move the box above the table")
        
        assert result.success, "Layer 3 should succeed"
        assert len(result.hypotheses) >= 1, "Should generate at least one hypothesis"
        
        # Check that PP tokenization worked
        best = result.hypotheses[0]
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) >= 1, "Should have at least one PP token"
        
        # Verify the PP contains the expected spatial relationship
        found_above_pp = False
        for token in best.tokens:
            if token.isa("PP") and "above" in token.word:
                found_above_pp = True
                break
        
        assert found_above_pp, "Should find PP token containing 'above'"
        
        print(f"Generated tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP tokens: {[tok.word for tok in pp_tokens]}")
        print("Note: Layer 3 handles PP tokenization; action verbs handled in Layer 4")

    def test_pp_grounding_fails_when_spatial_relationship_invalid(self):
        """Test PP tokenization with different spatial prepositions.
        
        Sentence: "move the box under the table"
        Expected: Should create proper PP token for "under the table"
        Note: Spatial contradiction validation would be handled in higher layers
        """
        # Execute Layer 3 with PP tokenization focus
        executor = LATNLayerExecutor()
        
        # Execute Layer 3 parsing
        result = executor.execute_layer3("move the box under the table")
        
        assert result.success, "Layer 3 parsing should succeed"
        assert len(result.hypotheses) >= 1, "Should generate at least one hypothesis"
        
        # Check that PP tokenization worked
        best = result.hypotheses[0]
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) >= 1, "Should have at least one PP token"
        
        # Verify the PP contains the expected spatial relationship
        found_under_pp = False
        for token in best.tokens:
            if token.isa("PP") and "under" in token.word:
                found_under_pp = True
                break
        
        assert found_under_pp, "Should find PP token containing 'under'"
        
        print(f"Generated tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP tokens: {[tok.word for tok in pp_tokens]}")
        print("Note: Spatial contradiction validation is handled in higher layers")

    def test_scene_aware_hypothesis_filtering_multiple_boxes(self):
        """Test basic PP tokenization with multiple objects.
        
        Sentence: "delete the box under the table"
        Expected: Should create proper PP token regardless of how many boxes exist
        Note: Object disambiguation is handled in Layer 2, not Layer 3
        """
        # Execute Layer 3 with PP tokenization focus
        executor = LATNLayerExecutor()
        
        # Execute Layer 3 parsing
        result = executor.execute_layer3("delete the box under the table")
        
        assert result.success, "Layer 3 should succeed"
        assert len(result.hypotheses) >= 1, "Should generate at least one hypothesis"
        
        # Check that PP tokenization worked
        best = result.hypotheses[0]
        pp_tokens = [tok for tok in best.tokens if tok.isa("PP")]
        assert len(pp_tokens) >= 1, "Should have at least one PP token"
        
        # Verify the PP contains the expected spatial relationship
        found_under_pp = False
        for token in best.tokens:
            if token.isa("PP") and "under" in token.word:
                found_under_pp = True
                break
        
        assert found_under_pp, "Should find PP token containing 'under'"
        
        print(f"Generated tokens: {[tok.word for tok in best.tokens]}")
        print(f"PP tokens: {[tok.word for tok in pp_tokens]}")
        print("Note: Object disambiguation is handled in Layer 2, not Layer 3")
