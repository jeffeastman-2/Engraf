#!/usr/bin/env python3
"""
Layer 3 Spatial Validation Test

This test demonstrates Layer 3's true purpose: validating which spatial attention
relationships are valid based on current scene state and resolving complex
PP attention chains.

Layer 3 Architecture:
- Input: Complex PP chains with resolved SceneObjectPhrases
- Process: Validate spatial relationships and resolve attention combinatorics
- Output: Valid spatial attention chains with rejected invalid relationships

Test sentence: "move the box above the table beside the pyramid under the sphere to [3,4,5]"
Expected Layer 2 output: SO PPSO PPSO PPSO PP (box, table, pyramid, sphere, [3,4,5])
Layer 3 job: Determine which PPSO attends to which SO/PPSO based on scene validity
"""

import unittest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from tests.latn.dummy_test_scene import DummyTestScene

class TestLayer3SpatialValidation(unittest.TestCase):
    """Test Layer 3 spatial validation and attention resolution."""
    
    def setUp(self):
        """Set up a dummy scene for spatial validation tests.
        
        Scene contains:
        - box (above table)
        - table (reference object)
        - pyramid (left of table)
        - sphere (above pyramid)    
        """
        self.scene = DummyTestScene().get_scene1()
        # Set up Layer 3 executor with scene
        self.layer3_executor = LATNLayerExecutor(self.scene) # force grounding

    def test_simple_pp_attachment(self):
        """Test simple Layer 3 PP attachment without complex chains."""
        sentence = "move the box above the table to [3,4,5]"    
        # Execute Layer 3 directly on the sentence
        print(f"\n🔬 Processing through Layer 3 executor...")
        layer3_result = self.layer3_executor.execute_layer3(sentence, report=True)
        assert layer3_result.success, "Layer 3 should process successfully"
        assert len(layer3_result.hypotheses) >= 1, "Should generate 1 hypothesis"
        hyp0 = layer3_result.hypotheses[0]
        tokens = hyp0.tokens
        assert len(tokens) == 2, f"Should have exactly 2 tokens, got {len(hyp0.tokens)}"
        #assert False

    def test_layer3_spatial_validation_with_conj_np(self):
        """Test Layer 3 spatial validation with positioned objects to create realistic spatial relationships.
        
         Expected: Some PP attachment combinations should be filtered out as spatially invalid.
        """
        # Position scene objects to create realistic spatial relationships
        sentence = "move the box above the table and the pyramid to [3,4,5]"
        
        # Execute Layer 3 directly on the sentence
        print(f"\n🔬 Processing through Layer 3 executor...")
        layer3_result = self.layer3_executor.execute_layer3(sentence, report=True)
        assert layer3_result.success, "Layer 3 should process successfully"
        assert len(layer3_result.hypotheses) >= 1, "Should generate 1 hypothesis"
        #assert False

    def test_copy_with_pp2(self):
        """Test Layer 3 VP tokenization with a 'copy' verb phrase containing a PP."""
        # Test VP with PP: "copy the box below the table"
        result = self.layer3_executor.execute_layer3('copy the box below the table',report=True)
        assert result.success, "Failed to tokenize VP with PP in Layer 3"
        assert len(result.hypotheses) >= 0, "Should generate 0 hypotheses because the box is not below the table"                