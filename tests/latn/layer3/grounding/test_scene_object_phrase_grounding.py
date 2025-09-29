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
from engraf.lexer.vocabulary_builder import vector_from_word
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject


class TestLayer3SpatialValidation(unittest.TestCase):
    """Test Layer 3 spatial validation and attention resolution."""
    
    def setUp(self):
        """Set up complex scene for Layer 3 spatial validation tests."""
        self.scene = SceneModel()
        
        # Create scene objects with distinct vectors
        box_vector = vector_from_word("box")
        table_vector = vector_from_word("table")
        pyramid_vector = vector_from_word("pyramid")
        sphere_vector = vector_from_word("sphere")
        
        self.box = SceneObject("box", box_vector, object_id="box-1")
        self.table = SceneObject("table", table_vector, object_id="table-1")
        self.pyramid = SceneObject("pyramid", pyramid_vector, object_id="pyramid-1")
        self.sphere = SceneObject("sphere", sphere_vector, object_id="sphere-1")

        self.box.position = {"x":0, "y":1, "z":0}      # Box is above the table
        self.table.position = {"x":0, "y":0, "z":0}    # Table is reference
        self.pyramid.position = {"x":-2, "y":0, "z":0} # Table is right of the pyramid  
        self.sphere.position = {"x":-2, "y":2, "z":0}  # Pyramid is under the sphere
        # Update vector positions to match
        self.box.vector['locX'], self.box.vector['locY'], self.box.vector['locZ'] = 0, 1, 0
        self.table.vector['locX'], self.table.vector['locY'], self.table.vector['locZ'] = 0, 0, 0
        self.pyramid.vector['locX'], self.pyramid.vector['locY'], self.pyramid.vector['locZ'] = 2, 0, 0
        self.sphere.vector['locX'], self.sphere.vector['locY'], self.sphere.vector['locZ'] = 0, -2, 0

        # Set reasonable scale values for spatial calculations
        for obj in [self.box, self.table, self.pyramid, self.sphere]:
            obj.vector['scaleX'] = 1.0
            obj.vector['scaleY'] = 1.0 
            obj.vector['scaleZ'] = 1.0
        
        # Add all objects to scene
        self.scene.add_object(self.box)
        self.scene.add_object(self.table)
        self.scene.add_object(self.pyramid) 
        self.scene.add_object(self.sphere)
        
        # Set up Layer 3 executor with scene
        self.layer3_executor = LATNLayerExecutor(self.scene)
        
        # Test sentence with spatial_location prepositions (including multi-word "right of")
    
    def test_simple_pp_attachment(self):
        """Test simple Layer 3 PP attachment without complex chains."""
        sentence = "move the box above the table to [3,4,5]"    
        # Execute Layer 3 directly on the sentence
        print(f"\n🔬 Processing through Layer 3 executor...")
        layer3_result = self.layer3_executor.execute_layer3(sentence, report=True)
        assert layer3_result.success, "Layer 3 should process successfully"
        assert len(layer3_result.hypotheses) == 2, "Should generate 2 hypotheses"
        hyp0 = layer3_result.hypotheses[0]
        tokens = hyp0.tokens
        assert len(tokens) == 4, f"Should have exactly 4 tokens, got {len(hyp0.tokens)}"
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
        assert len(layer3_result.hypotheses) == 6, "Should generate 7 hypotheses"
        #assert False

        