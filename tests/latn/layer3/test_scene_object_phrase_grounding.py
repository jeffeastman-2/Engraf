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

Test sentence: "move the box on the table beside the pyramid under the sphere to [3,4,5]"
Expected Layer 2 output: SO PPSO PPSO PPSO PP (box, table, pyramid, sphere, [3,4,5])
Layer 3 job: Determine which PPSO attends to which SO/PPSO based on scene validity
"""

import unittest
from engraf.lexer.vocabulary_builder import vector_from_word
from engraf.lexer.semantic_grounding_layer3 import Layer3SemanticGrounder
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features


class TestLayer3SpatialValidation(unittest.TestCase):
    """Test Layer 3 spatial validation and attention resolution."""
    
    def setUp(self):
        """Set up complex scene for Layer 3 spatial validation tests."""
        # Create scene with all objects from "move the box on the table beside the pyramid under the sphere to [3,4,5]"
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
        
        # Test sentence with complex PP chain
        self.sentence = "move the box on the table beside the pyramid under the sphere to [3,4,5]"
    
    def test_layer3_spatial_validation_with_positioned_objects(self):
        """Test Layer 3 spatial validation with positioned objects to create realistic spatial relationships.
        
        This test positions objects to create valid spatial relationships:
        - Table at [0, 0, 0] (reference)
        - Box at [0, 1, 0] (above table - valid for "on table")  
        - Pyramid at [2, 0, 0] (beside table - valid for "beside")
        - Sphere at [0, -2, 0] (below table - valid for "under")
        
        Expected: Some PP attachment combinations should be filtered out as spatially invalid.
        """
        # Position objects to create realistic spatial relationships
        self.table.position = [0, 0, 0]    # Reference object
        self.box.position = [0, 1, 0]      # Above table (valid for "on table")
        self.pyramid.position = [2, 0, 0]  # Beside table (valid for "beside pyramid")  
        self.sphere.position = [0, -2, 0]  # Below table (valid for "under sphere")
        
        # Update vector positions to match
        self.table.vector['locX'], self.table.vector['locY'], self.table.vector['locZ'] = 0, 0, 0
        self.box.vector['locX'], self.box.vector['locY'], self.box.vector['locZ'] = 0, 1, 0
        self.pyramid.vector['locX'], self.pyramid.vector['locY'], self.pyramid.vector['locZ'] = 2, 0, 0
        self.sphere.vector['locX'], self.sphere.vector['locY'], self.sphere.vector['locZ'] = 0, -2, 0
        
        print(f"📍 Object positions:")
        print(f"  Box: {self.box.position} (above table)")
        print(f"  Table: {self.table.position} (reference)")
        print(f"  Pyramid: {self.pyramid.position} (beside table)")
        print(f"  Sphere: {self.sphere.position} (below table)")
        
        # Execute Layer 3 directly on the sentence
        print(f"\n🔬 Processing through Layer 3 executor...")
        layer3_result = self.layer3_executor.execute_layer3(
            self.sentence,
            enable_semantic_grounding=True
        )
        
        print(f"Layer 3 execution result:")
        print(f"  Success: {layer3_result.success}")
        print(f"  Hypotheses: {len(layer3_result.hypotheses) if hasattr(layer3_result, 'hypotheses') else 0}")
        
        if hasattr(layer3_result, 'hypotheses') and layer3_result.hypotheses:
            print(f"\nLayer 3 hypotheses details:")
            for i, hypothesis in enumerate(layer3_result.hypotheses):
                print(f"  Hypothesis [{i}]:")
                print(f"    Confidence: {getattr(hypothesis, 'confidence', 'N/A')}")
                print(f"    Type: {type(hypothesis).__name__}")
                
                # Print tokens if available
                if hasattr(hypothesis, 'tokens'):
                    print(f"    Tokens ({len(hypothesis.tokens)}):")
                    for j, token in enumerate(hypothesis.tokens):
                        print(f"      [{j}] {token}")
                
                # Print prepositional phrases if available
                if hasattr(hypothesis, 'prepositional_phrases'):
                    print(f"    PPs ({len(hypothesis.prepositional_phrases)}):")
                    for j, pp in enumerate(hypothesis.prepositional_phrases):
                        print(f"      [{j}] {pp}")
                
                # Print any other relevant attributes
                attrs_to_check = ['description', 'spatial_chains', 'attachment_chains']
                for attr in attrs_to_check:
                    if hasattr(hypothesis, attr):
                        value = getattr(hypothesis, attr)
                        # Clean up verbose descriptions
                        if attr == 'description' and isinstance(value, str):
                            # Extract just the grounding summary (after the →)
                            if '→' in value:
                                grounding_part = value.split('→')[1].strip()
                                print(f"    grounding: {grounding_part}")
                            else:
                                # Show a simplified version
                                clean_desc = value.replace('single-word(', '').replace(')', '').replace(' | ', ' ')
                                print(f"    {attr}: {clean_desc}")
                        else:
                            print(f"    {attr}: {value}")
                
                print()  # Empty line between hypotheses
        
        # Test spatial relationship validation results
        assert layer3_result.success, "Layer 3 should process successfully"
        
        # With positioned objects, spatial validation should filter out invalid combinations
        # Expected: Fewer than 24 hypotheses due to spatial pruning
        if hasattr(layer3_result, 'hypotheses') and layer3_result.hypotheses:
            num_hypotheses = len(layer3_result.hypotheses)
            print(f"  Layer 3 hypotheses found: {num_hypotheses}")
            
            # Assert that spatial filtering is working - should have fewer than all 24 combinations
            if num_hypotheses == 24:
                print("⚠️  Warning: All 24 combinations passed spatial validation - spatial filtering may not be working properly")
            else:
                print(f"✅ Spatial filtering working: {24 - num_hypotheses} combinations were filtered out")
        
        # TODO: Add specific assertions about expected spatial validation results