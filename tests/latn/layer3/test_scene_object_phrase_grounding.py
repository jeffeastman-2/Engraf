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
        """Set up complex scene and Layer 2 parsing for spatial validation tests."""
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
        
        # Add all objects to scene
        self.scene.add_object(self.box)
        self.scene.add_object(self.table)
        self.scene.add_object(self.pyramid) 
        self.scene.add_object(self.sphere)
        
        # Set up Layer 2 executor with scene for grounding
        self.layer2_executor = LATNLayerExecutor(self.scene)
        self.layer3_processor = Layer3SemanticGrounder(self.scene)
        
        # Parse the complex sentence with Layer 2 to get token stream
        self.sentence = "move the box on the table beside the pyramid under the sphere to [3,4,5]"
        self._setup_layer2_token_stream()
    
    def _setup_layer2_token_stream(self):
        """Parse sentence through Layer 2 executor to get proper token stream."""
        # Execute Layer 2 with semantic grounding enabled
        self.layer2_result = self.layer2_executor.execute_layer2(
            self.sentence, 
            enable_semantic_grounding=True
        )
        
        self.assertTrue(self.layer2_result.success, "Layer 2 should process successfully")
        self.assertGreater(len(self.layer2_result.hypotheses), 0, "Should have hypotheses")
        
        # Get the best hypothesis token stream
        self.best_hypothesis = self.layer2_result.hypotheses[0]
        self.token_stream = self.best_hypothesis.tokens
        
        print(f"Layer 2 token stream ({len(self.token_stream)} tokens):")
        for i, token in enumerate(self.token_stream):
            print(f"  [{i}] {token}")  # VectorSpace.__repr__ already shows nonzero dimensions
        
        # Store references to the noun phrases for easier access
        self.noun_phrases = self.layer2_result.noun_phrases
        print(f"\nExtracted {len(self.noun_phrases)} noun phrases:")
        for i, np in enumerate(self.noun_phrases):
            resolved_obj = np.get_resolved_object() if isinstance(np, SceneObjectPhrase) and np.is_resolved() else None
            np_text = getattr(np, 'word', None) or str(np)
            print(f"  [{i}] {np_text} -> {resolved_obj.object_id if resolved_obj else 'unresolved'}")
        
        # Check Layer 2 grounding results
        print(f"\nLayer 2 grounding results ({len(self.layer2_result.grounding_results)}):")
        for i, result in enumerate(self.layer2_result.grounding_results):
            print(f"  [{i}] Success: {result.success}, Confidence: {result.confidence:.2f}")
            if result.resolved_object:
                print(f"      Object: {result.resolved_object.object_id}")
            else:
                print(f"      Error: {result.description}")
        
        # Check if grounding actually happened
        if not self.layer2_result.grounding_results:
            print("⚠️  No grounding results - semantic grounding may not have been applied!")
        
        # Check scene objects
        print(f"\nScene contains {len(self.scene.objects)} objects:")
        for obj in self.scene.objects:
            print(f"  - {obj.object_id}: {getattr(obj, 'word', 'no word')}")
    
    def test_layer2_token_stream_structure(self):
        """Verify that Layer 2 produces the expected token stream structure."""
        # This test verifies our Layer 2 executor creates the correct token stream
        # for the sentence: "move the box on the table beside the pyramid under the sphere to [3,4,5]"
        
        self.assertIsNotNone(self.token_stream, "Should have token stream")
        self.assertGreater(len(self.token_stream), 0, "Should have tokens")
        
        print(f"✓ Layer 2 token stream verified: {len(self.token_stream)} tokens")
        for i, token in enumerate(self.token_stream):
            print(f"  [{i}] {token.word}")
        
        print(f"✓ Layer 2 noun phrases: {len(self.noun_phrases)}")
        for i, np in enumerate(self.noun_phrases):
            resolved_obj = np.get_resolved_object() if isinstance(np, SceneObjectPhrase) and np.is_resolved() else None
            np_text = getattr(np, 'word', None) or str(np)
            print(f"  [{i}] {np_text} -> {resolved_obj.object_id if resolved_obj else 'unresolved'}")
    
    def test_layer3_spatial_validation_with_positioned_objects(self):
        """Test Layer 3 spatial validation with all objects positioned at [0,0,0].
        
        This skeleton test allows positioning all scene objects and then testing
        Layer 3's spatial validation and attention resolution logic.
        """
        # Position all objects at [0,0,0] to start
        self.box.position = [0, 0, 0]
        self.table.position = [0, 0, 0] 
        self.pyramid.position = [0, 0, 0]
        self.sphere.position = [0, 0, 0]
        
        print(f"Object positions:")
        print(f"  Box: {self.box.position}")
        print(f"  Table: {self.table.position}")
        print(f"  Pyramid: {self.pyramid.position}")
        print(f"  Sphere: {self.sphere.position}")
        
