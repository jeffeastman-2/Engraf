#!/usr/bin/env python3
"""
Layer 2 SceneObjectPhrase Creation Test

This test demonstrates the bug where Layer 2 grounding succeeds in matching
objects but fails to convert NounPhrases to SceneObjectPhrases.

Expected behavior:
- Layer 2 grounding finds matching scene objects ✅
- Successfully grounded NounPhrases become SceneObjectPhrases ❌ (BUG)
- SceneObjectPhrases have .is_resolved() = True ❌ (BUG)
- SceneObjectPhrases have .get_resolved_object() = scene object ❌ (BUG)
"""

import unittest
from engraf.lexer.vocabulary_builder import vector_from_word
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject


class TestLayer2SceneObjectPhraseCreation(unittest.TestCase):
    """Test Layer 2 conversion of grounded NPs to SceneObjectPhrases."""
    
    def setUp(self):
        """Set up scene with objects for grounding tests."""
        # Create scene with objects
        self.scene = SceneModel()
        
        # Create scene objects
        box_vector = vector_from_word("box")
        table_vector = vector_from_word("table")
        
        self.box = SceneObject("box", box_vector, object_id="box-1")
        self.table = SceneObject("table", table_vector, object_id="table-1")
        
        self.scene.add_object(self.box)
        self.scene.add_object(self.table)
        
        # Create Layer 2 executor with scene
        self.layer2_executor = LATNLayerExecutor(self.scene)
    
    def test_single_np_grounding_creates_scene_object_phrase(self):
        """Test that a single grounded NP becomes a SceneObjectPhrase."""
        # Simple sentence with one noun phrase
        sentence = "the box"
        
        result = self.layer2_executor.execute_layer2(sentence, enable_semantic_grounding=True)
        
        # Verify Layer 2 processing succeeded
        self.assertTrue(result.success, "Layer 2 should process successfully")
        self.assertEqual(len(result.noun_phrases), 1, "Should extract one noun phrase")
        self.assertEqual(len(result.grounding_results), 1, "Should have one grounding result")
        
        # Verify grounding succeeded
        grounding_result = result.grounding_results[0]
        self.assertTrue(grounding_result.success, "Grounding should succeed")
        self.assertEqual(grounding_result.resolved_object.object_id, "box-1", "Should match box-1")
        
        # THE BUG: Check if NP was converted to SceneObjectPhrase
        np = result.noun_phrases[0]
        print(f"NP type: {type(np)}")
        print(f"NP details: {np}")
        
        # This should pass but currently fails
        self.assertIsInstance(np, SceneObjectPhrase, 
            "Successfully grounded NP should become SceneObjectPhrase")
        self.assertTrue(np.is_resolved(), 
            "SceneObjectPhrase should be resolved")
        self.assertEqual(np.get_resolved_object().object_id, "box-1",
            "SceneObjectPhrase should reference the matched object")
    
    def test_multiple_np_grounding_creates_scene_object_phrases(self):
        """Test that multiple grounded NPs become SceneObjectPhrases."""
        # Sentence with multiple noun phrases
        sentence = "move the box on the table"
        
        result = self.layer2_executor.execute_layer2(sentence, enable_semantic_grounding=True)
        
        # Verify Layer 2 processing succeeded
        self.assertTrue(result.success, "Layer 2 should process successfully")
        self.assertEqual(len(result.noun_phrases), 2, "Should extract two noun phrases")
        self.assertEqual(len(result.grounding_results), 2, "Should have two grounding results")
        
        # Verify both groundings succeeded
        for i, grounding_result in enumerate(result.grounding_results):
            self.assertTrue(grounding_result.success, f"Grounding {i} should succeed")
            self.assertIsNotNone(grounding_result.resolved_object, f"Should have resolved object {i}")
        
        print(f"Grounding results:")
        for i, grounding_result in enumerate(result.grounding_results):
            print(f"  [{i}] {grounding_result.resolved_object.object_id} (success: {grounding_result.success})")
        
        # THE BUG: Check if NPs were converted to SceneObjectPhrases
        print(f"Noun phrases:")
        for i, np in enumerate(result.noun_phrases):
            print(f"  [{i}] Type: {type(np)}, Details: {np}")
            
            # This should pass but currently fails
            self.assertIsInstance(np, SceneObjectPhrase,
                f"Successfully grounded NP {i} should become SceneObjectPhrase")
            self.assertTrue(np.is_resolved(),
                f"SceneObjectPhrase {i} should be resolved")
            self.assertIsNotNone(np.get_resolved_object(),
                f"SceneObjectPhrase {i} should have resolved object")
    
    def test_mixed_grounding_success_and_failure(self):
        """Test mix of successful and failed grounding creates correct types."""
        # Test with an object that doesn't exist in scene
        sentence = "the sphere"  # sphere doesn't exist in scene
        
        result = self.layer2_executor.execute_layer2(sentence, enable_semantic_grounding=True)
        
        # Verify Layer 2 processing succeeded
        self.assertTrue(result.success, "Layer 2 should process successfully")
        self.assertEqual(len(result.noun_phrases), 1, "Should extract one noun phrase")
        self.assertEqual(len(result.grounding_results), 1, "Should have one grounding result")
        
        # Verify grounding failed
        sphere_result = result.grounding_results[0]
        self.assertFalse(sphere_result.success, "Sphere grounding should fail")
        
        print(f"Failed grounding result:")
        obj_id = sphere_result.resolved_object.object_id if sphere_result.resolved_object else "none"
        print(f"  Success: {sphere_result.success}, Object: {obj_id}")
        
        # THE BUG: Check type - failed grounding should remain NounPhrase
        sphere_np = result.noun_phrases[0]
        
        print(f"NP type after failed grounding: {type(sphere_np)}")
        
        # Failed grounding should remain NounPhrase (this should pass)
        self.assertIsInstance(sphere_np, NounPhrase,
            "Failed grounding should remain NounPhrase")
        self.assertNotIsInstance(sphere_np, SceneObjectPhrase,
            "Failed grounding should NOT become SceneObjectPhrase")
    
    def test_grounding_disabled_keeps_noun_phrases(self):
        """Test that with grounding disabled, NPs remain NounPhrases."""
        sentence = "the box"
        
        result = self.layer2_executor.execute_layer2(sentence, enable_semantic_grounding=False)
        
        # Verify Layer 2 processing succeeded
        self.assertTrue(result.success, "Layer 2 should process successfully")
        self.assertEqual(len(result.noun_phrases), 1, "Should extract one noun phrase")
        self.assertEqual(len(result.grounding_results), 0, "Should have no grounding results")
        
        # With grounding disabled, should remain NounPhrase
        np = result.noun_phrases[0]
        self.assertIsInstance(np, NounPhrase, "Should remain NounPhrase")
        self.assertNotIsInstance(np, SceneObjectPhrase, "Should NOT become SceneObjectPhrase")
        
        print(f"Grounding disabled - NP type: {type(np)}")
