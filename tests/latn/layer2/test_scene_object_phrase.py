#!/usr/bin/env python3
"""
Unit tests for SceneObjectPhrase functionality and Layer 2 grounding integration.
"""

import unittest
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace, vector_from_features
from engraf.lexer.vocabulary_builder import vector_from_word
from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS


class TestSceneObjectPhrase(unittest.TestCase):
    """Test SceneObjectPhrase creation and functionality."""
    
    def test_scene_object_phrase_creation(self):
        """Test creating SceneObjectPhrase from NounPhrase."""
        # Create a simple NounPhrase
        np = NounPhrase()
        
        # Add a noun token from vocabulary
        noun_token = vector_from_word("cube")
        np.apply_noun(noun_token)
        
        # Add a determiner from vocabulary
        det_token = vector_from_word("the")
        np.apply_determiner(det_token)
        
        # Create SceneObjectPhrase from NP
        so = SceneObjectPhrase.from_noun_phrase(np)
        
        # Verify SO marker was added
        self.assertEqual(so.vector["SO"], 1.0, "SO dimension should be set to 1.0")
        
        # Verify other attributes were copied
        self.assertEqual(so.noun, np.noun, "Noun should be copied")
        self.assertEqual(so.determiner, np.determiner, "Determiner should be copied")
        self.assertEqual(len(so.consumed_tokens), len(np.consumed_tokens), "Consumed tokens should be copied")
        
        # Test resolution to scene object
        scene_obj_vector = vector_from_word("cube")
        scene_obj = SceneObject("cube", scene_obj_vector, object_id="test_cube")
        
        so.resolve_to_scene_object(scene_obj)
        self.assertTrue(so.is_resolved(), "SO should be resolved")
        self.assertEqual(so.get_resolved_object(), scene_obj, "Resolved object should match")
    
    def test_np_vs_so_difference(self):
        """Test that NP and SO are different types with different capabilities."""
        # Create NP
        np = NounPhrase()
        noun_token = vector_from_word("box")
        np.apply_noun(noun_token)
        
        # Create SO from NP
        so = SceneObjectPhrase.from_noun_phrase(np)
        
        # Verify types
        self.assertIsInstance(np, NounPhrase)
        self.assertIsInstance(so, SceneObjectPhrase)
        self.assertIsInstance(so, NounPhrase)  # SO should inherit from NP
        
        # Verify SO dimension
        self.assertEqual(np.vector["SO"], 0.0, "NP should not have SO dimension")
        self.assertEqual(so.vector["SO"], 1.0, "SO should have SO dimension")
        
        # Verify NP doesn't have resolve methods
        self.assertFalse(hasattr(np, 'resolve_to_scene_object'), "NP should not have resolve_to_scene_object method")
        self.assertFalse(hasattr(np, 'is_resolved'), "NP should not have is_resolved method")
        self.assertFalse(hasattr(np, 'get_resolved_object'), "NP should not have get_resolved_object method")
        
        # Verify SO has resolve methods
        self.assertTrue(hasattr(so, 'resolve_to_scene_object'), "SO should have resolve_to_scene_object method")
        self.assertTrue(hasattr(so, 'is_resolved'), "SO should have is_resolved method")
        self.assertTrue(hasattr(so, 'get_resolved_object'), "SO should have get_resolved_object method")
    
    def test_scene_object_phrase_repr(self):
        """Test that SceneObjectPhrase has proper string representation."""
        np = NounPhrase()
        noun_token = vector_from_word("sphere")
        np.apply_noun(noun_token)
        
        so = SceneObjectPhrase.from_noun_phrase(np)
        
        # Before resolution
        repr_str = repr(so)
        self.assertIn("SO(", repr_str, "Should start with SO(")
        self.assertIn("noun=sphere", repr_str, "Should include noun")
        
        # After resolution
        sphere_vector = vector_from_word("sphere")
        scene_obj = SceneObject("sphere", sphere_vector, object_id="test_sphere")
        so.resolve_to_scene_object(scene_obj)
        
        repr_str = repr(so)
        self.assertIn("resolved_to=test_sphere", repr_str, "Should include resolved object ID")


class TestLayer2GroundingIntegration(unittest.TestCase):
    """Test Layer 2 grounding integration with SceneObjectPhrase."""
    
    def setUp(self):
        """Set up test scene and objects."""
        self.scene = SceneModel()
        
        # Create a red cube - combine cube + red features
        cube_vector = vector_from_word("cube")
        red_vector = vector_from_word("red")
        red_cube_vector = cube_vector.copy()
        red_cube_vector["red"] = red_vector["red"]
        red_cube_vector["green"] = red_vector["green"] 
        red_cube_vector["blue"] = red_vector["blue"]
        self.red_cube = SceneObject("cube", red_cube_vector, object_id="red_cube")
        self.scene.add_object(self.red_cube)
        
        # Create a blue sphere - combine sphere + blue features
        sphere_vector = vector_from_word("sphere")
        blue_vector = vector_from_word("blue")
        blue_sphere_vector = sphere_vector.copy()
        blue_sphere_vector["red"] = blue_vector["red"]
        blue_sphere_vector["green"] = blue_vector["green"]
        blue_sphere_vector["blue"] = blue_vector["blue"]
        self.blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere")
        self.scene.add_object(self.blue_sphere)
        
        self.grounder = Layer2SemanticGrounder(self.scene)
    
    def test_layer2_grounding_creates_scene_object_phrase(self):
        """Test that Layer 2 grounding creates SceneObjectPhrase instances."""
        # Create a NounPhrase for "red cube" that matches scene object name
        np = NounPhrase()
        
        # Add adjective "red" from vocabulary
        red_token = vector_from_word("red")
        np.apply_adjective(red_token)
        
        # Add noun "cube" from vocabulary (matches scene object name)
        cube_token = vector_from_word("cube")
        np.apply_noun(cube_token)
        
        # Verify original NP doesn't have resolution methods
        self.assertFalse(hasattr(np, 'resolve_to_scene_object'), "Original NP should not have resolution methods")
        
        # Ground the NP using Layer 2
        result = self.grounder.ground(np)
        
        # Debug output if grounding fails
        if not result.success:
            print(f"Grounding failed: {result.description}")
            print(f"NP vector: {[(dim, np.vector[dim]) for dim in VECTOR_DIMENSIONS if np.vector[dim] != 0.0]}")
            print(f"Red cube vector: {[(dim, self.red_cube.vector[dim]) for dim in VECTOR_DIMENSIONS if self.red_cube.vector[dim] != 0.0]}")
        
        # Verify results
        self.assertTrue(result.success, f"Grounding should succeed. Error: {result.description}")
        self.assertEqual(result.resolved_object, self.red_cube, "Should resolve to red cube")
        self.assertIsNotNone(result.scene_object_phrase, "Should create SceneObjectPhrase")
        self.assertIsInstance(result.scene_object_phrase, SceneObjectPhrase, "Should be SceneObjectPhrase type")
        self.assertTrue(result.scene_object_phrase.is_resolved(), "SO should be resolved")
        self.assertEqual(result.scene_object_phrase.get_resolved_object(), self.red_cube, "SO should resolve to red cube")
        self.assertEqual(result.scene_object_phrase.vector["SO"], 1.0, "SO should have SO dimension")
        
        # Verify original NP is unchanged
        self.assertEqual(np.vector["SO"], 0.0, "Original NP should not have SO dimension")
    
    def test_layer2_grounding_multiple_candidates(self):
        """Test Layer 2 grounding with multiple possible matches."""
        # Create a generic NP that could match either object
        np = NounPhrase()
        noun_token = vector_from_word("object")
        np.apply_noun(noun_token)
        
        # Ground with return_all_matches=True
        result = self.grounder.ground(np, return_all_matches=True)
        
        if result.success:
            self.assertIsNotNone(result.scene_object_phrase, "Should create SceneObjectPhrase")
            self.assertIsInstance(result.scene_object_phrase, SceneObjectPhrase, "Should be SceneObjectPhrase type")
            self.assertTrue(result.scene_object_phrase.is_resolved(), "SO should be resolved")
            self.assertEqual(result.scene_object_phrase.vector["SO"], 1.0, "SO should have SO dimension")
    
    def test_layer2_grounding_failure(self):
        """Test Layer 2 grounding when no objects match."""
        # Create NP for something not in the scene
        np = NounPhrase()
        noun_token = vector_from_word("triangle")
        np.apply_noun(noun_token)
        
        # Ground the NP
        result = self.grounder.ground(np)
        
        # Should fail gracefully
        self.assertFalse(result.success, "Grounding should fail for non-existent object")
        self.assertIsNone(result.resolved_object, "Should not resolve to any object")
        self.assertIsNone(result.scene_object_phrase, "Should not create SceneObjectPhrase")


if __name__ == '__main__':
    unittest.main()
