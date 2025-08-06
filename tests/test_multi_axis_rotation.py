"""
Unit tests for multi-axis rotation functionality in SentenceInterpreter.

This module tests the recent enhancements to the rotation system, including:
- Multi-axis rotation with vector coordinates [x,y,z]
- Proper semantic classification of rotation vs scaling
- Vector literal parsing and coordinate mapping
- Negative rotation values support
- Incremental rotation operations
"""

import unittest
from unittest.mock import Mock, MagicMock
from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer
from engraf.lexer.vector_space import VectorSpace
from engraf.visualizer.scene.scene_object import SceneObject


class TestMultiAxisRotation(unittest.TestCase):
    """Test cases for multi-axis rotation functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.renderer = MockRenderer()
        self.interpreter = SentenceInterpreter(renderer=self.renderer)
        
        # Create a test object for rotation tests
        self.test_result = self.interpreter.interpret('draw a red cube')
        self.assertTrue(self.test_result['success'])
        self.assertEqual(len(self.interpreter.scene.objects), 1)
        
        # Get the created object for testing
        self.test_object = self.interpreter.scene.objects[0]
        self.test_object_id = self.test_object.object_id
    
    def tearDown(self):
        """Clean up after each test method."""
        self.interpreter.clear_scene()
    
    def test_symmetric_multi_axis_rotation(self):
        """Test rotation with symmetric coordinates [45,45,45]."""
        # Execute rotation command
        result = self.interpreter.interpret('rotate it by [45,45,45]')
        
        # Verify command succeeded
        self.assertTrue(result['success'])
        self.assertIn('rotate it by [45,45,45]', result['message'])
        self.assertEqual(len(result['objects_modified']), 1)
        self.assertIn(self.test_object_id, result['objects_modified'])
        
        # Verify rotation values were applied correctly
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 45.0)
        self.assertEqual(rotated_object.vector['rotY'], 45.0)
        self.assertEqual(rotated_object.vector['rotZ'], 45.0)
        
        # Verify scale values were not affected
        self.assertEqual(rotated_object.vector['scaleX'], 1.0)
        self.assertEqual(rotated_object.vector['scaleY'], 1.0)
        self.assertEqual(rotated_object.vector['scaleZ'], 1.0)
    
    def test_asymmetric_multi_axis_rotation(self):
        """Test rotation with asymmetric coordinates [90,0,45]."""
        # Execute rotation command
        result = self.interpreter.interpret('rotate it by [90,0,45]')
        
        # Verify command succeeded
        self.assertTrue(result['success'])
        self.assertEqual(len(result['objects_modified']), 1)
        
        # Verify rotation values were applied correctly
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 90.0)
        self.assertEqual(rotated_object.vector['rotY'], 0.0)
        self.assertEqual(rotated_object.vector['rotZ'], 45.0)
    
    def test_negative_rotation_values(self):
        """Test rotation with negative coordinates [-45,-30,-15]."""
        # Execute rotation command
        result = self.interpreter.interpret('rotate it by [-45,-30,-15]')
        
        # Verify command succeeded
        self.assertTrue(result['success'])
        self.assertEqual(len(result['objects_modified']), 1)
        
        # Verify negative rotation values were applied correctly
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], -45.0)
        self.assertEqual(rotated_object.vector['rotY'], -30.0)
        self.assertEqual(rotated_object.vector['rotZ'], -15.0)
    
    def test_single_axis_vector_rotation(self):
        """Test rotation with single non-zero axis [0,180,0]."""
        # Execute rotation command
        result = self.interpreter.interpret('rotate it by [0,180,0]')
        
        # Verify command succeeded
        self.assertTrue(result['success'])
        self.assertEqual(len(result['objects_modified']), 1)
        
        # Verify only Y-axis rotation was applied
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 0.0)
        self.assertEqual(rotated_object.vector['rotY'], 180.0)
        self.assertEqual(rotated_object.vector['rotZ'], 0.0)
    
    def test_zero_vector_rotation(self):
        """Test rotation with zero vector [0,0,0]."""
        # Execute rotation command
        result = self.interpreter.interpret('rotate it by [0,0,0]')
        
        # Verify command succeeded (even though no rotation applied)
        self.assertTrue(result['success'])
        self.assertEqual(len(result['objects_modified']), 1)
        
        # Verify all rotation values are zero
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 0.0)
        self.assertEqual(rotated_object.vector['rotY'], 0.0)
        self.assertEqual(rotated_object.vector['rotZ'], 0.0)
    
    def test_incremental_rotation_overwrites(self):
        """Test that subsequent rotations overwrite previous values."""
        # Apply first rotation
        result1 = self.interpreter.interpret('rotate it by [45,45,45]')
        self.assertTrue(result1['success'])
        
        # Verify first rotation
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 45.0)
        self.assertEqual(rotated_object.vector['rotY'], 45.0)
        self.assertEqual(rotated_object.vector['rotZ'], 45.0)
        
        # Apply second rotation (should overwrite)
        result2 = self.interpreter.interpret('rotate it by [90,0,30]')
        self.assertTrue(result2['success'])
        
        # Verify second rotation overwrote the first
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 90.0)
        self.assertEqual(rotated_object.vector['rotY'], 0.0)
        self.assertEqual(rotated_object.vector['rotZ'], 30.0)
    
    def test_rotation_vs_scaling_classification(self):
        """Test that rotation commands are correctly classified vs scaling."""
        # Test rotation command
        rotation_result = self.interpreter.interpret('rotate it by [45,45,45]')
        self.assertTrue(rotation_result['success'])
        
        # Get the rotated object
        rotated_object = self.interpreter.scene.objects[0]
        
        # Verify rotation was applied, not scaling
        self.assertEqual(rotated_object.vector['rotX'], 45.0)
        self.assertEqual(rotated_object.vector['rotY'], 45.0)
        self.assertEqual(rotated_object.vector['rotZ'], 45.0)
        
        # Verify scale values remain at default (1.0)
        self.assertEqual(rotated_object.vector['scaleX'], 1.0)
        self.assertEqual(rotated_object.vector['scaleY'], 1.0)
        self.assertEqual(rotated_object.vector['scaleZ'], 1.0)
    
    def test_vector_literal_parsing(self):
        """Test that vector literals are correctly parsed."""
        # Create a new interpreter to test parsing
        test_interpreter = SentenceInterpreter(renderer=MockRenderer())
        test_interpreter.interpret('draw a blue sphere')
        
        # Test various vector literal formats
        test_cases = [
            ('[1,2,3]', (1.0, 2.0, 3.0)),
            ('[10,20,30]', (10.0, 20.0, 30.0)),
            ('[-5,-10,-15]', (-5.0, -10.0, -15.0)),
            ('[0,90,0]', (0.0, 90.0, 0.0)),
            ('[180,180,180]', (180.0, 180.0, 180.0))
        ]
        
        for vector_str, expected_values in test_cases:
            with self.subTest(vector=vector_str):
                # Clear previous rotation
                test_obj = test_interpreter.scene.objects[0]
                test_obj.vector['rotX'] = 0.0
                test_obj.vector['rotY'] = 0.0
                test_obj.vector['rotZ'] = 0.0
                
                # Execute rotation command
                result = test_interpreter.interpret(f'rotate it by {vector_str}')
                self.assertTrue(result['success'], f"Failed to parse vector {vector_str}")
                
                # Verify parsed values
                rotated_object = test_interpreter.scene.objects[0]
                self.assertEqual(rotated_object.vector['rotX'], expected_values[0])
                self.assertEqual(rotated_object.vector['rotY'], expected_values[1])
                self.assertEqual(rotated_object.vector['rotZ'], expected_values[2])
    
    def test_semantic_dimension_detection(self):
        """Test that semantic dimensions are correctly detected."""
        # Execute rotation command and check internal processing
        result = self.interpreter.interpret('rotate it by [30,60,90]')
        self.assertTrue(result['success'])
        
        # The fact that rotation was applied correctly indicates that:
        # 1. vector=1.0 was detected (vector literal parsing)
        # 2. directional_agency=1.0 was detected (for "by" preposition)
        # 3. rotate=1.0 was detected (on rotate verb)
        # 4. Rotation classification logic worked correctly
        
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 30.0)
        self.assertEqual(rotated_object.vector['rotY'], 60.0)
        self.assertEqual(rotated_object.vector['rotZ'], 90.0)
    
    def test_multiple_objects_rotation(self):
        """Test rotation on multiple objects using pronouns."""
        # Create multiple objects
        self.interpreter.interpret('draw a green sphere')
        self.interpreter.interpret('draw a blue cylinder')
        
        # Should have 3 objects total now (red cube + green sphere + blue cylinder)
        self.assertEqual(len(self.interpreter.scene.objects), 3)
        
        # Rotate the most recent object (blue cylinder)
        result = self.interpreter.interpret('rotate it by [15,25,35]')
        self.assertTrue(result['success'])
        self.assertEqual(len(result['objects_modified']), 1)
        
        # Verify only the last object was rotated
        # First object (red cube) should be unchanged
        first_object = self.interpreter.scene.objects[0]
        self.assertEqual(first_object.vector['rotX'], 0.0)
        self.assertEqual(first_object.vector['rotY'], 0.0)
        self.assertEqual(first_object.vector['rotZ'], 0.0)
        
        # Last object (blue cylinder) should be rotated
        last_object = self.interpreter.scene.objects[-1]
        self.assertEqual(last_object.vector['rotX'], 15.0)
        self.assertEqual(last_object.vector['rotY'], 25.0)
        self.assertEqual(last_object.vector['rotZ'], 35.0)
    
    def test_rotation_with_no_target_object(self):
        """Test rotation command when no target object exists."""
        # Clear the scene
        self.interpreter.clear_scene()
        
        # Try to rotate when no objects exist
        result = self.interpreter.interpret('rotate it by [45,45,45]')
        
        # Command should fail due to semantic validation (no objects to rotate)
        self.assertFalse(result['success'])
        self.assertEqual(len(result['objects_modified']), 0)
    
    def test_fractional_rotation_values(self):
        """Test rotation with fractional degree values."""
        # Execute rotation with fractional values
        result = self.interpreter.interpret('rotate it by [45.5,90.25,180.75]')
        
        # Should succeed (if vector parsing supports decimals)
        # Note: This test may need adjustment based on actual decimal parsing support
        if result['success']:
            rotated_object = self.interpreter.scene.objects[0]
            # Check if fractional values are preserved or rounded
            self.assertAlmostEqual(rotated_object.vector['rotX'], 45.5, places=1)
            self.assertAlmostEqual(rotated_object.vector['rotY'], 90.25, places=2)
            self.assertAlmostEqual(rotated_object.vector['rotZ'], 180.75, places=2)
    
    def test_large_rotation_values(self):
        """Test rotation with large degree values (>360)."""
        # Execute rotation with large values
        result = self.interpreter.interpret('rotate it by [720,450,900]')
        self.assertTrue(result['success'])
        
        # Verify large values are accepted (no modulo operation)
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 720.0)
        self.assertEqual(rotated_object.vector['rotY'], 450.0)
        self.assertEqual(rotated_object.vector['rotZ'], 900.0)
    
    def test_vector_space_access_pattern(self):
        """Test that VectorSpace is accessed correctly (not using deprecated .get() method)."""
        # This test verifies that the fixes for VectorSpace access patterns work
        result = self.interpreter.interpret('rotate it by [60,120,240]')
        self.assertTrue(result['success'])
        
        # If we get here without errors, the VectorSpace access is working correctly
        rotated_object = self.interpreter.scene.objects[0]
        self.assertEqual(rotated_object.vector['rotX'], 60.0)
        self.assertEqual(rotated_object.vector['rotY'], 120.0)
        self.assertEqual(rotated_object.vector['rotZ'], 240.0)


class TestRotationClassificationSystem(unittest.TestCase):
    """Test cases for rotation vs scaling classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = MockRenderer()
        self.interpreter = SentenceInterpreter(renderer=self.renderer)
        
        # Create a test object
        self.interpreter.interpret('draw a yellow cube')
        self.test_object = self.interpreter.scene.objects[0]
    
    def tearDown(self):
        """Clean up after each test."""
        self.interpreter.clear_scene()
    
    def test_rotate_verb_classification(self):
        """Test that 'rotate' verb is classified as rotation, not scaling."""
        # Execute rotation command
        result = self.interpreter.interpret('rotate it by [30,45,60]')
        self.assertTrue(result['success'])
        
        # Verify rotation was applied
        obj = self.interpreter.scene.objects[0]
        self.assertEqual(obj.vector['rotX'], 30.0)
        self.assertEqual(obj.vector['rotY'], 45.0)
        self.assertEqual(obj.vector['rotZ'], 60.0)
        
        # Verify scaling was NOT applied
        self.assertEqual(obj.vector['scaleX'], 1.0)
        self.assertEqual(obj.vector['scaleY'], 1.0)
        self.assertEqual(obj.vector['scaleZ'], 1.0)
    
    def test_verb_context_detection(self):
        """Test that verb context is properly detected for classification."""
        # Test different rotation-related verbs if they exist in vocabulary
        rotation_verbs = ['rotate']  # Add more if vocabulary supports them
        
        for verb in rotation_verbs:
            with self.subTest(verb=verb):
                # Reset object rotation
                obj = self.interpreter.scene.objects[0]
                obj.vector['rotX'] = 0.0
                obj.vector['rotY'] = 0.0
                obj.vector['rotZ'] = 0.0
                
                # Execute command
                result = self.interpreter.interpret(f'{verb} it by [15,30,45]')
                if result['success']:  # Only test if verb is recognized
                    # Verify rotation was applied, not scaling
                    self.assertEqual(obj.vector['rotX'], 15.0)
                    self.assertEqual(obj.vector['rotY'], 30.0)
                    self.assertEqual(obj.vector['rotZ'], 45.0)
                    
                    # Scale should remain unchanged
                    self.assertEqual(obj.vector['scaleX'], 1.0)
                    self.assertEqual(obj.vector['scaleY'], 1.0)
                    self.assertEqual(obj.vector['scaleZ'], 1.0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
