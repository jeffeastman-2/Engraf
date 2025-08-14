"""
Tests for object movement and spatial relationship commands.

This module tests the ability to move objects using spatial relationships
like "above", "below", "next to", etc.
"""

import unittest
from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import MockVPythonRenderer
from engraf.utils.debug import set_debug


class TestObjectMovement(unittest.TestCase):
    """Test object movement and spatial relationships."""
    
    def setUp(self):
        """Set up test environment."""
        # Use debug mode for detailed test output
        set_debug(True)
        
        # Use mock renderer for testing
        self.renderer = MockVPythonRenderer()
        self.interpreter = SentenceInterpreter(renderer=self.renderer)
    
    def tearDown(self):
        """Clean up after tests."""
        set_debug(False)
    
    def test_basic_object_creation(self):
        """Test that objects can be created successfully."""
        # Create a cube
        result = self.interpreter.interpret('draw a red cube at [0, 0, 0]')
        self.assertTrue(result['success'])
        self.assertEqual(len(result['objects_created']), 1)
        self.assertIn('cube', result['objects_created'][0])
        
        # Create a sphere
        result = self.interpreter.interpret('draw a big blue sphere at [3, 0, 0]')
        self.assertTrue(result['success'])
        self.assertEqual(len(result['objects_created']), 1)
        self.assertIn('sphere', result['objects_created'][0])
        
        # Check that both objects exist in the scene
        self.assertEqual(len(self.interpreter.scene.objects), 2)
    
    def test_object_resolution_by_type(self):
        """Test that objects can be resolved by their type."""
        # Create objects
        self.interpreter.interpret('draw a red cube at [0, 0, 0]')
        self.interpreter.interpret('draw a big blue sphere at [3, 0, 0]')
        
        # Test parsing of movement command
        from engraf.atn.sentence import build_sentence_atn
        from engraf.atn.core import run_atn
        from engraf.lexer.token_stream import tokenize, TokenStream
        from engraf.pos.sentence_phrase import SentencePhrase
        
        sentence = 'move the sphere above the cube'
        tokens = tokenize(sentence)
        token_stream = TokenStream(tokens)
        sent = SentencePhrase()
        start, end = build_sentence_atn(sent, token_stream)
        parsed_sentence = run_atn(start, end, token_stream, sent)
        
        # Check that sentence parsed correctly
        self.assertIsNotNone(parsed_sentence)
        self.assertEqual(parsed_sentence.predicate.verb, 'move')
        
        # Test object resolver
        vp = parsed_sentence.predicate
        target_objects = self.interpreter.object_resolver.resolve_target_objects(vp)
        
        # Should find the sphere as the target object
        self.assertEqual(len(target_objects), 1)
        
        # Verify the found object is actually a sphere
        found_object_id = target_objects[0]
        found_object = self.interpreter.scene.find_object_by_id(found_object_id)
        self.assertIsNotNone(found_object)
        self.assertEqual(found_object.name, 'sphere')
    
    def test_sphere_movement_above_cube(self):
        """Test the specific issue: moving sphere above cube."""
        # Create initial objects
        cube_result = self.interpreter.interpret('draw a red cube at [0, 0, 0]')
        sphere_result = self.interpreter.interpret('draw a big blue sphere at [3, 0, 0]')
        
        # Verify objects were created
        self.assertTrue(cube_result['success'])
        self.assertTrue(sphere_result['success'])
        
        # Get initial sphere position
        sphere_obj = None
        for obj in self.interpreter.scene.objects:
            if 'sphere' in obj.object_id:
                sphere_obj = obj
                break
        
        self.assertIsNotNone(sphere_obj, "Sphere object should exist")
        initial_position = (sphere_obj.vector['locX'], sphere_obj.vector['locY'], sphere_obj.vector['locZ'])
        
        # Attempt to move sphere above cube
        move_result = self.interpreter.interpret('move the sphere above the cube')
        
        # This should succeed (even if it doesn't actually move the sphere yet)
        self.assertTrue(move_result['success'])
        
        # Check if the sphere was actually modified
        # This test will currently fail, documenting the bug we need to fix
        if move_result['objects_modified']:
            # If objects were modified, check the new position
            final_position = (sphere_obj.vector['locX'], sphere_obj.vector['locY'], sphere_obj.vector['locZ'])
            
            # Position should have changed, especially Y coordinate (above)
            self.assertNotEqual(initial_position, final_position, 
                              "Sphere position should change when moved above cube")
            
            # Y coordinate should be higher (above the cube)
            self.assertGreater(sphere_obj.vector['locY'], 0, 
                             "Sphere should be moved above (positive Y)")
        else:
            # Document the current bug: objects_modified is empty
            self.fail(f"BUG: move command reported success but objects_modified is empty: {move_result['objects_modified']}")
    
    def test_pronoun_movement(self):
        """Test moving objects using pronouns like 'it'."""
        # Create a cube
        result = self.interpreter.interpret('draw a red cube at [0, 0, 0]')
        self.assertTrue(result['success'])
        
        # Move it using pronoun
        result = self.interpreter.interpret('move it to [5, 5, 5]')
        self.assertTrue(result['success'])
        
        # Check that the cube was moved
        if result['objects_modified']:
            cube_obj = None
            for obj in self.interpreter.scene.objects:
                if 'cube' in obj.object_id:
                    cube_obj = obj
                    break
            
            self.assertIsNotNone(cube_obj)
            # Should be at the new position
            self.assertEqual(cube_obj.vector['locX'], 5)
            self.assertEqual(cube_obj.vector['locY'], 5)
            self.assertEqual(cube_obj.vector['locZ'], 5)
    
    def test_spatial_relationship_parsing(self):
        """Test that spatial relationships are parsed correctly."""
        from engraf.atn.sentence import build_sentence_atn
        from engraf.atn.core import run_atn
        from engraf.lexer.token_stream import tokenize, TokenStream
        from engraf.pos.sentence_phrase import SentencePhrase
        
        # Test different spatial prepositions
        spatial_commands = [
            'move the sphere above the cube',
            'move the cube below the sphere'
        ]
        
        for command in spatial_commands:
            with self.subTest(command=command):
                tokens = tokenize(command)
                token_stream = TokenStream(tokens)
                sent = SentencePhrase()
                start, end = build_sentence_atn(sent, token_stream)
                parsed_sentence = run_atn(start, end, token_stream, sent)
                
                self.assertIsNotNone(parsed_sentence, f"Failed to parse: {command}")
                self.assertEqual(parsed_sentence.predicate.verb, 'move')
                
                # Check that we have a noun phrase with a prepositional phrase
                noun_phrase = parsed_sentence.predicate.noun_phrase
                self.assertIsNotNone(noun_phrase, f"No noun phrase found in: {command}")
                
                # The structure should include prepositional phrases
                if hasattr(noun_phrase, 'PPs') and noun_phrase.PPs:
                    # Check spatial preposition in the prepositional phrase
                    pp = noun_phrase.PPs[0]
                    spatial_preps = ['above', 'below', 'next', 'behind', 'to']
                    self.assertIn(pp.preposition, spatial_preps, 
                                f"Unexpected preposition '{pp.preposition}' in: {command}")


class TestObjectResolver(unittest.TestCase):
    """Test the object resolver component specifically."""
    
    def setUp(self):
        """Set up test environment."""
        set_debug(True)
        self.renderer = MockVPythonRenderer()
        self.interpreter = SentenceInterpreter(renderer=self.renderer)
        
        # Create test objects
        self.interpreter.interpret('draw a red cube at [0, 0, 0]')
        self.interpreter.interpret('draw a blue sphere at [3, 0, 0]')
        self.interpreter.interpret('draw a green cylinder at [0, 3, 0]')
    
    def tearDown(self):
        """Clean up after tests."""
        set_debug(False)
    
    def test_resolve_by_noun_type(self):
        """Test resolving objects by their noun type."""
        from engraf.pos.noun_phrase import NounPhrase
        from engraf.lexer.vector_space import VectorSpace
        
        # Create noun phrase for "sphere"
        sphere_vector = VectorSpace(word='sphere')
        sphere_vector['noun'] = 1.0
        np = NounPhrase()
        np.noun = 'sphere'
        np.vector = sphere_vector
        
        # Resolve objects matching "sphere"
        objects = self.interpreter.object_resolver.find_objects_by_description(np)
        
        # Should find exactly one sphere
        self.assertEqual(len(objects), 1)
        
        # Get the found object and verify it's a sphere
        found_object_id = objects[0]
        found_object = self.interpreter.scene.find_object_by_id(found_object_id)
        self.assertIsNotNone(found_object)
        self.assertEqual(found_object.name, 'sphere')
    
    def test_resolve_by_color(self):
        """Test resolving objects by color."""
        from engraf.pos.noun_phrase import NounPhrase
        from engraf.lexer.vector_space import VectorSpace
        
        # Create noun phrase for "red cube"
        cube_vector = VectorSpace(word='cube')
        cube_vector['noun'] = 1.0
        cube_vector['red'] = 1.0  # Red color
        np = NounPhrase()
        np.noun = 'cube'
        np.vector = cube_vector
        
        objects = self.interpreter.object_resolver.find_objects_by_description(np)
        
        # Should find the red cube
        self.assertEqual(len(objects), 1)
        
        # Get the found object and verify it's a red cube
        found_object_id = objects[0]
        found_object = self.interpreter.scene.find_object_by_id(found_object_id)
        self.assertIsNotNone(found_object)
        self.assertEqual(found_object.name, 'cube')
        self.assertEqual(found_object.vector['red'], 1.0)
    
    def test_resolve_multiple_objects(self):
        """Test resolving when multiple objects match."""
        # Add another cube
        self.interpreter.interpret('draw a yellow cube at [6, 0, 0]')
        
        from engraf.pos.noun_phrase import NounPhrase
        from engraf.lexer.vector_space import VectorSpace
        
        # Create noun phrase for just "cube" (no color specified)
        cube_vector = VectorSpace(word='cube')
        cube_vector['noun'] = 1.0
        np = NounPhrase()
        np.noun = 'cube'
        np.vector = cube_vector
        
        objects = self.interpreter.object_resolver.find_objects_by_description(np)
        
        # Should find both cubes
        self.assertEqual(len(objects), 2)
        
        # Verify both found objects are cubes
        for obj_id in objects:
            found_object = self.interpreter.scene.find_object_by_id(obj_id)
            self.assertIsNotNone(found_object)
            self.assertEqual(found_object.name, 'cube')


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
