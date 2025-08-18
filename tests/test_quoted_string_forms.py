#!/usr/bin/env python3
"""
Unit tests for different forms of quoted strings in ENGRAF.

Tests three main forms:
1. User-defined object names: "call that an 'arch'"
2. Color definitions: "'sky blue' is blue and green"
3. Size definitions: "'huge' is very very large"
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import MockVPythonRenderer


class TestQuotedStringForms(unittest.TestCase):
    """Test different forms of quoted strings in natural language commands."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.renderer = MockVPythonRenderer()
        self.interpreter = SentenceInterpreter(renderer=self.renderer)
    
    def test_user_defined_object_names(self):
        """Test user-defined object names with 'call that an' syntax."""
        # Create some objects first
        result = self.interpreter.interpret("draw a box at [0, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("draw a sphere at [1, 0, 0]")
        self.assertTrue(result['success'])
        
        # Group them
        result = self.interpreter.interpret("group them")
        self.assertTrue(result['success'])
        self.assertIn('assembly-1', result.get('assemblies_created', []))
        
        # Name the group with user-defined name
        result = self.interpreter.interpret("name it 'arch'")
        self.assertTrue(result['success'])
        
        # Verify we can reference it by the custom name
        result = self.interpreter.interpret("move the arch to [5, 5, 5]")
        self.assertTrue(result['success'])
        
        # Test another custom name
        result = self.interpreter.interpret("draw a very tall cylinder at [2, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("call it 'tower'")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("color the tower red")
        self.assertTrue(result['success'])
    
    def test_color_definitions(self):
        """Test custom color definitions with quoted strings."""
        # Define custom colors
        result = self.interpreter.interpret("'sky blue' is blue and green")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("'sunset orange' is red and yellow")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("'forest green' is very green")
        self.assertTrue(result['success'])
        
        # Use the custom colors
        result = self.interpreter.interpret("draw a sky blue box at [0, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("draw a sunset orange sphere at [1, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("color it forest green")
        self.assertTrue(result['success'])
    
    def test_size_definitions(self):
        """Test custom size definitions with quoted strings."""
        # Define custom sizes
        result = self.interpreter.interpret("'huge' is very very large")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("'tiny' is very very small")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("'gigantic' is huge and large")
        self.assertTrue(result['success'])
        
        # Use the custom sizes
        result = self.interpreter.interpret("draw a huge box at [0, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("draw a tiny sphere at [1, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("make it gigantic")
        self.assertTrue(result['success'])
    
    def test_mixed_quoted_string_usage(self):
        """Test mixing all three forms of quoted strings."""
        # Define custom adjectives
        result = self.interpreter.interpret("'massive' is very very large")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("'ocean blue' is blue and green")
        self.assertTrue(result['success'])
        
        # Create objects using custom adjectives
        result = self.interpreter.interpret("draw a massive ocean blue cylinder at [0, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("draw a box at [2, 0, 0]")
        self.assertTrue(result['success'])
        
        # Group and name with custom name
        result = self.interpreter.interpret("group them")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("call that a 'monument'")
        self.assertTrue(result['success'])
        
        # Manipulate the custom-named object
        result = self.interpreter.interpret("move the monument to [5, 5, 5]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("rotate the monument by 45 degrees")
        self.assertTrue(result['success'])
    
    def test_proper_noun_vs_type_designation(self):
        """Test distinction between proper nouns and type designations."""
        # Create an object
        result = self.interpreter.interpret("draw a sphere at [0, 0, 0]")
        self.assertTrue(result['success'])
        
        # Test proper noun (should be unique name)
        result = self.interpreter.interpret("call it 'Charlie'")
        self.assertTrue(result['success'])
        
        # Test type designation (should be a category)
        result = self.interpreter.interpret("draw a box at [1, 0, 0]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("call that a 'building'")
        self.assertTrue(result['success'])
        
        # Verify we can reference both
        result = self.interpreter.interpret("move Charlie to [2, 2, 2]")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("color the building red")
        self.assertTrue(result['success'])
    
    def test_complex_adjective_combinations(self):
        """Test complex combinations of custom and built-in adjectives."""
        # Define custom adjectives that reference other adjectives
        result = self.interpreter.interpret("'enormous' is very very very large")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("'bright red' is very red")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("'super tall' is very very tall")
        self.assertTrue(result['success'])
        
        # Use in complex combinations
        result = self.interpreter.interpret("draw an enormous bright red super tall box at [0, 0, 0]")
        self.assertTrue(result['success'])
        
        # Define combinations of custom adjectives
        result = self.interpreter.interpret("'mega' is enormous and huge")
        self.assertTrue(result['success'])
        
        result = self.interpreter.interpret("draw a mega sphere at [3, 0, 0]")
        self.assertTrue(result['success'])
    
    def test_quoted_string_error_handling(self):
        """Test error handling for malformed quoted strings."""
        # Test incomplete naming commands
        result = self.interpreter.interpret("call that")
        self.assertFalse(result['success'])
        
        # Test invalid definition syntax
        result = self.interpreter.interpret("'invalid is missing quote")
        self.assertFalse(result['success'])
        
        # Test empty quoted strings
        result = self.interpreter.interpret("call that an ''")
        self.assertFalse(result['success'])
    
    def test_demo_sentence_sequence(self):
        """Test the exact sequence from the demo to ensure it works."""
        demo_commands = [
            "draw a very tall box at [-1, 0, 0]",
            "draw a very tall box at [1, 0, 0]", 
            "draw a very tall box at [0,1,0] and rotate it by 90 degrees",
            "group them",
            "call that an 'arch'",
            "move the arch to [5,5,5]",
            "draw a red cube at [0, 0, 0]",
            "move it to [2, 3, 4]",
            "make it bigger",
            "color it blue",
            "draw a big blue sphere at [3, 0, 0]",
            "move it above the cube",
            "'sky blue' is blue and green",
            "'huge' is very very large",
            "draw a huge sky blue cylinder at [-3, 0, 0]",
            "rotate it by 90 degrees"
        ]
        
        for i, command in enumerate(demo_commands, 1):
            with self.subTest(command=command, step=i):
                result = self.interpreter.interpret(command)
                self.assertTrue(result['success'], 
                               f"Command {i} failed: '{command}' - {result.get('message', '')}")


if __name__ == '__main__':
    unittest.main()
