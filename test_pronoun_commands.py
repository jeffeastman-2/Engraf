#!/usr/bin/env python3
"""
Test pronoun-based commands like 'make it bigger' and 'color it blue'
to debug issues in the demo sequence.
"""

import pytest
from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer


class TestPronounCommands:
    """Test pronoun-based modification commands."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    def test_make_it_bigger_sequence(self):
        """Test the exact sequence: draw cube, then make it bigger."""
        print("\n=== Testing 'make it bigger' sequence ===")
        
        # Step 1: Create a cube
        print("Step 1: Creating cube...")
        result1 = self.interpreter.interpret("draw a red cube at [0, 0, 0]")
        print(f"Result 1: {result1}")
        
        assert result1['success'] == True, f"Failed to create cube: {result1}"
        assert len(result1['objects_created']) == 1
        
        cube_id = result1['objects_created'][0]
        print(f"Created cube with ID: {cube_id}")
        
        # Check initial cube properties
        cube = next(obj for obj in self.interpreter.scene.objects if obj.object_id == cube_id)
        initial_scale_x = cube.vector['scaleX']
        print(f"Initial cube scaleX: {initial_scale_x}")
        
        # Step 2: Make it bigger
        print("\nStep 2: Making it bigger...")
        result2 = self.interpreter.interpret("make it bigger")
        print(f"Result 2: {result2}")
        
        if not result2['success']:
            print(f"ERROR: Failed to make it bigger: {result2['message']}")
            # Let's check what happened
            print(f"Scene objects: {[obj.object_id for obj in self.interpreter.scene.objects]}")
            
        assert result2['success'] == True, f"Failed to make it bigger: {result2}"
        
        # Check if cube was modified
        cube_after = next(obj for obj in self.interpreter.scene.objects if obj.object_id == cube_id)
        final_scale_x = cube_after.vector['scaleX']
        print(f"Final cube scaleX: {final_scale_x}")
        
        assert final_scale_x > initial_scale_x, f"Scale didn't increase: {initial_scale_x} -> {final_scale_x}"
    
    def test_color_it_blue_sequence(self):
        """Test the exact sequence: draw cube, then color it blue."""
        print("\n=== Testing 'color it blue' sequence ===")
        
        # Step 1: Create a red cube
        print("Step 1: Creating red cube...")
        result1 = self.interpreter.interpret("draw a red cube at [0, 0, 0]")
        print(f"Result 1: {result1}")
        
        assert result1['success'] == True
        cube_id = result1['objects_created'][0]
        
        # Check initial color
        cube = next(obj for obj in self.interpreter.scene.objects if obj.object_id == cube_id)
        initial_red = cube.vector['red']
        initial_blue = cube.vector['blue']
        print(f"Initial colors - red: {initial_red}, blue: {initial_blue}")
        
        # Step 2: Color it blue
        print("\nStep 2: Coloring it blue...")
        result2 = self.interpreter.interpret("color it blue")
        print(f"Result 2: {result2}")
        
        if not result2['success']:
            print(f"ERROR: Failed to color it blue: {result2['message']}")
            
        assert result2['success'] == True, f"Failed to color it blue: {result2}"
        
        # Check if color changed
        cube_after = next(obj for obj in self.interpreter.scene.objects if obj.object_id == cube_id)
        final_red = cube_after.vector['red']
        final_blue = cube_after.vector['blue']
        print(f"Final colors - red: {final_red}, blue: {final_blue}")
        
        assert final_blue > initial_blue, f"Blue didn't increase: {initial_blue} -> {final_blue}"
    
    def test_move_it_sequence(self):
        """Test the exact sequence: draw cube, then move it."""
        print("\n=== Testing 'move it' sequence ===")
        
        # Step 1: Create a cube
        print("Step 1: Creating cube...")
        result1 = self.interpreter.interpret("draw a red cube at [0, 0, 0]")
        print(f"Result 1: {result1}")
        
        assert result1['success'] == True
        cube_id = result1['objects_created'][0]
        
        # Check initial position
        cube = next(obj for obj in self.interpreter.scene.objects if obj.object_id == cube_id)
        initial_x = cube.vector['locX']
        initial_y = cube.vector['locY']
        initial_z = cube.vector['locZ']
        print(f"Initial position: [{initial_x}, {initial_y}, {initial_z}]")
        
        # Step 2: Move it
        print("\nStep 2: Moving it...")
        result2 = self.interpreter.interpret("move it to [1, 2, 3]")
        print(f"Result 2: {result2}")
        
        if not result2['success']:
            print(f"ERROR: Failed to move it: {result2['message']}")
            
        assert result2['success'] == True, f"Failed to move it: {result2}"
        
        # Check if position changed
        cube_after = next(obj for obj in self.interpreter.scene.objects if obj.object_id == cube_id)
        final_x = cube_after.vector['locX']
        final_y = cube_after.vector['locY']
        final_z = cube_after.vector['locZ']
        print(f"Final position: [{final_x}, {final_y}, {final_z}]")
        
        assert final_x == 1.0 and final_y == 2.0 and final_z == 3.0, \
            f"Position not updated correctly: [{final_x}, {final_y}, {final_z}]"
    
    def test_full_demo_sequence(self):
        """Test the full sequence from the demo."""
        print("\n=== Testing full demo sequence ===")
        
        commands = [
            "draw a red cube at [0, 0, 0]",
            "move it to [1, 2, 3]", 
            "make it bigger",
            "color it blue"
        ]
        
        for i, command in enumerate(commands, 1):
            print(f"\nStep {i}: {command}")
            result = self.interpreter.interpret(command)
            print(f"Result {i}: {result}")
            
            if not result['success']:
                print(f"ERROR at step {i}: {result['message']}")
                # Print scene state for debugging
                print(f"Scene objects: {[obj.object_id for obj in self.interpreter.scene.objects]}")
                if hasattr(result, 'sentence_parsed'):
                    print(f"Parsed sentence: {result['sentence_parsed']}")
                
            assert result['success'] == True, f"Failed at step {i} ('{command}'): {result['message']}"
    
    def test_pronoun_parsing(self):
        """Test if pronouns are being parsed correctly."""
        print("\n=== Testing pronoun parsing ===")
        
        # First create an object
        result1 = self.interpreter.interpret("draw a cube")
        assert result1['success'] == True
        
        # Test parsing of pronoun commands
        test_sentences = [
            "make it bigger",
            "color it blue", 
            "move it to [1, 2, 3]"
        ]
        
        for sentence in test_sentences:
            print(f"\nTesting sentence: '{sentence}'")
            try:
                result = self.interpreter.interpret(sentence)
                print(f"Parse result: {result}")
                
                if 'sentence_parsed' in result:
                    parsed = result['sentence_parsed']
                    print(f"Parsed sentence: {parsed}")
                    
                    # Check if pronoun was detected
                    if hasattr(parsed, 'predicate') and hasattr(parsed.predicate, 'noun_phrase'):
                        np = parsed.predicate.noun_phrase
                        if hasattr(np, 'vector'):
                            pronoun_value = np.vector.get('pronoun', 0)
                            print(f"Pronoun value in noun phrase: {pronoun_value}")
                            
            except Exception as e:
                print(f"Exception parsing '{sentence}': {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    # Run the tests directly
    test = TestPronounCommands()
    test.setup_method()
    
    try:
        test.test_make_it_bigger_sequence()
        print("✅ make it bigger test passed")
    except Exception as e:
        print(f"❌ make it bigger test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test.setup_method()  # Reset
        test.test_color_it_blue_sequence()
        print("✅ color it blue test passed")
    except Exception as e:
        print(f"❌ color it blue test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test.setup_method()  # Reset
        test.test_move_it_sequence()
        print("✅ move it test passed")
    except Exception as e:
        print(f"❌ move it test failed: {e}")
        import traceback
        traceback.print_exc()
