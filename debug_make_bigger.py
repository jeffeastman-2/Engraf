#!/usr/bin/env python3
"""
Debug script to test "make it bigger" command
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_make_bigger():
    """Test the 'make it bigger' command."""
    print("🔧 Testing 'make it bigger' command")
    print("=" * 40)
    
    # Create interpreter with mock renderer
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # First create a cube
    print("1. Creating a red cube...")
    result1 = interpreter.interpret('draw a red cube at [0, 0, 0]')
    print(f"   Result: {result1}")
    
    if result1['success'] and result1['objects_created']:
        cube_id = result1['objects_created'][0]
        cube_obj = None
        for obj in interpreter.scene.objects:
            if obj.object_id == cube_id:
                cube_obj = obj
                break
        
        if cube_obj:
            print(f"   Created cube: {cube_obj.name}")
            print(f"   Initial scale: X={cube_obj.vector['scaleX']}, Y={cube_obj.vector['scaleY']}, Z={cube_obj.vector['scaleZ']}")
    
    print()
    
    # Now try to make it bigger
    print("2. Making it bigger...")
    result2 = interpreter.interpret('make it bigger')
    print(f"   Result: {result2}")
    
    if cube_obj:
        print(f"   Final scale: X={cube_obj.vector['scaleX']}, Y={cube_obj.vector['scaleY']}, Z={cube_obj.vector['scaleZ']}")
        
        # Check if scale actually changed
        initial_scale = 1.0  # Default scale
        final_scale = cube_obj.vector['scaleX']
        if final_scale > initial_scale:
            print("   ✅ Scale increased successfully!")
        else:
            print("   ❌ Scale did not increase")
    
    print()

if __name__ == "__main__":
    test_make_bigger()
