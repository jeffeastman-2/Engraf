#!/usr/bin/env python3
"""
Test color conflict detection in object resolution.
"""

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_color_conflict():
    """Test that blue cube query doesn't match orange cube object."""
    
    # Create interpreter with mock renderer
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer)
    
    print("=== Testing color conflict detection ===")
    
    # Create an orange cube
    result1 = interpreter.interpret("draw an orange cube at [0, 0, 0]")
    print(f"Created object: {result1}")
    
    # Create a blue sphere for comparison
    result2 = interpreter.interpret("draw a blue sphere at [2, 0, 0]")
    print(f"Created object: {result2}")
    
    print(f"\n=== Scene state ===")
    for i, obj in enumerate(interpreter.scene.objects):
        print(f"Object {i}: id='{obj.object_id}', name='{obj.name}'")
        print(f"  Vector red value: {obj.vector['red']}")
        print(f"  Vector green value: {obj.vector['green']}")
        print(f"  Vector blue value: {obj.vector['blue']}")
    
    # Now test resolution for "blue cube" - should NOT match the orange cube
    print(f"\n=== Testing 'blue cube' resolution ===")
    
    # Test the sentence "move the blue cube" - this should find no matches
    result3 = interpreter.interpret("move the blue cube")
    print(f"Move blue cube result: {result3}")
    
    # Test the sentence "move the blue sphere" - this should find the sphere
    result4 = interpreter.interpret("move the blue sphere")
    print(f"Move blue sphere result: {result4}")
    
    print(f"\n=== Summary ===")
    print(f"✅ Orange cube was created successfully")
    print(f"✅ Blue sphere was created successfully") 
    print(f"❓ 'Move blue cube' should fail (no blue cubes available)")
    print(f"❓ 'Move blue sphere' should succeed (blue sphere exists)")

if __name__ == "__main__":
    test_color_conflict()
