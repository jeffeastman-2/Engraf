#!/usr/bin/env python3
"""Debug test for empty scene behavior."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_empty_scene_behavior():
    """Test behavior when scene is empty."""
    print("================================================================================")
    print("DEBUG: Empty Scene Behavior Test")
    print("================================================================================")
    
    # Create interpreter with MockRenderer
    mock_renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=mock_renderer)
    
    print(f"Initial scene objects: {len(interpreter.scene.objects)}")
    
    # Add an object first
    print("\n1. Adding an object...")
    result = interpreter.interpret('draw a red cube')
    print(f"Draw result: {result['success']}")
    print(f"Objects in scene after draw: {len(interpreter.scene.objects)}")
    
    # Clear the scene
    print("\n2. Clearing scene...")
    interpreter.clear_scene()
    print(f"Objects in scene after clear: {len(interpreter.scene.objects)}")
    
    # Try rotation on empty scene
    print("\n3. Testing rotation on empty scene...")
    result = interpreter.interpret('rotate it by [45,45,45]')
    print(f"Rotate result: {result}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Objects modified: {result['objects_modified']}")
    
    # Verify scene is still empty
    print(f"Objects in scene after rotate attempt: {len(interpreter.scene.objects)}")
    
    # Test manual exception path
    print("\n4. Testing ObjectResolver directly...")
    from engraf.interpreter.handlers.object_resolver import ObjectResolver
    resolver = ObjectResolver(interpreter.scene, interpreter._last_acted_object)
    target_objects = resolver.resolve_target_objects('it')
    print(f"Target objects for 'it': {target_objects}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_empty_scene_behavior()
