#!/usr/bin/env python3
"""Test that semantic validation is properly integrated into the SentenceInterpreter."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.renderers.mock_renderer import MockRenderer
from engraf.lexer.vector_space import VectorSpace

def test_integrated_semantic_validation():
    """Test that SentenceInterpreter rejects semantically invalid commands."""
    print("================================================================================")
    print("INTEGRATION TEST: Semantic Validation in SentenceInterpreter")
    print("================================================================================")
    
    # Create interpreter with MockRenderer for testing
    mock_renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=mock_renderer)
    
    # Add a circle to the interpreter's scene
    circle_vector = VectorSpace()
    circle_vector.word = "circle"
    circle_vector.noun = 1.0
    circle = SceneObject(name="circle", vector=circle_vector)
    interpreter.scene.add_object(circle)
    
    print(f"Scene state: {len(interpreter.scene.objects)} circle(s) available")
    print()
    
    # Test valid command that should work
    print("Testing: 'move a circle to [1, 2, 3]'")
    print("Expected: ✅ Should succeed - 1 circle requested, 1 available")
    try:
        result = interpreter.interpret("move a circle to [1, 2, 3]")
        print(f"Result: ✅ SUCCESS - Command executed")
    except Exception as e:
        print(f"Result: ❌ UNEXPECTED ERROR - {e}")
    
    print("-" * 60)
    
    # Test invalid command that should fail
    print("Testing: 'move 3 circles to [1, 2, 3]'")
    print("Expected: ❌ Should fail - 3 circles requested, only 1 available")
    print("DEBUG: About to call interpreter.interpret()...")
    try:
        result = interpreter.interpret("move 3 circles to [1, 2, 3]")
        print(f"DEBUG: Result returned: {result}")
        if result.get('success', False):
            print(f"Result: ❌ UNEXPECTED SUCCESS - Command should have failed!")
        else:
            print(f"Result: ✅ CORRECT FAILURE - {result.get('message', 'Unknown error')}")
    except Exception as e:
        if "semantic" in str(e).lower() or "available" in str(e).lower():
            print(f"Result: ✅ CORRECT FAILURE - {e}")
        else:
            print(f"Result: ❌ UNEXPECTED ERROR TYPE - {e}")
    
    print("-" * 60)
    
    # Test another invalid command
    print("Testing: 'move the square to [1, 2, 3]'")
    print("Expected: ❌ Should fail - no squares in scene")
    try:
        result = interpreter.interpret("move the square to [1, 2, 3]")
        print(f"DEBUG: Result returned: {result}")
        if result.get('success', False):
            print(f"Result: ❌ UNEXPECTED SUCCESS - Command should have failed!")
        else:
            print(f"Result: ✅ CORRECT FAILURE - {result.get('message', 'Unknown error')}")
    except Exception as e:
        if "semantic" in str(e).lower() or "available" in str(e).lower():
            print(f"Result: ✅ CORRECT FAILURE - {e}")
        else:
            print(f"Result: ❌ UNEXPECTED ERROR TYPE - {e}")
    
    print()
    print("================================================================================")
    print("🎉 INTEGRATION TEST COMPLETED")
    print("================================================================================")

if __name__ == "__main__":
    test_integrated_semantic_validation()
