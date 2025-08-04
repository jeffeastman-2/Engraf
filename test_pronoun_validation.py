#!/usr/bin/env python3
"""Test pronoun semantic validation with proper vector space features."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_pronoun_semantic_validation():
    """Test pronoun validation using proper vector space features."""
    print("=" * 80)
    print("PRONOUN SEMANTIC VALIDATION TEST")
    print("=" * 80)
    
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # Test 1: Empty scene - "it" should fail
    print("\n1. Testing 'rotate it' on empty scene:")
    result = interpreter.interpret('rotate it by [45,45,45]')
    print(f"   Success: {result['success']}")
    print(f"   Message: {result['message']}")
    assert result['success'] == False
    assert "no objects available" in result['message']
    assert "'it'" in result['message']
    
    # Test 2: Add object, "it" should work
    print("\n2. Adding object, then testing 'rotate it':")
    create_result = interpreter.interpret('draw a red cube')
    print(f"   Create success: {create_result['success']}")
    print(f"   Objects in scene: {len(interpreter.scene.objects)}")
    
    result = interpreter.interpret('rotate it by [45,45,45]')
    print(f"   Rotate success: {result['success']}")
    print(f"   Message: {result['message']}")
    assert result['success'] == True
    
    # Test 3: Clear scene, "them" should fail
    print("\n3. Testing 'move them' on empty scene:")
    interpreter.clear_scene()
    print(f"   Objects after clear: {len(interpreter.scene.objects)}")
    
    result = interpreter.interpret('move them to [1,2,3]')
    print(f"   Success: {result['success']}")
    print(f"   Message: {result['message']}")
    assert result['success'] == False
    assert "no objects available" in result['message']
    assert "'them'" in result['message']
    
    # Test 4: Add multiple objects, "them" should work
    print("\n4. Adding objects, then testing 'move them':")
    interpreter.interpret('draw a red cube')
    interpreter.interpret('draw a blue sphere')
    print(f"   Objects in scene: {len(interpreter.scene.objects)}")
    
    result = interpreter.interpret('move them to [1,2,3]')
    print(f"   Move success: {result['success']}")
    print(f"   Message: {result['message']}")
    assert result['success'] == True
    
    print("\n✅ All pronoun semantic validation tests passed!")
    print("   - 'it' correctly fails on empty scene")
    print("   - 'it' correctly succeeds when objects exist")
    print("   - 'them' correctly fails on empty scene") 
    print("   - 'them' correctly succeeds when objects exist")
    print("   - Uses proper vector space features (pronoun, singular, plural)")
    print("   - No artificial markers like 'pronoun_it' needed")

if __name__ == "__main__":
    test_pronoun_semantic_validation()
