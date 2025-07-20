#!/usr/bin/env python3
"""
Test transform functionality specifically
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_transforms():
    print("🔧 Testing Transform Functionality")
    print("=" * 50)
    
    # Use mock renderer
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # First create an object
    print("\n1. Creating an object to transform:")
    result = interpreter.interpret("draw a red cube")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Created: {result['objects_created']}")
    else:
        print(f"   Error: {result['message']}")
    
    # Test move command with pronoun
    print("\n2. Testing move with pronoun:")
    result = interpreter.interpret("move it to [5, 0, 0]")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Modified: {result['objects_modified']}")
    else:
        print(f"   Error: {result['message']}")
    
    # Test move command with description
    print("\n3. Testing move with description:")
    result = interpreter.interpret("move the red cube to [0, 5, 0]")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Modified: {result['objects_modified']}")
    else:
        print(f"   Error: {result['message']}")
    
    # Test scale command
    print("\n4. Testing scale:")
    result = interpreter.interpret("scale the cube by 2")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Modified: {result['objects_modified']}")
    else:
        print(f"   Error: {result['message']}")
    
    # Test rotate command
    print("\n5. Testing rotate:")
    result = interpreter.interpret("rotate the cube by 45 degrees")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Modified: {result['objects_modified']}")
    else:
        print(f"   Error: {result['message']}")
    
    # Show current objects
    print(f"\n📊 Current objects:")
    for obj in interpreter.scene.objects:
        print(f"   {obj.object_id}: {obj.name}")
        print(f"     Position: ({obj.vector['locX']:.1f}, {obj.vector['locY']:.1f}, {obj.vector['locZ']:.1f})")
    
    print("\n✅ Transform test completed!")

if __name__ == "__main__":
    test_transforms()
