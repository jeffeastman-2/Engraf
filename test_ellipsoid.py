#!/usr/bin/env python3
"""
Test ellipsoid functionality
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_ellipsoid():
    print("🧪 Testing Ellipsoid Support")
    print("=" * 40)
    
    # Use mock renderer to avoid VPython popups
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Test basic ellipsoid creation
    print("\n1. Testing basic ellipsoid creation:")
    result = interpreter.interpret("draw a red ellipsoid")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Created: {result['objects_created']}")
    else:
        print(f"   Error: {result['message']}")
    
    # Test tall ellipsoid (this should work!)
    print("\n2. Testing tall ellipsoid:")
    result = interpreter.interpret("draw a tall blue ellipsoid")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Created: {result['objects_created']}")
        # Check the object's vector
        obj = interpreter.scene.objects[-1]
        print(f"   Scale: X={obj.vector['scaleX']:.2f}, Y={obj.vector['scaleY']:.2f}, Z={obj.vector['scaleZ']:.2f}")
    else:
        print(f"   Error: {result['message']}")
    
    # Test very tall ellipsoid
    print("\n3. Testing very tall ellipsoid:")
    result = interpreter.interpret("draw a very tall green ellipsoid")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Created: {result['objects_created']}")
        # Check the object's vector
        obj = interpreter.scene.objects[-1]
        print(f"   Scale: X={obj.vector['scaleX']:.2f}, Y={obj.vector['scaleY']:.2f}, Z={obj.vector['scaleZ']:.2f}")
    else:
        print(f"   Error: {result['message']}")
    
    # Test wide ellipsoid
    print("\n4. Testing wide ellipsoid:")
    result = interpreter.interpret("draw a wide yellow ellipsoid")
    print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
    if result['success']:
        print(f"   Created: {result['objects_created']}")
        # Check the object's vector
        obj = interpreter.scene.objects[-1]
        print(f"   Scale: X={obj.vector['scaleX']:.2f}, Y={obj.vector['scaleY']:.2f}, Z={obj.vector['scaleZ']:.2f}")
    else:
        print(f"   Error: {result['message']}")
    
    # Final summary
    print(f"\n📊 Final Scene Summary:")
    summary = interpreter.get_scene_summary()
    print(f"   Total objects: {summary['total_objects']}")
    print(f"   Object types: {summary['object_types']}")
    
    print("\n✅ Ellipsoid test completed!")

if __name__ == "__main__":
    test_ellipsoid()
