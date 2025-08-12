#!/usr/bin/env python3
"""
Full demo test: red cube, make it bigger, draw big blue sphere above cube
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_full_demo():
    print("🔧 Testing Full Demo Sequence")
    print("=" * 50)
    
    # Use mock renderer
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    print("\n1. Creating red cube:")
    result1 = interpreter.interpret("draw a red cube")
    print(f"   Result: {'✅ Success' if result1['success'] else '❌ Failed'}")
    if result1['success']:
        print(f"   Created: {result1['objects_created']}")
    else:
        print(f"   Error: {result1['message']}")
    
    print("\n2. Making it bigger:")
    result2 = interpreter.interpret("make it bigger")
    print(f"   Result: {'✅ Success' if result2['success'] else '❌ Failed'}")
    if result2['success']:
        print(f"   Modified: {result2['objects_modified']}")
    else:
        print(f"   Error: {result2['message']}")
    
    print("\n3. Creating big blue sphere above cube:")
    result3 = interpreter.interpret("draw a big blue sphere")
    print(f"   Result: {'✅ Success' if result3['success'] else '❌ Failed'}")
    if result3['success']:
        print(f"   Created: {result3['objects_created']}")
    else:
        print(f"   Error: {result3['message']}")
    
    print("\n📊 Final Scene State:")
    if hasattr(interpreter, 'scene') and interpreter.scene.objects:
        for obj in interpreter.scene.objects:
            print(f"   {obj.object_id}: {obj.name}")
            # Handle both dict and object attribute access for position
            if hasattr(obj.position, 'x'):
                print(f"     Position: ({obj.position.x:.1f}, {obj.position.y:.1f}, {obj.position.z:.1f})")
            else:
                print(f"     Position: ({obj.position['x']:.1f}, {obj.position['y']:.1f}, {obj.position['z']:.1f})")
            # Handle both dict and object attribute access for scale    
            if hasattr(obj.scale, 'x'):
                print(f"     Scale: ({obj.scale.x:.1f}, {obj.scale.y:.1f}, {obj.scale.z:.1f})")
            else:
                print(f"     Scale: ({obj.scale['x']:.1f}, {obj.scale['y']:.1f}, {obj.scale['z']:.1f})")
            # Handle both dict and object attribute access for color
            if hasattr(obj.color, 'red'):
                print(f"     Color: ({obj.color.red:.1f}, {obj.color.green:.1f}, {obj.color.blue:.1f})")
            else:
                print(f"     Color: ({obj.color['red']:.1f}, {obj.color['green']:.1f}, {obj.color['blue']:.1f})")
    else:
        print("   No objects found")
    
    print("\n✅ Full demo test completed!")

if __name__ == "__main__":
    test_full_demo()
