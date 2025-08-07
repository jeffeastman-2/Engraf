#!/usr/bin/env python3

"""
Debug Object Modifier - test the object modification path for "make it bigger"
"""

import sys
import os
import time

# Add the engraf package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engraf'))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter

def test_object_modification():
    print("🔧 Testing ObjectModifier path for 'make it bigger'")
    print("=" * 60)
    
    # Create interpreter
    interpreter = SentenceInterpreter()
    
    # First create a cube
    print("1. Creating a red cube...")
    result1 = interpreter.interpret("draw a red cube at [0, 0, 0]")
    print(f"   Result: {result1}")
    
    if result1['success'] and result1['objects_created']:
        cube_id = result1['objects_created'][0]
        print(f"   Created cube with ID: {cube_id}")
        
        # Get initial scale
        scene_obj = None
        for obj in interpreter.scene.objects:
            if obj.object_id == cube_id:
                scene_obj = obj
                break
        
        if scene_obj:
            print(f"   Initial scale: X={scene_obj.vector['scaleX']}, Y={scene_obj.vector['scaleY']}, Z={scene_obj.vector['scaleZ']}")
            
            # Wait 2 seconds before testing modification
            print("\n   Waiting 2 seconds before modification...")
            time.sleep(2)
            
            # Now test the modification
            print("\n2. Making it bigger...")
            print("   Testing object modification path...")
            
            # Parse the sentence
            result2 = interpreter.interpret("make it bigger")
            print(f"   Result: {result2}")
            
            # Check final scale  
            print(f"   Final scale: X={scene_obj.vector['scaleX']}, Y={scene_obj.vector['scaleY']}, Z={scene_obj.vector['scaleZ']}")
            
            if scene_obj.vector['scaleX'] > 1.0:
                print("   ✅ Scale increased successfully!")
            else:
                print("   ❌ Scale did not increase")
                
                # Debug: check if ObjectModifier was called
                print("\n   🔍 Debug information:")
                print(f"     'it' should resolve to: {cube_id}")
                print(f"     Objects modified: {result2.get('objects_modified', [])}")
        else:
            print("   ❌ Could not find scene object")
    else:
        print("   ❌ Failed to create initial cube")

if __name__ == "__main__":
    test_object_modification()
