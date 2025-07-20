#!/usr/bin/env python3

import sys
sys.path.append('/Users/jeff/Python/Engraf')

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import MockVPythonRenderer

def test_transform_visual_updates():
    """Test that transforms update the visual representation."""
    print("🧪 Testing Transform Visual Updates")
    print("=" * 50)
    
    # Create interpreter with mock renderer
    interpreter = SentenceInterpreter()
    interpreter.renderer = MockVPythonRenderer()
    
    # Test 1: Create an object
    print("\n1. Creating object...")
    result = interpreter.interpret("draw a red cube")
    print(f"   Result: {result}")
    
    # Check if object was created
    if result['objects_created']:
        obj_id = result['objects_created'][0]
        print(f"   Created object: {obj_id}")
        
        # Get initial position
        scene_obj = None
        for obj in interpreter.scene.objects:
            if obj.object_id == obj_id:
                scene_obj = obj
                break
        
        if scene_obj:
            initial_pos = [scene_obj.vector['locX'], scene_obj.vector['locY'], scene_obj.vector['locZ']]
            print(f"   Initial position: {initial_pos}")
            
            # Test 2: Move the object
            print("\n2. Moving object...")
            result = interpreter.interpret("move it to [5, 0, 0]")
            print(f"   Result: {result}")
            
            # Check if position changed
            new_pos = [scene_obj.vector['locX'], scene_obj.vector['locY'], scene_obj.vector['locZ']]
            print(f"   New position: {new_pos}")
            
            if new_pos != initial_pos:
                print("   ✅ Position updated successfully!")
            else:
                print("   ❌ Position did not change")
            
            # Test 3: Scale the object
            print("\n3. Scaling object...")
            initial_scale = [scene_obj.vector['scaleX'], scene_obj.vector['scaleY'], scene_obj.vector['scaleZ']]
            print(f"   Initial scale: {initial_scale}")
            
            result = interpreter.interpret("scale the cube by [2, 2, 2]")
            print(f"   Result: {result}")
            
            new_scale = [scene_obj.vector['scaleX'], scene_obj.vector['scaleY'], scene_obj.vector['scaleZ']]
            print(f"   New scale: {new_scale}")
            
            if new_scale != initial_scale:
                print("   ✅ Scale updated successfully!")
            else:
                print("   ❌ Scale did not change")
            
            # Test 4: Rotate the object
            print("\n4. Rotating object...")
            initial_rot = [scene_obj.vector['rotX'], scene_obj.vector['rotY'], scene_obj.vector['rotZ']]
            print(f"   Initial rotation: {initial_rot}")
            
            result = interpreter.interpret("rotate the cube by 45 degrees")
            print(f"   Result: {result}")
            
            new_rot = [scene_obj.vector['rotX'], scene_obj.vector['rotY'], scene_obj.vector['rotZ']]
            print(f"   New rotation: {new_rot}")
            
            if new_rot != initial_rot:
                print("   ✅ Rotation updated successfully!")
            else:
                print("   ❌ Rotation did not change")
        else:
            print("   ❌ Could not find created object")
    else:
        print("   ❌ No object was created")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_transform_visual_updates()
