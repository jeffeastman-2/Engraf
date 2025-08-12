#!/usr/bin/env python3
"""
Simple test for cylinder rotation - focuses just on the core issue
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer

def test_simple_rotation():
    print("🔧 Simple Rotation Test")
    print("=" * 30)
    
    # Create renderer in headless mode to avoid hanging
    renderer = VPythonRenderer(width=800, height=600, title="Test", headless=True)
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # Test 1: Create green cylinder
    print("📍 Creating green cylinder...")
    result1 = interpreter.interpret("draw a green cylinder at [0, 0, 0]")
    print(f"Create result: success={result1.get('success')}, message='{result1.get('message')}'")
    
    if not result1['success']:
        print("❌ Failed to create cylinder, stopping test")
        return
    
    # Test 2: Rotate the cylinder
    print("🔄 Rotating the cylinder...")
    result2 = interpreter.interpret("rotate it by 90 degrees")
    print(f"Rotate result: success={result2.get('success')}, message='{result2.get('message')}'")
    
    if result2['success']:
        print("✅ Rotation command completed successfully!")
        
        # Check if we can access the scene to verify rotation
        if hasattr(interpreter.scene, 'objects') and interpreter.scene.objects:
            for obj in interpreter.scene.objects:
                print(f"Object {obj.object_id}: has_rotation={obj.has_rotation()}")
                if obj.has_rotation():
                    print(f"  Rotation: {obj.rotation}")
                    
        # Final detailed check - verify rotation state
        final_obj = None
        for obj in interpreter.scene.objects:
            if obj.object_id == "green_small_cylinder_1":
                final_obj = obj
                break
                
        if final_obj:
            print(f"\n🔍 Final verification:")
            print(f"Object {final_obj.object_id}: has_rotation={final_obj.has_rotation()}")
            print(f"SceneObject rotation: {final_obj.rotation}")
            print(f"Vector rotX: {final_obj.vector['rotX']}")
            print(f"Vector rotY: {final_obj.vector['rotY']}")
            print(f"Vector rotZ: {final_obj.vector['rotZ']}")
            
            # Manually call update_transformations and check again
            print(f"\n🔧 Manual update check:")
            print(f"Before update: rotation={final_obj.rotation}")
            
            # Check vector space directly
            print(f"Vector space check:")
            print(f"  'rotX' in vector: {'rotX' in final_obj.vector}")
            print(f"  'rotY' in vector: {'rotY' in final_obj.vector}")
            print(f"  'rotZ' in vector: {'rotZ' in final_obj.vector}")
            
            # Try manual access
            try:
                print(f"  Manual rotZ access: {final_obj.vector['rotZ']}")
            except Exception as e:
                print(f"  Error accessing rotZ: {e}")
            
            final_obj.update_transformations()
            print(f"After update: rotation={final_obj.rotation}")
        else:
            print("❌ Object not found for final verification")
    else:
        print("❌ Rotation failed")
    
    print("✅ Test completed")

if __name__ == "__main__":
    test_simple_rotation()
