#!/usr/bin/env python3
"""
Test the full demo scaling issue - checking if "make it bigger" works in the full pipeline
"""

import os
import sys

# Add the project root to the Python path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_full_scaling_pipeline():
    """Test the exact sequence from the demo to see if scaling works."""
    print("🧪 Testing full scaling pipeline...")
    print("=" * 50)
    
    # Use MockRenderer to avoid display issues
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # The exact sequence from the demo
    demo_sentences = [
        "draw a red cube at [0, 0, 0]",
        "move it to [2, 3, 4]", 
        "make it bigger"
    ]
    
    cube_obj = None
    
    for i, sentence in enumerate(demo_sentences, 1):
        print(f"\n{i}. Processing: '{sentence}'")
        
        try:
            result = interpreter.interpret(sentence)
            
            print(f"   Result: {result}")
            
            if result['success']:
                print(f"   ✅ Success: {result['message']}")
                if result['objects_created']:
                    print(f"   📦 Created: {', '.join(result['objects_created'])}")
                    # Get the cube object for inspection
                    if 'cube' in result['objects_created'][0]:
                        cube_obj = result['objects_created'][0]
                if result['objects_modified']:
                    print(f"   🔧 Modified: {', '.join(result['objects_modified'])}")
                if result['actions_performed']:
                    print(f"   🎬 Actions: {', '.join(result['actions_performed'])}")
            else:
                print(f"   ❌ Failed: {result['message']}")
                
            # Check cube properties after each step
            if cube_obj:
                scene_objects = renderer.get_scene_objects()
                if cube_obj in scene_objects:
                    obj = scene_objects[cube_obj]
                    print(f"   📊 Cube '{cube_obj}' state:")
                    print(f"       Position: X={obj.position[0]}, Y={obj.position[1]}, Z={obj.position[2]}")
                    print(f"       Scale: X={obj.scale[0]}, Y={obj.scale[1]}, Z={obj.scale[2]}")
                    print(f"       Color: R={obj.color[0]}, G={obj.color[1]}, B={obj.color[2]}")
                else:
                    print(f"   ⚠️  Cube '{cube_obj}' not found in scene objects")
            
        except Exception as e:
            print(f"   💥 Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📋 Final Scene Summary:")
    scene_objects = renderer.get_scene_objects()
    print(f"   Total objects: {len(scene_objects)}")
    for obj_id, obj in scene_objects.items():
        print(f"   • {obj_id}: pos={obj.position}, scale={obj.scale}, color={obj.color}")

if __name__ == "__main__":
    test_full_scaling_pipeline()
