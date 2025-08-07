#!/usr/bin/env python3

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_pronoun_resolution():
    print("🔧 Testing pronoun resolution")
    print("=" * 40)
    
    # Create interpreter with mock renderer
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    scene = interpreter.scene
    
    # 1. Create a cube
    print("1. Creating a red cube...")
    result1 = interpreter.interpret("draw a red cube at [0, 0, 0]")
    
    if result1['success']:
        cube_id = result1['objects_created'][0]
        print(f"   Created cube: {cube_id}")
        print(f"   Last acted object: {interpreter.last_acted_object}")
        
        # Check cube's initial state
        cube = None
        for obj in scene.objects:
            if obj.object_id == cube_id:
                cube = obj
                break
        print(f"   Initial scale: X={cube.vector['scaleX']}, Y={cube.vector['scaleY']}, Z={cube.vector['scaleZ']}")
    else:
        print(f"   Failed to create cube: {result1['message']}")
        return
    
    # 2. Test "make it bigger"
    print("\n2. Testing pronoun resolution for 'make it bigger'...")
    sentence = "make it bigger"
    
    # Parse the sentence and examine the structure
    result2 = interpreter.interpret(sentence)
    
    print(f"   Parsing result: {result2['success']}")
    if result2['success']:
        print(f"   Objects modified: {result2['objects_modified']}")
        parsed = result2['sentence_parsed']
        
        # Examine the noun phrase structure
        np = parsed.predicate.noun_phrase
        print(f"   Noun phrase: {np}")
        print(f"   NP.noun: {np.noun}")
        print(f"   NP.pronoun: {np.pronoun}")
        print(f"   NP.vector['pronoun']: {np.vector['pronoun']}")
        
        # Check if the object resolver finds the target
        from engraf.interpreter.handlers.object_resolver import ObjectResolver
        resolver = ObjectResolver(scene, interpreter._last_acted_object)
        target_objects = resolver.resolve_target_objects(parsed.predicate)
        print(f"   Target objects found: {target_objects}")
        
        # Check final cube scale
        cube = None
        for obj in scene.objects:
            if obj.object_id == cube_id:
                cube = obj
                break
        print(f"   Final scale: X={cube.vector['scaleX']}, Y={cube.vector['scaleY']}, Z={cube.vector['scaleZ']}")
        
        if cube.vector['scaleX'] > 1.0:
            print("   ✅ Scale increased!")
        else:
            print("   ❌ Scale did not increase")
    else:
        print(f"   Failed to parse: {result2['message']}")

if __name__ == "__main__":
    test_pronoun_resolution()
