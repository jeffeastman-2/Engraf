#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/jeff/Python/Engraf')

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer
from engraf.interpreter.handlers.object_resolver import ObjectResolver
from engraf.interpreter.handlers.object_modifier import ObjectModifier

def test_modifier_issue():
    print("🔧 Testing modifier issue for 'make it bigger'")
    print("=" * 50)
    
    # Create interpreter with mock renderer
    mock_renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=mock_renderer)
    scene = interpreter.scene
    
    # Create a cube first 
    print("1. Creating a red cube...")
    result1 = interpreter.interpret("draw a red cube at [0, 0, 0]")
    print(f"   Result: {result1}")
    
    if result1['success']:
        created_objects = result1.get('objects_created', [])
        if created_objects:
            cube_id = created_objects[0]
            print(f"   Created cube: {cube_id}")
        else:
            cube_id = None
            print("   No objects created!")
            
        print(f"   Last acted object: {interpreter._last_acted_object}")
        
        # Find the cube object
        cube = None
        for obj in scene.objects:
            if hasattr(obj, 'object_id') and obj.object_id == cube_id:
                cube = obj
                break
                
        if cube:
            print(f"   Initial scale: X={cube.vector['scaleX']}, Y={cube.vector['scaleY']}, Z={cube.vector['scaleZ']}")
        else:
            print(f"   ❌ Could not find cube object with ID: {cube_id}")
            print(f"   Scene objects: {[obj.object_id for obj in scene.objects if hasattr(obj, 'object_id')]}")
            return
        
        print("\n2. Testing 'make it bigger' command step by step...")
        
        # Parse the command
        result2 = interpreter.interpret("make it bigger")
        print(f"   Parse result: {result2}")
        
        if result2['success']:
            sentence = result2['sentence_parsed']
            print(f"   Parsed sentence: {sentence}")
            print(f"   Verb: {sentence.predicate.verb}")
            print(f"   Adjective complement: {sentence.predicate.adjective_complement}")
            print(f"   Noun phrase pronoun: {sentence.predicate.noun_phrase.pronoun}")
            
            # Test ObjectResolver
            print("\n3. Testing ObjectResolver...")
            resolver = interpreter.object_resolver
            target_objects = resolver.resolve_target_objects(sentence.predicate)
            print(f"   Target objects: {target_objects}")  # These are object IDs, not objects
            
            # Test ObjectModifier directly
            print("\n4. Testing ObjectModifier...")
            modifier = interpreter.object_modifier
            
            # Check what the modifier thinks about this sentence
            from engraf.lexer.vocabulary_builder import get_from_vocabulary
            verb_vector = get_from_vocabulary(sentence.predicate.verb)
            if verb_vector:
                print(f"   Verb '{sentence.predicate.verb}' vector:")
                print(f"     scale: {verb_vector['scale']}")
                print(f"     create: {verb_vector['create']}")
                print(f"     style: {verb_vector['style']}")
            else:
                print(f"   ❌ No vector found for verb '{sentence.predicate.verb}'")
            
            # Check if VerbPhrase has vector
            print(f"\n4.5. VerbPhrase debug...")
            vp = sentence.predicate
            print(f"   hasattr(vp, 'vector'): {hasattr(vp, 'vector')}")
            if hasattr(vp, 'vector'):
                print(f"   vp.vector type: {type(vp.vector)}")
                print(f"   vp.vector: {vp.vector}")
                if hasattr(vp.vector, '__getitem__'):
                    print(f"   vp.vector['scale']: {vp.vector['scale']}")
            
            # Manually call the modifier
            print("\n5. Calling modifier directly...")
            try:
                for obj_id in target_objects:
                    result = modifier.modify_scene_object(obj_id, sentence.predicate)
                    print(f"   modify_scene_object({obj_id}) returned: {result}")
                print("   Modifier called successfully")
            except Exception as e:
                print(f"   Modifier error: {e}")
                import traceback
                traceback.print_exc()
            
            # Check final scale
            print(f"\n6. Final scale check...")
            final_cube = None
            for obj in scene.objects:
                if obj.object_id == cube_id:
                    final_cube = obj
                    break
                    
            print(f"   Final scale: X={final_cube.vector['scaleX']}, Y={final_cube.vector['scaleY']}, Z={final_cube.vector['scaleZ']}")
            
            if final_cube.vector['scaleX'] > 1.0:
                print("   ✅ Scale increased successfully")
            else:
                print("   ❌ Scale did not increase")
                
        else:
            print("   ❌ Failed to parse 'make it bigger'")
    else:
        print("   ❌ Failed to create cube")

if __name__ == "__main__":
    test_modifier_issue()
