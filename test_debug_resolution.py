#!/usr/bin/env python3
"""
Direct test of object resolution logic without full parsing.
"""

import sys
import os
sys.path.append('/Users/jeff/Python/Engraf')

from engraf.visualizer.renderers.mock_renderer import MockRenderer
from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.noun_phrase import NounPhrase
from engraf.visualizer.scene.scene_object import SceneObject

def test_object_resolution():
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    
    print("=== CREATING SCENE OBJECTS DIRECTLY ===")
    
    # Create objects directly to avoid parsing output
    red_cube_vector = VectorSpace()
    red_cube_vector.word = "red cube"
    red_cube_vector["red"] = 1.0
    red_cube_vector["noun"] = 1.0
    
    red_cube = SceneObject(
        object_id="red_cube",
        name="cube", 
        vector=red_cube_vector
    )
    
    blue_sphere_vector = VectorSpace()
    blue_sphere_vector.word = "blue sphere" 
    blue_sphere_vector["blue"] = 1.0
    blue_sphere_vector["noun"] = 1.0
    
    blue_sphere = SceneObject(
        object_id="blue_sphere",
        name="sphere",
        vector=blue_sphere_vector
    )
    
    green_cylinder_vector = VectorSpace()
    green_cylinder_vector.word = "green cylinder"
    green_cylinder_vector["green"] = 1.0
    green_cylinder_vector["noun"] = 1.0
    
    green_cylinder = SceneObject(
        object_id="green_cylinder", 
        name="cylinder",
        vector=green_cylinder_vector
    )
    
    interpreter.scene.objects = [red_cube, blue_sphere, green_cylinder]
    
    print(f"Created objects: {[obj.object_id for obj in interpreter.scene.objects]}")
    
    # Test noun phrase for "the red cube"
    test_np = NounPhrase(noun="cube")
    test_np.determiner = "the"
    test_np.vector = VectorSpace()
    test_np.vector.word = "red cube"
    test_np.vector["red"] = 1.0
    test_np.vector["noun"] = 1.0
    test_np.vector["det"] = 1.0
    test_np.vector["def"] = 1.0
    
    print(f"\nTest NP: noun={test_np.noun}, red={test_np.vector['red']}")
    
    # Test individual object matching
    print("\n=== TESTING OBJECT MATCHING ===")
    for obj in interpreter.scene.objects:
        matches = interpreter._object_matches_description(obj, test_np)
        print(f"{obj.object_id}: matches={matches} (obj red={obj.vector['red']}, name={obj.name})")
    
    # Test find_objects_by_description
    print("\n=== TESTING FIND OBJECTS ===")
    found_objects = interpreter._find_objects_by_description(test_np)
    print(f"Found objects: {found_objects}")
    
    # Set the last acted object to test pronoun resolution
    print("\n=== TESTING PRONOUN RESOLUTION ===")
    interpreter.last_acted_object = "red_cube"
    print(f"Last acted object set to: {interpreter.last_acted_object}")
    
    # Test pronoun NP for "it"
    pronoun_np = NounPhrase()
    pronoun_np.pronoun = "it"
    from engraf.pos.verb_phrase import VerbPhrase
    test_vp = VerbPhrase(verb="move")
    test_vp.noun_phrase = pronoun_np
    
    resolved_objects = interpreter._resolve_target_objects(test_vp)
    print(f"'it' resolves to: {resolved_objects}")

if __name__ == "__main__":
    test_object_resolution()
