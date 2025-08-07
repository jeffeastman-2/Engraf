#!/usr/bin/env python3
"""
Debug object resolution issue.
"""
import sys
import os
sys.path.append('/Users/jeff/Python/Engraf')

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import MockVPythonRenderer
from engraf.utils.debug import set_debug

def debug_object_resolution():
    set_debug(True)
    renderer = MockVPythonRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    
    print("=== Creating objects ===")
    result1 = interpreter.interpret('draw a red cube at [0, 0, 0]')
    result2 = interpreter.interpret('draw a big blue sphere at [3, 0, 0]')
    
    print(f"\nResult1: {result1}")
    print(f"Result2: {result2}")
    
    print(f"\n=== Scene state ===")
    print(f"Number of objects: {len(interpreter.scene.objects)}")
    for i, obj in enumerate(interpreter.scene.objects):
        print(f"Object {i}: id='{obj.object_id}', name='{obj.name}'")
        print(f"  Vector noun value: {obj.vector['noun']}")
        print(f"  Vector red value: {obj.vector['red']}")
        print(f"  Vector blue value: {obj.vector['blue']}")
    
    print(f"\n=== Testing object resolution ===")
    # Parse the command "move the sphere above the cube"
    from engraf.atn.sentence import build_sentence_atn
    from engraf.atn.core import run_atn
    from engraf.lexer.token_stream import tokenize, TokenStream
    from engraf.pos.sentence_phrase import SentencePhrase
    
    sentence = 'move the sphere above the cube'
    tokens = tokenize(sentence)
    token_stream = TokenStream(tokens)
    sent = SentencePhrase()
    start, end = build_sentence_atn(sent, token_stream)
    parsed_sentence = run_atn(start, end, token_stream, sent)
    
    print(f"Parsed sentence: {parsed_sentence}")
    print(f"Verb phrase: {parsed_sentence.predicate}")
    print(f"Noun phrase: {parsed_sentence.predicate.noun_phrase}")
    print(f"Noun phrase noun: {parsed_sentence.predicate.noun_phrase.noun}")
    
    # Test object resolver directly
    vp = parsed_sentence.predicate
    target_objects = interpreter.object_resolver.resolve_target_objects(vp)
    print(f"\nTarget objects found: {target_objects}")
    
    # Test find_objects_by_description directly
    np = vp.noun_phrase
    print(f"\nTesting find_objects_by_description with noun phrase: {np}")
    print(f"  np.noun: {np.noun}")
    print(f"  np.determiner: {np.determiner}")
    print(f"  np.vector: {np.vector}")
    
    objects = interpreter.object_resolver.find_objects_by_description(np)
    print(f"Objects found by description: {objects}")
    
    # Test _object_matches_description for each object
    print(f"\n=== Testing individual object matches ===")
    for obj in interpreter.scene.objects:
        matches = interpreter.object_resolver._object_matches_description(obj, np)
        print(f"Object '{obj.object_id}' (name='{obj.name}') matches: {matches}")
        print(f"  Noun check: np.noun='{np.noun}' == obj.name='{obj.name}' -> {np.noun == obj.name}")
        
        # If noun check passes, let's see what happens with vector distance
        if np.noun == obj.name:
            print(f"  → Noun check passed, checking vector distance...")
            print(f"  → Object vector: {obj.vector}")
            print(f"  → Query vector: {np.vector}")
            print(f"  → Has vector: {hasattr(np, 'vector') and np.vector is not None}")
            
            if hasattr(np, 'vector') and np.vector:
                distance = interpreter.object_resolver._calculate_vector_distance(obj.vector, np.vector)
                print(f"  → Vector distance: {distance}")
                print(f"  → Threshold: 0.5")
                print(f"  → Distance <= threshold: {distance <= 0.5}")
            else:
                print(f"  → No vector, would return True")

if __name__ == "__main__":
    debug_object_resolution()
