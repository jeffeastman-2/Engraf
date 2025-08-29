#!/usr/bin/env python3

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene

scene = setup_demo_scene()
executor = LATNLayerExecutor(scene)

phrase = "the cube above the table"
result = executor.execute_layer3(phrase, enable_semantic_grounding=True)

if result.success and result.hypotheses:
    target_token = result.hypotheses[0].tokens[0]
    print("target_token type:", type(target_token))
    print("target_token.vector type:", type(target_token.vector))
    print("Has len:", hasattr(target_token.vector, "__len__"))
    
    # Check if it's a VectorSpace with nested vector
    if hasattr(target_token.vector, "vector"):
        print("target_token.vector.vector type:", type(target_token.vector.vector))
        print("target_token.vector.vector len:", len(target_token.vector.vector))
    
    # Check if target_token.vector itself has len
    try:
        print("target_token.vector len:", len(target_token.vector))
    except Exception as e:
        print("Error getting len(target_token.vector):", e)
        
    # Check the structure
    print("target_token attributes:")
    for attr in dir(target_token):
        if not attr.startswith('__'):
            print(f"  {attr}: {type(getattr(target_token, attr))}")
