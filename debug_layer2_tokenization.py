#!/usr/bin/env python3
"""Debug script to analyze Layer 2 tokenization before Layer 3 processing."""

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vocabulary_builder import vector_from_word

def debug_layer2_tokenization():
    """Debug what Layer 2 produces for our test sentence."""
    
    # Create the same scene as the test
    scene = SceneModel()
    
    box_vector = vector_from_word("box")
    table_vector = vector_from_word("table")
    pyramid_vector = vector_from_word("pyramid")
    sphere_vector = vector_from_word("sphere")
    
    box = SceneObject("box", box_vector, object_id="box-1")
    table = SceneObject("table", table_vector, object_id="table-1") 
    pyramid = SceneObject("pyramid", pyramid_vector, object_id="pyramid-1")
    sphere = SceneObject("sphere", sphere_vector, object_id="sphere-1")
    
    for obj in [box, table, pyramid, sphere]:
        obj.vector['scaleX'] = 1.0
        obj.vector['scaleY'] = 1.0 
        obj.vector['scaleZ'] = 1.0
    
    scene.add_object(box)
    scene.add_object(table)
    scene.add_object(pyramid) 
    scene.add_object(sphere)
    
    # Position objects
    table.position = [0, 0, 0]
    box.position = [0, 1, 0]
    pyramid.position = [2, 0, 0]
    sphere.position = [0, -2, 0]
    
    table.vector['locX'], table.vector['locY'], table.vector['locZ'] = 0, 0, 0
    box.vector['locX'], box.vector['locY'], box.vector['locZ'] = 0, 1, 0
    pyramid.vector['locX'], pyramid.vector['locY'], pyramid.vector['locZ'] = 2, 0, 0
    sphere.vector['locX'], sphere.vector['locY'], sphere.vector['locZ'] = 0, -2, 0
    
    executor = LATNLayerExecutor(scene)
    sentence = "move the box above the table right of the pyramid under the sphere to [3,4,5]"
    
    print(f"🔍 DEBUGGING LAYER 2 TOKENIZATION")
    print(f"=" * 80)
    print(f"Sentence: {sentence}")
    print()
    
    # Execute Layer 1 first
    print("🔬 LAYER 1 (Tokenization):")
    layer1_result = executor.execute_layer1(sentence)
    print(f"  Success: {layer1_result.success}")
    if hasattr(layer1_result, 'tokens'):
        print(f"  Tokens ({len(layer1_result.tokens)}):")
        for i, token in enumerate(layer1_result.tokens):
            print(f"    [{i}] '{token.word}' -> {token}")
    print()
    
    # Execute Layer 2 
    print("🔬 LAYER 2 (NP/PP Parsing + Grounding):")
    layer2_result = executor.execute_layer2(sentence, enable_semantic_grounding=True)
    print(f"  Success: {layer2_result.success}")
    
    if hasattr(layer2_result, 'hypotheses'):
        print(f"  Hypotheses: {len(layer2_result.hypotheses)}")
        for i, hypothesis in enumerate(layer2_result.hypotheses):
            print(f"\n  HYPOTHESIS [{i+1}] (confidence: {getattr(hypothesis, 'confidence', 'N/A')}):")
            if hasattr(hypothesis, 'tokens'):
                print(f"    Tokens ({len(hypothesis.tokens)}):")
                for j, token in enumerate(hypothesis.tokens):
                    token_type = type(token).__name__
                    
                    # Analyze token content
                    if hasattr(token, 'word'):
                        word = token.word
                    else:
                        word = str(token)[:30]
                    
                    # Check if it's grounded
                    grounding_info = ""
                    if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                        grounding_info = f" -> GROUNDED: {token._grounded_phrase}"
                    elif hasattr(token, '_original_np') and token._original_np:
                        grounding_info = f" -> HAS_NP: {token._original_np}"
                    elif hasattr(token, '_original_pp') and token._original_pp:
                        grounding_info = f" -> HAS_PP: {token._original_pp}"
                    
                    print(f"      [{j}] '{word}' ({token_type}){grounding_info}")
            
            # Check for noun phrases
            if hasattr(hypothesis, 'noun_phrases'):
                print(f"    Noun Phrases ({len(hypothesis.noun_phrases)}):")
                for j, np in enumerate(hypothesis.noun_phrases):
                    np_type = type(np).__name__
                    if hasattr(np, 'scene_object') and np.scene_object:
                        obj_id = np.scene_object.object_id
                        grounding = f" -> GROUNDED to {obj_id}"
                    else:
                        grounding = " -> ungrounded"
                    print(f"      [{j}] {np} ({np_type}){grounding}")
            
            # Check for prepositional phrases  
            if hasattr(hypothesis, 'prepositional_phrases'):
                print(f"    Prepositional Phrases ({len(hypothesis.prepositional_phrases)}):")
                for j, pp in enumerate(hypothesis.prepositional_phrases):
                    pp_type = type(pp).__name__
                    print(f"      [{j}] {pp} ({pp_type})")
    
    print(f"\n🎯 ANALYSIS:")
    print(f"Expected tokens from sentence:")
    print(f"  1. 'move' (verb)")
    print(f"  2. 'the box' (NP -> should ground to box-1)")
    print(f"  3. 'above the table' (PP -> should contain grounded table-1)")
    print(f"  4. 'right of the pyramid' (PP -> should contain grounded pyramid-1)")
    print(f"  5. 'under the sphere' (PP -> should contain grounded sphere-1)")
    print(f"  6. 'to [3,4,5]' (PP -> vector location)")
    print()
    
    # Look for specific issues
    print(f"🔍 ISSUE DETECTION:")
    if hasattr(layer2_result, 'hypotheses') and layer2_result.hypotheses:
        best_hyp = layer2_result.hypotheses[0]
        
        # Check for missing "to [3,4,5]"
        to_vector_found = False
        right_of_found = False
        
        if hasattr(best_hyp, 'tokens'):
            for token in best_hyp.tokens:
                if hasattr(token, 'word'):
                    if 'to' in token.word and '[3,4,5]' in token.word:
                        to_vector_found = True
                    if 'right of' in token.word:
                        right_of_found = True
        
        if not to_vector_found:
            print(f"  ❌ MISSING: 'to [3,4,5]' vector location token")
        else:
            print(f"  ✅ FOUND: 'to [3,4,5]' vector location token")
            
        if not right_of_found:
            print(f"  ❌ MISSING: 'right of the pyramid' multi-word preposition")
        else:
            print(f"  ✅ FOUND: 'right of the pyramid' multi-word preposition")
    
    return layer2_result

if __name__ == "__main__":
    debug_layer2_tokenization()
