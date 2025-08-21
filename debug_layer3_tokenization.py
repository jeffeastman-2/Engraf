#!/usr/bin/env python3
"""
Debug script to test Layer 3 (PP) tokenization processing.

This script shows how Layer 3 processes Layer 2 output and forms PP tokens.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.latn_tokenizer_layer3 import latn_tokenize_layer3
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vocabulary_builder import vector_from_word

def print_separator(title):
    print("=" * 80)
    print(f"🔬 {title}")
    print("=" * 80)

def debug_layer3():
    print("🔍 DEBUGGING LAYER 3 TOKENIZATION")
    print_separator("DEBUGGING LAYER 3 TOKENIZATION")
    
    sentence = "move the box above the table right of the pyramid under the sphere to [3,4,5]"
    print(f"Sentence: {sentence}")
    print()
    
    # Create scene with positioned objects using SceneModel
    scene = SceneModel()
    
    # Create scene objects with distinct vectors
    box_vector = vector_from_word("box")
    table_vector = vector_from_word("table")
    pyramid_vector = vector_from_word("pyramid")
    sphere_vector = vector_from_word("sphere")
    
    # Add scene objects at specific positions
    box = SceneObject("box", box_vector, object_id="box-1")
    box.position = [1, 2, 3]
    scene.add_object(box)
    
    table = SceneObject("table", table_vector, object_id="table-1")
    table.position = [5, 6, 7]
    scene.add_object(table)
    
    pyramid = SceneObject("pyramid", pyramid_vector, object_id="pyramid-1")
    pyramid.position = [8, 9, 10]
    scene.add_object(pyramid)
    
    sphere = SceneObject("sphere", sphere_vector, object_id="sphere-1")
    sphere.position = [11, 12, 13]
    scene.add_object(sphere)
    
    # Create executor
    executor = LATNLayerExecutor(scene)
    
    # Execute Layer 2 to get the NP tokenization results
    print_separator("LAYER 2 EXECUTION")
    layer2_result = executor.execute_layer2(sentence, enable_semantic_grounding=True)
    
    print(f"  Success: {layer2_result.success}")
    print(f"  Confidence: {layer2_result.confidence}")
    print(f"  Description: {layer2_result.description}")
    print(f"  Hypotheses: {len(layer2_result.hypotheses) if hasattr(layer2_result, 'hypotheses') else 0}")
    print()
    
    if not layer2_result.success or not hasattr(layer2_result, 'hypotheses'):
        print("❌ Layer 2 failed - stopping")
        return
    
    # Show Layer 2 hypotheses briefly
    for i, hyp in enumerate(layer2_result.hypotheses[:2]):  # Show first 2
        print(f"  L2 HYPOTHESIS [{i+1}] (confidence: {hyp.confidence}):")
        print(f"    Tokens ({len(hyp.tokens)}):")
        for j, token in enumerate(hyp.tokens):
            print(f"      [{j}] '{token.word}' ({type(token).__name__})")
        print()
    
    # Now invoke Layer 3 tokenizer directly on the Layer 2 hypotheses
    print_separator("LAYER 3 TOKENIZER DIRECT CALL")
    layer3_hypotheses = latn_tokenize_layer3(layer2_result.hypotheses)
    
    print(f"  Success: {bool(layer3_hypotheses)}")
    print(f"  Hypotheses: {len(layer3_hypotheses)}")
    print()
    
    if not layer3_hypotheses:
        print("❌ Layer 3 tokenizer failed - stopping")
        return
    
    # Show detailed Layer 3 output
    for i, hyp in enumerate(layer3_hypotheses):
        print(f"  L3 HYPOTHESIS [{i+1}] (confidence: {hyp.confidence}):")
        if hasattr(hyp, 'description'):
            print(f"    Description: {hyp.description}")
        if hasattr(hyp, 'pp_replacements'):
            print(f"    PP Replacements: {len(hyp.pp_replacements)}")
        print(f"    Tokens ({len(hyp.tokens)}):")
        for j, token in enumerate(hyp.tokens):
            if hasattr(token, '_original_pp') and token._original_pp:
                pp = token._original_pp
                print(f"      [{j}] '{token.word}' (PP token) -> {pp.preposition} + {pp.noun_phrase}")
            elif hasattr(token, '_original_np') and token._original_np:
                np = token._original_np
                print(f"      [{j}] '{token.word}' (NP token) -> {np}")
            else:
                print(f"      [{j}] '{token.word}' (VectorSpace)")
        print()
        
        # Show PP replacements details
        if hasattr(hyp, 'pp_replacements') and hyp.pp_replacements:
            print(f"    PP REPLACEMENT DETAILS:")
            for start, end, pp in hyp.pp_replacements:
                if pp:
                    print(f"      [{start}:{end}] -> PP({pp.preposition} + {pp.noun_phrase})")
                else:
                    print(f"      [{start}:{end}] -> PP(None)")
        print()
        print(f"  Hypotheses: {len(layer3_hypotheses)}")
        
        # Show detailed Layer 3 output
        for i, hyp in enumerate(layer3_hypotheses):
            print(f"  L3 HYPOTHESIS [{i+1}] (confidence: {hyp.confidence}):")
            if hasattr(hyp, 'description'):
                print(f"    Description: {hyp.description}")
            if hasattr(hyp, 'pp_replacements'):
                print(f"    PP Replacements: {len(hyp.pp_replacements)}")
            print(f"    Tokens ({len(hyp.tokens)}):")
            for j, token in enumerate(hyp.tokens):
                if hasattr(token, '_original_pp') and token._original_pp:
                    pp = token._original_pp
                    print(f"      [{j}] '{token.word}' (PP token) -> {pp.preposition} + {pp.noun_phrase}")
                elif hasattr(token, '_original_np') and token._original_np:
                    np = token._original_np
                    print(f"      [{j}] '{token.word}' (NP token) -> {np}")
                else:
                    print(f"      [{j}] '{token.word}' (VectorSpace)")
            print()
            
            # Show PP replacements details
            if hasattr(hyp, 'pp_replacements') and hyp.pp_replacements:
                print(f"    PP REPLACEMENT DETAILS:")
                for start, end, pp in hyp.pp_replacements:
                    if pp:
                        print(f"      [{start}:{end}] -> PP({pp.preposition} + {pp.noun_phrase})")
                    else:
                        print(f"      [{start}:{end}] -> PP(None)")
            print()
    
    print_separator("ANALYSIS")
    print("Expected PP formations:")
    print("  1. 'above' + 'NP(the table)' -> 'PP(above the table)'")
    print("  2. 'right of' + 'NP(the pyramid)' -> 'PP(right of the pyramid)'") 
    print("  3. 'under' + 'NP(the sphere)' -> 'PP(under the sphere)'")
    print("  4. 'to' + 'NP([3,4,5])' -> 'PP(to [3,4,5])'")
    print()
    
    if layer3_hypotheses:
        best_hyp = layer3_hypotheses[0]
        pp_count = len(best_hyp.pp_replacements) if hasattr(best_hyp, 'pp_replacements') else 0
        print(f"🎯 RESULT: {pp_count} PP(s) formed in best hypothesis")
        
        if pp_count < 4:
            print("❌ ISSUE: Expected 4 PPs but found fewer")
            print("   Check if PP ATN is parsing all preposition + NP combinations")
        else:
            print("✅ SUCCESS: All expected PPs formed")

if __name__ == "__main__":
    debug_layer3()
