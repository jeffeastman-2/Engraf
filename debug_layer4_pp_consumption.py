#!/usr/bin/env python3
"""
Debug Layer 4 PP Token Consumption Issue

This script demonstrates the issue where Layer 4 grounding doesn't properly 
consume PP tokens like Layer 3 does.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from engraf.utils.debug import set_debug


def debug_pp_consumption():
    """Compare Layer 3 vs Layer 4 PP token consumption behavior."""
    print("=== Layer 4 PP Token Consumption Debug ===")
    print("Comparing Layer 3 vs Layer 4 PP token consumption behavior")
    print()
    
    # Create a simple scene
    scene = SceneModel()
    
    # Create VectorSpace objects for scene objects
    cube_vector = VectorSpace("red cube")
    cube_vector.update({
        'cube': 1.0, 'red': 1.0, 'noun': 1.0,
        'locX': 0.0, 'locY': 1.0, 'locZ': 0.0
    })
    scene.add_object(SceneObject("cube", cube_vector, "red_cube_1"))
    
    sphere_vector = VectorSpace("blue sphere")  
    sphere_vector.update({
        'sphere': 1.0, 'blue': 1.0, 'noun': 1.0,
        'locX': 2.0, 'locY': 1.0, 'locZ': 0.0
    })
    scene.add_object(SceneObject("sphere", sphere_vector, "blue_sphere_1"))
    
    table_vector = VectorSpace("table")
    table_vector.update({
        'table': 1.0, 'brown': 1.0, 'noun': 1.0,
        'locX': 0.0, 'locY': 0.0, 'locZ': 0.0
    })
    scene.add_object(SceneObject("table", table_vector, "table_1"))
    
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "move the cube to [5, 0, 0]",
        "create a red sphere at [1, 2, 3]", 
        "place the cylinder behind the table"
    ]
    
    for phrase in test_phrases:
        print(f"Testing phrase: '{phrase}'")
        print()
        
        # Layer 3 grounding (reference behavior)
        print("Layer 3 grounding (expected PP consumption):")
        layer3_result = executor.execute_layer3(phrase, enable_semantic_grounding=True)
        
        if layer3_result.success and layer3_result.hypotheses:
            best_l3 = layer3_result.hypotheses[0]
            print(f"  Tokens: {[tok.word for tok in best_l3.tokens]}")
            
            # Show token details
            for i, token in enumerate(best_l3.tokens):
                token_info = f"    [{i}] '{token.word}' ({type(token).__name__})"
                
                if token.isa("NP"):
                    if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                        token_info += " → GNP"
                    else:
                        token_info += " → ungrounded NP"
                elif token.isa("PP"):
                    if hasattr(token, '_original_pp') and token._original_pp:
                        token_info += " → GPP"
                    else:
                        token_info += " → ungrounded PP"
                elif token.isa("verb"):
                    token_info += " → verb"
                else:
                    token_info += " → other"
                
                print(token_info)
        else:
            print(f"  ❌ Layer 3 failed: {layer3_result.description}")
        
        print()
        
        # Layer 4 processing (issue behavior)  
        print("Layer 4 processing (PP consumption issue):")
        layer4_result = executor.execute_layer4(phrase)
        
        if layer4_result.success and layer4_result.hypotheses:
            best_l4 = layer4_result.hypotheses[0]
            print(f"  Tokens: {[tok.word for tok in best_l4.tokens]}")
            
            # Show token details
            for i, token in enumerate(best_l4.tokens):
                token_info = f"    [{i}] '{token.word}' ({type(token).__name__})"
                
                if token.isa("VP"):
                    token_info += " → VP"
                    if hasattr(token, '_original_vp') and token._original_vp:
                        vp = token._original_vp
                        if hasattr(vp, 'noun_phrase') and vp.noun_phrase:
                            token_info += f" (with NP: {vp.noun_phrase.head_noun if hasattr(vp.noun_phrase, 'head_noun') else 'unknown'})"
                        if hasattr(vp, 'preps') and vp.preps:
                            token_info += f" (with {len(vp.preps)} PPs)"
                elif token.isa("NP"):
                    if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                        token_info += " → GNP"
                    else:
                        token_info += " → ungrounded NP"
                elif token.isa("PP"):
                    if hasattr(token, '_original_pp') and token._original_pp:
                        token_info += " → GPP"
                    else:
                        token_info += " → ungrounded PP"
                elif token.isa("verb"):
                    token_info += " → verb"
                else:
                    token_info += " → other"
                
                print(token_info)
                
            # Check if PP tokens are still present after VP formation
            pp_tokens = [tok for tok in best_l4.tokens if tok.isa("PP")]
            if pp_tokens:
                print(f"  ⚠️  Issue: {len(pp_tokens)} PP tokens still present after VP formation")
                for pp_tok in pp_tokens:
                    print(f"      - '{pp_tok.word}' should have been consumed into VP")
            else:
                print(f"  ✓ No remaining PP tokens (good)")
        else:
            print(f"  ❌ Layer 4 failed: {layer4_result.description}")
        
        print("-" * 60)
        print()


if __name__ == "__main__":
    set_debug(False)
    debug_pp_consumption()
