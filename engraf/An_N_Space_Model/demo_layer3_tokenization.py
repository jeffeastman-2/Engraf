#!/usr/bin/env python3
"""
LATN Layer 3 Tokenization Demo

This demo shows the complete LATN pipeline through Layer 3, demonstrating:
1. Layer 1: Lexical tokenization
2. Layer 2: NP tokenization with semantic grounding
3. Layer 3: PP tokenization for spatial relationships

The demo focuses on the final results rather than intermediate processing steps.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene
from engraf.utils.debug import set_debug


def demo_layer3_spatial_relationships():
    """Demonstrate Layer 3 processing of spatial prepositional phrases."""
    print("=== Layer 3 Spatial Relationship Processing ===")
    print("Testing prepositional phrases that reference scene objects")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "the cube above the table",
        "the sphere right of the cube", 
        "the cylinder behind the cube",
        "move the cube below the sphere",
        "place a box above the table"
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Execute full Layer 3 pipeline
        result = executor.execute_layer3(phrase, enable_semantic_grounding=False)
        if not result.success:
            print(f"  ❌ Processing failed: {result.description}")
            continue
            
        # Show all tokenization hypotheses
        if result.hypotheses:
            print(f"  Generated {len(result.hypotheses)} hypothesis(es):")
            for i, hypothesis in enumerate(result.hypotheses):
                print(f"    Hypothesis {i+1} (confidence: {hypothesis.confidence:.3f}):")
                print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
                
                # Check for different token types
                np_tokens = [tok for tok in hypothesis.tokens if tok.isa("NP")]
                pp_tokens = [tok for tok in hypothesis.tokens if tok.isa("PP")]
                verb_tokens = [tok for tok in hypothesis.tokens if tok.isa("verb")]
                other_tokens = [tok for tok in hypothesis.tokens if not any([tok.isa("NP"), tok.isa("PP"), tok.isa("verb")])]
                
                if verb_tokens:
                    print(f"      → Verbs: {[tok.word for tok in verb_tokens]}")
                if np_tokens:
                    print(f"      → Noun Phrases (NP): {[tok.word for tok in np_tokens]}")
                if pp_tokens:
                    print(f"      → Prepositional Phrases (PP): {[tok.word for tok in pp_tokens]}")
                if other_tokens:
                    print(f"      → Other tokens: {[tok.word for tok in other_tokens]}")
                print(f"      Description: {hypothesis.description}")
                
        print()


def demo_layer3_complex_structures():
    """Demonstrate Layer 3 processing of complex grammatical structures."""
    print("=== Layer 3 Complex Structure Processing ===")
    print("Testing complex sentences with multiple spatial relationships")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "draw a red cube above the table",
        "move the cube below the table",
        "create a sphere above the cube behind the table"
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Execute Layer 3 tokenization only
        result = executor.execute_layer3(phrase, enable_semantic_grounding=False)
        if not result.success:
            print(f"  ❌ Processing failed: {result.description}")
            continue
            
        # Show detailed analysis of all hypotheses
        if result.hypotheses:
            print(f"  Generated {len(result.hypotheses)} hypothesis(es):")
            for i, hypothesis in enumerate(result.hypotheses):
                print(f"    Hypothesis {i+1}:")
                print(f"      Confidence: {hypothesis.confidence:.3f}")
                print(f"      Token count: {len(hypothesis.tokens)}")
                print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
                
                # Group tokens by type
                token_groups = {
                    "Verbs": [tok for tok in hypothesis.tokens if tok.isa("verb")],
                    "Noun Phrases": [tok for tok in hypothesis.tokens if tok.isa("NP")],
                    "Prepositional Phrases": [tok for tok in hypothesis.tokens if tok.isa("PP")],
                    "Other": [tok for tok in hypothesis.tokens if not any([tok.isa("verb"), tok.isa("NP"), tok.isa("PP")])]
                }
                
                for group_name, tokens in token_groups.items():
                    if tokens:
                        print(f"      {group_name}: {[tok.word for tok in tokens]}")
                
                print(f"      Description: {hypothesis.description}")
                print()
                    
        print()


def demo_layer3_grounding_failures():
    """Demonstrate how Layer 3 handles phrases that don't ground properly."""
    print("=== Layer 3 Grounding Edge Cases ===")
    print("Testing phrases with non-existent objects or invalid spatial relationships")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "the pyramid above the table",      # pyramid doesn't exist
        "the cube above the sphere",        # valid spatial relationship but might be invalid spatially
        "the table above the cube"          # reversed spatial relationship
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Execute Layer 3 tokenization only
        result = executor.execute_layer3(phrase, enable_semantic_grounding=False)
        if not result.success:
            print(f"  ❌ Processing failed: {result.description}")
        else:
            if result.hypotheses:
                print(f"  Generated {len(result.hypotheses)} hypothesis(es):")
                for i, hypothesis in enumerate(result.hypotheses):
                    np_count = len([tok for tok in hypothesis.tokens if tok.isa("NP")])
                    pp_count = len([tok for tok in hypothesis.tokens if tok.isa("PP")])
                    verb_count = len([tok for tok in hypothesis.tokens if tok.isa("verb")])
                    
                    print(f"    Hypothesis {i+1} (confidence: {hypothesis.confidence:.3f}):")
                    print(f"      {verb_count} Verbs, {np_count} NPs, {pp_count} PPs")
                    print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
                    print(f"      Description: {hypothesis.description}")
                    print()
            else:
                print(f"  ❌ No hypotheses generated")
                
        print()


def main():
    """Run all Layer 3 tokenization demonstrations."""
    set_debug(False)
    
    print("LATN Layer 3 Tokenization Demo")
    print("==============================")
    print()
    print("This demo shows the complete LATN pipeline through Layer 3,")
    print("demonstrating how spatial prepositional phrases are processed")
    print("and tokenized as PP tokens with spatial relationship information.")
    print()
    
    # Run the demonstrations
    demo_layer3_spatial_relationships()
    demo_layer3_complex_structures()
    demo_layer3_grounding_failures()
    
    print("Demo complete! Layer 3 processing enables spatial reasoning")
    print("about relationships between grounded scene objects.")


if __name__ == "__main__":
    main()
