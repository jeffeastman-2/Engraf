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
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features
from engraf.utils.debug import set_debug


def setup_demo_scene():
    """Create a demo scene with positioned objects for spatial relationships."""
    scene = SceneModel()
    
    # Create objects with specific positions for spatial testing
    # Base table
    table_vector = vector_from_features("noun", locX=0.0, locY=-1.0, locZ=0.0, scaleX=2.0, scaleY=0.2, scaleZ=2.0)
    table = SceneObject("table", table_vector, object_id="table_1")
    scene.add_object(table)
    
    # Red cube on the table
    red_cube_vector = vector_from_features("noun", red=1.0, locX=0.0, locY=0.0, locZ=0.0)
    red_cube = SceneObject("cube", red_cube_vector, object_id="red_cube_1")
    scene.add_object(red_cube)
    
    # Blue sphere to the right of the cube
    blue_sphere_vector = vector_from_features("noun", blue=1.0, locX=2.0, locY=0.0, locZ=0.0)
    blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere_1")
    scene.add_object(blue_sphere)
    
    # Green cylinder behind the cube
    green_cylinder_vector = vector_from_features("noun", green=1.0, locX=0.0, locY=0.0, locZ=-2.0)
    green_cylinder = SceneObject("cylinder", green_cylinder_vector, object_id="green_cylinder_1")
    scene.add_object(green_cylinder)
    
    return scene


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
            
        # Show the final tokenization result
        if result.hypotheses:
            hypothesis = result.hypotheses[0]
            print(f"  Final tokens: {[tok.word for tok in hypothesis.tokens]}")
            
            # Check for different token types
            np_tokens = [tok for tok in hypothesis.tokens if tok.isa("NP")]
            pp_tokens = [tok for tok in hypothesis.tokens if tok.isa("PP")]
            
            if np_tokens:
                print(f"  → Noun Phrases (NP): {[tok.word for tok in np_tokens]}")
            if pp_tokens:
                print(f"  → Prepositional Phrases (PP): {[tok.word for tok in pp_tokens]}")
                
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
            
        # Show detailed analysis of the result
        if result.hypotheses:
            hypothesis = result.hypotheses[0]
            print(f"  Confidence: {hypothesis.confidence:.3f}")
            print(f"  Token count: {len(hypothesis.tokens)}")
            
            # Group tokens by type
            token_groups = {
                "Verbs": [tok for tok in hypothesis.tokens if tok.isa("verb")],
                "Noun Phrases": [tok for tok in hypothesis.tokens if tok.isa("NP")],
                "Prepositional Phrases": [tok for tok in hypothesis.tokens if tok.isa("PP")],
                "Other": [tok for tok in hypothesis.tokens if not any([tok.isa("verb"), tok.isa("NP"), tok.isa("PP")])]
            }
            
            for group_name, tokens in token_groups.items():
                if tokens:
                    print(f"  {group_name}: {[tok.word for tok in tokens]}")
                    
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
            hypothesis = result.hypotheses[0] if result.hypotheses else None
            if hypothesis:
                np_count = len([tok for tok in hypothesis.tokens if tok.isa("NP")])
                pp_count = len([tok for tok in hypothesis.tokens if tok.isa("PP")])
                
                print(f"  ✓ Processed: {np_count} NPs, {pp_count} PPs")
                print(f"  Tokens: {[tok.word for tok in hypothesis.tokens]}")
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
