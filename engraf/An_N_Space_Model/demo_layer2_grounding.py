#!/usr/bin/env python3
"""
LATN Layer 2 Semantic Grounding Demo

This demo shows how Layer 2 semantic grounding works to resolve noun phrases
to actual objects in a 3D scene. Unlike the tokenization demo which just forms
grammatical structures, this demo shows how those structures map to real objects.

Key concepts demonstrated:
1. Scene object creation and management
2. Noun phrase resolution to scene objects
3. Handling ambiguous references
4. Multiple object resolution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder
from engraf.lexer.vector_space import vector_from_features
from engraf.pos.noun_phrase import NounPhrase
from engraf.utils.debug import set_debug


def setup_demo_scene():
    """Create a demo scene with various objects for grounding demonstrations."""
    scene = SceneModel()
    
    # Add some objects with different properties using vector_from_features
    red_box_vector = vector_from_features("noun", red=1.0)
    red_box = SceneObject("box", red_box_vector, object_id="red_box_1")
    scene.add_object(red_box)
    
    blue_sphere_vector = vector_from_features("noun", blue=1.0)
    blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere_1")
    scene.add_object(blue_sphere)
    
    large_green_cube_vector = vector_from_features("noun", green=1.0, scaleX=1.5, scaleY=1.5, scaleZ=1.5)
    large_green_cube = SceneObject("cube", large_green_cube_vector, object_id="green_cube_1")
    scene.add_object(large_green_cube)
    
    small_red_sphere_vector = vector_from_features("noun", red=1.0, scaleX=0.5, scaleY=0.5, scaleZ=0.5)
    small_red_sphere = SceneObject("sphere", small_red_sphere_vector, object_id="red_sphere_1")
    scene.add_object(small_red_sphere)
    
    # Add another red object to test ambiguity
    red_cube_vector = vector_from_features("noun", red=1.0)
    red_cube = SceneObject("cube", red_cube_vector, object_id="red_cube_1")
    scene.add_object(red_cube)
    
    return scene


def demo_specific_object_grounding():
    """Demonstrate grounding of specific, unambiguous noun phrases."""
    print("=== Specific Object Grounding ===")
    print("Testing noun phrases that should resolve to single, specific objects")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "the blue sphere",
        "the large green cube", 
        "the small red sphere"
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Get Layer 2 tokenization with semantic grounding enabled
        result = executor.execute_layer2(phrase, enable_semantic_grounding=True)
        if not result.success:
            print(f"  ❌ Tokenization failed: {result.description}")
            continue
            
        # Show the tokenization result - let the __repr__ methods handle grounding display
        print(f"  Results: {len(result.hypotheses)} hypothesis(es)")
        for i, hypothesis in enumerate(result.hypotheses):
            print(f"    Hypothesis {i+1} (confidence: {hypothesis.confidence:.3f}):")
            print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
            # Print all tokens using the new method
            hypothesis.print_tokens()

            #for j, token in enumerate(hypothesis.tokens):
            #    if token.isa("NP"):
            #        print(f"        [{j}] {token._original_np if hasattr(token, '_original_np') else 'No original NP'}")
        print()


def demo_ambiguous_grounding():
    """Demonstrate handling of ambiguous noun phrases that match multiple objects."""
    print("=== Ambiguous Object Grounding ===")
    print("Testing noun phrases that could match multiple objects")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "the red object",  # Should match red_box, red_sphere, red_cube
        "the sphere",      # Should match blue_sphere, red_sphere  
        "the cube"         # Should match green_cube, red_cube
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Get Layer 2 tokenization with semantic grounding enabled and return_all_matches=True
        result = executor.execute_layer2(phrase, enable_semantic_grounding=True, return_all_matches=True)
        if not result.success:
            print(f"  ❌ Tokenization failed: {result.description}")
            continue
            
        # Show the tokenization result - should have multiple hypotheses for ambiguous phrases
        print(f"  Results: {len(result.hypotheses)} hypothesis(es)")
        for i, hypothesis in enumerate(result.hypotheses):
            print(f"    Hypothesis {i+1} (confidence: {hypothesis.confidence:.3f}):")
            print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
            # Print all tokens using the new method that shows grounding info
            hypothesis.print_tokens()
        print()


def demo_no_match_grounding():
    """Demonstrate handling when noun phrases don't match any scene objects."""
    print("=== No Match Grounding ===")
    print("Testing noun phrases that don't match any objects in the scene")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "the yellow sphere",  # No yellow objects
        "the purple cube",    # No purple objects
        "the tiny object"     # No tiny objects
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Get Layer 2 tokenization with semantic grounding enabled
        result = executor.execute_layer2(phrase, enable_semantic_grounding=True)
        if not result.success:
            print(f"  ❌ Tokenization failed: {result.description}")
            continue
            
        # Show the tokenization result
        print(f"  Results: {len(result.hypotheses)} hypothesis(es)")
        for i, hypothesis in enumerate(result.hypotheses):
            print(f"    Hypothesis {i+1} (confidence: {hypothesis.confidence:.3f}):")
            print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
            # Print all tokens using the new method that shows grounding info
            hypothesis.print_tokens()
        print()


def demo_complex_grounding():
    """Demonstrate grounding of complex noun phrases with multiple adjectives."""
    print("=== Complex Noun Phrase Grounding ===")
    print("Testing complex noun phrases with multiple adjectives")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "the large green object",
        "the small red sphere", 
        "a red cube"
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Get Layer 2 tokenization with semantic grounding enabled
        result = executor.execute_layer2(phrase, enable_semantic_grounding=True)
        if not result.success:
            print(f"  ❌ Tokenization failed: {result.description}")
            continue
            
        # Show the tokenization result
        print(f"  Results: {len(result.hypotheses)} hypothesis(es)")
        for i, hypothesis in enumerate(result.hypotheses):
            print(f"    Hypothesis {i+1} (confidence: {hypothesis.confidence:.3f}):")
            print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
            # Print all tokens using the new method that shows grounding info
            hypothesis.print_tokens()
        print()


def main():
    """Run all Layer 2 grounding demonstrations."""
    # Suppress debug output for clean demo
    set_debug(False)
    
    print("LATN Layer 2 Semantic Grounding Demo")
    print("=====================================")
    print()
    print("This demo shows how Layer 2 semantic grounding resolves noun phrases")
    print("from grammatical structures to actual objects in a 3D scene.")
    print()
    
    # Run the demonstrations
    demo_specific_object_grounding()
    demo_ambiguous_grounding()
    demo_no_match_grounding()
    demo_complex_grounding()
    
    print("Demo complete! Layer 2 grounding bridges grammatical analysis")
    print("with semantic understanding of the visual scene.")


if __name__ == "__main__":
    main()
