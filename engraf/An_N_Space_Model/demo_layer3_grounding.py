#!/usr/bin/env python3
"""
LATN Layer 3 Semantic Grounding Demo

This demo showcases Layer 3 semantic grounding capabilities, which handle:
1. Spatial relationship validation between objects
2. Prepositional phrase grounding to spatial locations
3. PP attachment resolution for complex spatial chains
4. Vector coordinate grounding (e.g., "to [3,4,5]")

Layer 3 grounding builds on Layer 2 grounded NounPhrases and validates
spatial relationships to create semantically meaningful PP tokens.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info
from engraf.utils.debug import set_debug


def demo_spatial_validation():
    """Demonstrate spatial validation in Layer 3 grounding."""
    print("=== Layer 3 Spatial Validation Demo ===")
    print("Testing how Layer 3 validates spatial relationships between objects")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    # Test cases with spatial relationships that should validate differently
    test_cases = [
        {
            "phrase": "the cube above the table",
            "expected": "Valid (cube Y=1 > table Y=0)",
            "description": "Cube is positioned above the table"
        },
        {
            "phrase": "the sphere right of the cube", 
            "expected": "Valid (sphere X=2 > cube X=0)",
            "description": "Sphere is positioned to the right of the cube"
        },
        {
            "phrase": "the cylinder behind the table",
            "expected": "Valid (cylinder Z=-2 < table Z=0)",
            "description": "Cylinder is positioned behind the table"
        },
        {
            "phrase": "the table above the cube",
            "expected": "Invalid (table Y=0 < cube Y=1)",
            "description": "Table is actually below the cube - should fail validation"
        },
        {
            "phrase": "the cube left of the sphere",
            "expected": "Valid (cube X=0 < sphere X=2)",
            "description": "Cube is positioned to the left of the sphere"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: '{case['phrase']}'")
        print(f"  Context: {case['description']}")
        print(f"  Expected: {case['expected']}")
        
        # First show Layer 3 tokenization (Layer 2 grounding always enabled)
        print(f"  Tokenization (L3 grounding=False):")
        token_result = executor.execute_layer3(case['phrase'], enable_semantic_grounding=False)
        if token_result.success and token_result.hypotheses:
            token_hypothesis = token_result.hypotheses[0]
            print(f"    Tokens: {[tok.word for tok in token_hypothesis.tokens]}")
            pp_tokens = [tok.word for tok in token_hypothesis.tokens if tok.isa("PP")]
            np_tokens = [tok.word for tok in token_hypothesis.tokens if tok.isa("NP")]
            if pp_tokens:
                print(f"    PP tokens: {pp_tokens}")
            if np_tokens:
                print(f"    NP tokens: {np_tokens}")
        else:
            print(f"    ❌ Tokenization failed")
        
        # Then show Layer 3 with full grounding enabled  
        print(f"  Grounding (L3 grounding=True):")
        result = executor.execute_layer3(case['phrase'], enable_semantic_grounding=True)
        
        if result.success and result.hypotheses:
            print(f"  Results: {len(result.hypotheses)} hypothesis(es)")
            
            for h_idx, hypothesis in enumerate(result.hypotheses):
                print(f"    Hypothesis {h_idx + 1} (confidence: {hypothesis.confidence:.3f}):")
                print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
                
                # Show detailed token analysis
                for i, token in enumerate(hypothesis.tokens):
                    token_info = f"        [{i}] '{token.word}' ({type(token).__name__})"
                    
                    # Check token type and grounding
                    if token.isa("NP"):
                        if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                            if hasattr(token._grounded_phrase, 'grounding') and token._grounded_phrase.grounding:
                                obj = token._grounded_phrase.grounding['scene_object']
                                conf = token._grounded_phrase.grounding['confidence']
                                token_info += f" → GROUNDED to {obj.object_id} (conf: {conf:.2f})"
                            else:
                                token_info += " → grounded phrase (no scene object)"
                        else:
                            token_info += " → ungrounded NP"
                    elif token.isa("PP"):
                        token_info += " → PP token"
                        if hasattr(token, '_original_pp') and token._original_pp:
                            pp = token._original_pp
                            if hasattr(pp, 'preposition'):
                                token_info += f" (prep: {pp.preposition})"
                    elif token.isa("verb"):
                        token_info += " → verb"
                    else:
                        token_info += " → other"
                    
                    # Show spatial chain if present
                    if hasattr(token, '_spatial_chain') and token._spatial_chain:
                        token_info += f" [spatial: {token._spatial_chain}]"
                        
                    print(token_info)
                
                # Show hypothesis metadata
                if hasattr(hypothesis, 'description'):
                    print(f"      Description: {hypothesis.description}")
                
                if h_idx < len(result.hypotheses) - 1:  # Add separator except for last
                    print()
                    
        else:
            print(f"  ❌ Processing failed: {result.description}")
            
        print()


def demo_vector_coordinate_grounding():
    """Demonstrate grounding of vector coordinates like 'to [3,4,5]'."""
    print("=== Layer 3 Vector Coordinate Grounding Demo ===")
    print("Testing how Layer 3 grounds vector coordinates and absolute positions")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "move the cube to [5, 0, 0]",
        "place the sphere at [1, 2, 3]", 
        "position the cylinder to [-1, -1, -1]",
        "put the table at the origin [0, 0, 0]"
    ]
    
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}'")
        
        # Execute with grounding enabled
        result = executor.execute_layer3(phrase, enable_semantic_grounding=True)
        
        if result.success and result.hypotheses:
            print(f"  Results: {len(result.hypotheses)} hypothesis(es)")
            
            for h_idx, hypothesis in enumerate(result.hypotheses):
                print(f"    Hypothesis {h_idx + 1} (confidence: {hypothesis.confidence:.3f}):")
                print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
                
                # Detailed token analysis
                for i, token in enumerate(hypothesis.tokens):
                    token_info = f"        [{i}] '{token.word}'"
                    
                    if token.isa("PP"):
                        token_info += " (PP)"
                        if hasattr(token, '_original_pp') and token._original_pp:
                            pp = token._original_pp
                            if hasattr(pp, 'vector') and pp.vector:
                                coords = []
                                for dim in ['locX', 'locY', 'locZ']:
                                    if dim in pp.vector and pp.vector[dim] != 0.0:
                                        coords.append(f"{dim}={pp.vector[dim]}")
                                if coords:
                                    token_info += f" [{', '.join(coords)}]"
                    elif token.isa("NP"):
                        token_info += " (NP)"
                        if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                            token_info += " - grounded"
                    elif token.isa("verb"):
                        token_info += " (verb)"
                    else:
                        token_info += " (other)"
                    
                    print(token_info)
                
                if h_idx < len(result.hypotheses) - 1:
                    print()
        else:
            print(f"  ❌ Processing failed: {result.description}")
            
        print()


def demo_complex_spatial_chains():
    """Demonstrate grounding of complex spatial relationship chains."""
    print("=== Layer 3 Complex Spatial Chain Grounding Demo ===")
    print("Testing complex sentences with multiple spatial relationships")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    test_phrases = [
        "move the cube above the table right of the sphere",
        "place the sphere below the cube behind the cylinder", 
        "position the cylinder left of the table above the sphere",
        "put the cube above the table below the sphere to [2, 2, 2]"
    ]
    
    for phrase in test_phrases:
        print(f"Complex phrase: '{phrase}'")
        
        # Show Layer 2 results first (what gets passed to Layer 3)
        layer2_result = executor.execute_layer2(phrase, enable_semantic_grounding=True)
        if layer2_result.success and layer2_result.hypotheses:
            layer2_hypothesis = layer2_result.hypotheses[0]
            grounded_nps = [tok for tok in layer2_hypothesis.tokens 
                          if tok.isa("NP") and hasattr(tok, '_grounded_phrase') and tok._grounded_phrase]
            
            print(f"  Layer 2 input: {len(grounded_nps)} grounded NPs")
            for np_token in grounded_nps:
                if hasattr(np_token._grounded_phrase, 'grounding') and np_token._grounded_phrase.grounding:
                    obj = np_token._grounded_phrase.grounding['scene_object']
                    print(f"    {np_token.word} → {obj.object_id}")
        
        # Execute Layer 3 grounding
        result = executor.execute_layer3(phrase, enable_semantic_grounding=True)
        
        if result.success and result.hypotheses:
            print(f"  Layer 3 output: {len(result.hypotheses)} hypothesis(es)")
            
            for h_idx, hypothesis in enumerate(result.hypotheses):
                print(f"    Hypothesis {h_idx + 1}:")
                print(f"      Confidence: {hypothesis.confidence:.3f}")
                print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
                
                # Show detailed token breakdown
                spatial_tokens = []
                grounded_tokens = []
                pp_tokens = []
                
                for token in hypothesis.tokens:
                    if hasattr(token, '_spatial_chain') and token._spatial_chain:
                        spatial_tokens.append(f"{token.word} → {token._spatial_chain}")
                    if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                        grounded_tokens.append(token.word)
                    if token.isa("PP"):
                        pp_tokens.append(token.word)
                
                if grounded_tokens:
                    print(f"      Grounded tokens: {grounded_tokens}")
                if pp_tokens:
                    print(f"      PP tokens: {pp_tokens}")
                if spatial_tokens:
                    print(f"      Spatial attachments: {spatial_tokens}")
                else:
                    print(f"      No spatial attachments formed")
                
                if h_idx < len(result.hypotheses) - 1:
                    print()
                
        else:
            print(f"  ❌ Layer 3 processing failed: {result.description}")
            
        print()


def demo_grounding_edge_cases():
    """Demonstrate how Layer 3 grounding handles edge cases and failures."""
    print("=== Layer 3 Grounding Edge Cases Demo ===")
    print("Testing how grounding handles ambiguous or invalid spatial relationships")
    print()
    
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene_model=scene)
    
    edge_cases = [
        {
            "phrase": "the pyramid above the table",
            "issue": "Object doesn't exist in scene",
            "expected": "NP grounding should fail"
        },
        {
            "phrase": "the cube next to itself",
            "issue": "Self-referential spatial relationship", 
            "expected": "Spatial validation should reject"
        },
        {
            "phrase": "the table inside the cube",
            "issue": "Physically impossible relationship",
            "expected": "Low spatial validation score"
        },
        {
            "phrase": "move everything to [999, 999, 999]",
            "issue": "Pronoun resolution + extreme coordinates",
            "expected": "Should handle pronoun and extreme vector"
        }
    ]
    
    for i, case in enumerate(edge_cases, 1):
        print(f"Edge Case {i}: '{case['phrase']}'")
        print(f"  Issue: {case['issue']}")
        print(f"  Expected: {case['expected']}")
        
        result = executor.execute_layer3(case['phrase'], enable_semantic_grounding=True)
        
        if result.success and result.hypotheses:
            print(f"  ✓ Processed: {len(result.hypotheses)} hypothesis(es)")
            
            for h_idx, hypothesis in enumerate(result.hypotheses):
                print(f"    Hypothesis {h_idx + 1} (confidence: {hypothesis.confidence:.3f}):")
                print(f"      Tokens: {[tok.word for tok in hypothesis.tokens]}")
                
                # Analyze grounding success/failure
                grounded_tokens = []
                ungrounded_nps = []
                pp_tokens = []
                
                for token in hypothesis.tokens:
                    if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                        grounded_tokens.append(token.word)
                    elif token.isa("NP"):
                        ungrounded_nps.append(token.word)
                    elif token.isa("PP"):
                        pp_tokens.append(token.word)
                
                print(f"      Grounded: {len(grounded_tokens)} tokens {grounded_tokens if grounded_tokens else ''}")
                print(f"      Ungrounded NPs: {len(ungrounded_nps)} tokens {ungrounded_nps if ungrounded_nps else ''}")
                if pp_tokens:
                    print(f"      PP tokens: {pp_tokens}")
                
                if h_idx < len(result.hypotheses) - 1:
                    print()
                
        else:
            print(f"  ❌ Processing failed: {result.description}")
            
        print()


def main():
    """Run all Layer 3 grounding demonstrations."""
    set_debug(False)
    
    print("LATN Layer 3 Semantic Grounding Demo")
    print("====================================")
    print()
    print("This demo showcases Layer 3's semantic grounding capabilities:")
    print("- Spatial relationship validation between scene objects")
    print("- Vector coordinate grounding for absolute positioning") 
    print("- Complex spatial chain resolution and attachment")
    print("- Edge case handling for ambiguous or invalid relationships")
    print()
    
    # Show scene setup info
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    # Run all demonstrations
    demo_spatial_validation()
    demo_vector_coordinate_grounding()
    demo_complex_spatial_chains()
    demo_grounding_edge_cases()
    
    print("Demo complete! Layer 3 grounding enables sophisticated spatial reasoning")
    print("and validation of prepositional phrase relationships in 3D scenes.")


if __name__ == "__main__":
    main()
