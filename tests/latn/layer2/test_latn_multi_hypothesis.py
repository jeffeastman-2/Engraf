#!/usr/bin/env python3
"""
Test LATN Layer 2 Multi-Hypothesis Generation with Confidence Scores

This test demonstrates how LATN generates multiple NP resolution hypotheses
when queries are underspecified, with confidence scores based on semantic similarity.
"""

from engraf.lexer.vector_space import vector_from_features
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.latn_tokenizer import latn_tokenize_layer1
from engraf.atn.subnet_np import run_np


def test_latn_multi_hypothesis_generation():
    """Test LATN generating multiple NP hypotheses with confidence scores."""
    
    print("=== LATN Multi-Hypothesis Generation Test ===")
    
    # Create scene with multiple spheres having different attributes
    scene = SceneModel()
    
    # Create spheres with different colors and sizes
    red_sphere_vector = vector_from_features('noun', red=1.0)
    red_sphere = SceneObject("sphere", red_sphere_vector, object_id="red_sphere_1")
    
    green_sphere_vector = vector_from_features('noun', green=1.0)
    green_sphere = SceneObject("sphere", green_sphere_vector, object_id="green_sphere_1") 
    
    blue_sphere_vector = vector_from_features('noun', blue=1.0)
    blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere_1")
    
    # Add two tall spheres
    tall_red_sphere_vector = vector_from_features('noun', red=1.0, scaleY=2.0)
    tall_red_sphere = SceneObject("sphere", tall_red_sphere_vector, object_id="tall_red_sphere_1")
    
    tall_blue_sphere_vector = vector_from_features('noun', blue=1.0, scaleY=2.0)
    tall_blue_sphere = SceneObject("sphere", tall_blue_sphere_vector, object_id="tall_blue_sphere_1")
    
    # Add objects to scene
    scene.add_object(red_sphere)
    scene.add_object(green_sphere) 
    scene.add_object(blue_sphere)
    scene.add_object(tall_red_sphere)
    scene.add_object(tall_blue_sphere)
    
    print(f"Created scene with {len(scene.objects)} spheres:")
    for obj in scene.objects:
        print(f"  {obj.object_id}: {obj.vector}")
    
    # Test Case 1: "the sphere" should match all 5 spheres with high confidence
    print(f"\n=== Test Case 1: 'the sphere' (should match all 5) ===")
    multi_hypothesis_query(scene, "the sphere", expected_count=5)
    
    # Test Case 2: "the tall sphere" should match 2 tall spheres  
    print(f"\n=== Test Case 2: 'the tall sphere' (should match 2 tall ones) ===")
    multi_hypothesis_query(scene, "the tall sphere", expected_count=2)
    
    # Test Case 3: "the blue sphere" should match 2 blue spheres
    print(f"\n=== Test Case 3: 'the blue sphere' (should match 2 blue ones) ===")
    multi_hypothesis_query(scene, "the blue sphere", expected_count=2)
    
    # Test Case 4: "the tall blue sphere" should match 1 specific sphere
    print(f"\n=== Test Case 4: 'the tall blue sphere' (should match 1 specific) ===")
    multi_hypothesis_query(scene, "the tall blue sphere", expected_count=1)


def multi_hypothesis_query(scene, query, expected_count):
    """Test a query and show all matching hypotheses with confidence scores."""
    
    # Parse the query into an NP
    hypotheses = latn_tokenize_layer1(query)
    tokens = hypotheses[0].tokens  # Use best hypothesis
    token_stream = TokenStream(tokens)
    
    np = run_np(token_stream)
    
    if not np:
        print(f"❌ Failed to parse query: {query}")
        return
        
    print(f"Query: '{query}'")
    print(f"Parsed NP: {np}")
    print(f"NP vector: {np.vector}")
    
    # Get all matching candidates with confidence scores
    candidates = scene.find_noun_phrase(np, return_all_matches=True)
    
    print(f"\nFound {len(candidates)} matching candidates:")
    for i, (confidence, obj) in enumerate(candidates, 1):
        print(f"  Hypothesis {i}: {obj.object_id} (confidence: {confidence:.3f})")
        print(f"    Object vector: {obj.vector}")
    
    # Verify expected count
    if len(candidates) == expected_count:
        print(f"✅ Correct: Found {len(candidates)} matches (expected {expected_count})")
    else:
        print(f"❌ Error: Found {len(candidates)} matches, expected {expected_count}")
    
    return candidates


