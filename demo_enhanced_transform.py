#!/usr/bin/env python3
"""
Enhanced Transform System Demo

This script demonstrates the enhanced object matching and transform system
without using VPython windows (no popups).

Usage:
    python demo_enhanced_transform.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer


def demo_enhanced_transform():
    """Demonstrate the enhanced transform system with object matching."""
    print("🎨 ENGRAF Enhanced Transform System Demo")
    print("=" * 50)
    print()
    
    # Create interpreter with mock renderer (no popups!)
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Step 1: Create multiple objects
    print("📦 Step 1: Creating objects...")
    creation_commands = [
        "draw a red cube",
        "draw a blue sphere", 
        "draw a small green cube",
        "draw a large yellow sphere",
        "draw a tiny purple cube"
    ]
    
    for cmd in creation_commands:
        result = interpreter.interpret(cmd)
        if result['success']:
            print(f"   ✅ {cmd} → {result['objects_created']}")
        else:
            print(f"   ❌ {cmd} → {result['message']}")
    
    print()
    
    # Step 2: Show created objects
    print("📊 Step 2: Current scene objects:")
    for obj in interpreter.scene.objects:
        print(f"   {obj.object_id}: {obj.name}")
        print(f"     Colors: R={obj.vector['red']:.1f}, G={obj.vector['green']:.1f}, B={obj.vector['blue']:.1f}")
        print(f"     Scale: {obj.vector['scaleX']:.1f}")
    
    print()
    
    # Step 3: Test enhanced object matching with transforms
    print("🎯 Step 3: Testing enhanced object matching...")
    transform_commands = [
        "move the red cube to [5, 0, 0]",
        "move the blue sphere to [0, 5, 0]", 
        "move the small green cube to [-5, 0, 0]",
        "move the large yellow sphere to [0, 0, 5]",
        "move the tiny purple cube to [0, -5, 0]"
    ]
    
    for cmd in transform_commands:
        result = interpreter.interpret(cmd)
        if result['success']:
            print(f"   ✅ {cmd} → Modified: {result['objects_modified']}")
        else:
            print(f"   ❌ {cmd} → {result['message']}")
    
    print()
    
    # Step 4: Test vector distance calculation
    print("🔍 Step 4: Vector distance analysis...")
    
    # Test finding red cube
    from engraf.lexer.token_stream import TokenStream, tokenize
    from engraf.atn.subnet_sentence import run_sentence
    
    query_phrases = ["the red cube", "the blue sphere", "the small green cube"]
    
    for phrase in query_phrases:
        print(f"\\nQuery: '{phrase}'")
        
        # Parse the phrase
        tokens = tokenize(phrase)
        ts = TokenStream(tokens)
        parsed = run_sentence(ts)
        
        if parsed and parsed.predicate and parsed.predicate.noun_phrase:
            query_np = parsed.predicate.noun_phrase
            
            # Find objects by description
            found_objects = interpreter._find_objects_by_description(query_np)
            
            print(f"   Found {len(found_objects)} matching objects:")
            for obj in found_objects:
                distance = interpreter._calculate_vector_distance(obj.vector, query_np.vector)
                matches = interpreter._object_matches_description(obj, query_np)
                print(f"     {obj.object_id}: distance={distance:.3f}, matches={matches}")
    
    print()
    
    # Step 5: Test complex scenarios
    print("🧪 Step 5: Complex scenario testing...")
    
    # Test with ambiguous queries
    complex_commands = [
        "move the cube to [10, 0, 0]",  # Should find best matching cube
        "move the sphere to [0, 10, 0]",  # Should find best matching sphere
        "move the small cube to [0, 0, 10]"  # Should find the small green cube
    ]
    
    for cmd in complex_commands:
        result = interpreter.interpret(cmd)
        if result['success']:
            print(f"   ✅ {cmd} → Modified: {result['objects_modified']}")
        else:
            print(f"   ❌ {cmd} → {result['message']}")
    
    print()
    print("✅ Enhanced Transform System Demo Complete!")
    print()
    
    # Final summary
    summary = interpreter.get_scene_summary()
    print("📊 Final Scene Summary:")
    print(f"   Total objects: {summary['total_objects']}")
    print(f"   Object types: {summary['object_types']}")
    print(f"   Object IDs: {summary['object_ids']}")
    print()


def test_vector_distance_algorithm():
    """Test the vector distance calculation algorithm directly."""
    print("🧮 Testing Vector Distance Algorithm")
    print("=" * 40)
    
    from engraf.lexer.vector_space import VectorSpace
    from engraf.interpreter.sentence_interpreter import SentenceInterpreter
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create test vectors
    red_cube = VectorSpace()
    red_cube['red'] = 1.0
    red_cube['noun'] = 1.0
    red_cube['adj'] = 1.0
    
    blue_sphere = VectorSpace()
    blue_sphere['blue'] = 1.0
    blue_sphere['noun'] = 1.0
    blue_sphere['adj'] = 1.0
    
    green_cube = VectorSpace()
    green_cube['green'] = 1.0
    green_cube['noun'] = 1.0
    green_cube['adj'] = 1.0
    green_cube['scaleX'] = 0.5  # small
    green_cube['scaleY'] = 0.5
    green_cube['scaleZ'] = 0.5
    
    # Test query vector
    query_red_cube = VectorSpace()
    query_red_cube['red'] = 1.0
    query_red_cube['noun'] = 1.0
    query_red_cube['adj'] = 1.0
    
    objects = [
        ("red_cube", red_cube),
        ("blue_sphere", blue_sphere), 
        ("green_small_cube", green_cube)
    ]
    
    print("Query: red cube")
    print("Objects and their distances:")
    
    for name, obj_vector in objects:
        distance = interpreter._calculate_vector_distance(obj_vector, query_red_cube)
        print(f"   {name}: {distance:.3f}")
    
    print()


if __name__ == "__main__":
    try:
        demo_enhanced_transform()
        test_vector_distance_algorithm()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
