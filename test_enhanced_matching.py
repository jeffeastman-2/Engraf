#!/usr/bin/env python3
"""
Test script for enhanced object matching system
"""

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_enhanced_matching():
    print("🧪 Testing Enhanced Object Matching System")
    print("=" * 50)
    
    # Create interpreter with mock renderer (no popups!)
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Test 1: Create objects
    print("\n1. Creating test objects...")
    
    result1 = interpreter.interpret('draw a red cube')
    print(f"   Red cube: {'✅' if result1['success'] else '❌'}")
    if result1['objects_created']:
        print(f"   Created: {result1['objects_created']}")
    
    result2 = interpreter.interpret('draw a blue sphere')
    print(f"   Blue sphere: {'✅' if result2['success'] else '❌'}")
    if result2['objects_created']:
        print(f"   Created: {result2['objects_created']}")
    
    result3 = interpreter.interpret('draw a small green cube')
    print(f"   Green cube: {'✅' if result3['success'] else '❌'}")
    if result3['objects_created']:
        print(f"   Created: {result3['objects_created']}")
    
    # Test 2: Show current objects
    print(f"\n2. Current scene objects ({len(interpreter.scene.objects)}):")
    for obj in interpreter.scene.objects:
        print(f"   {obj.object_id}: {obj.name}")
        print(f"     Vector: red={obj.vector['red']:.2f}, green={obj.vector['green']:.2f}, blue={obj.vector['blue']:.2f}")
    
    # Test 3: Test object matching for transforms
    print("\n3. Testing object matching...")
    
    result4 = interpreter.interpret('move the red cube to [5, 0, 0]')
    print(f"   Move red cube: {'✅' if result4['success'] else '❌'}")
    if result4['objects_modified']:
        print(f"   Modified: {result4['objects_modified']}")
    elif not result4['success']:
        print(f"   Error: {result4.get('message', 'Unknown error')}")
    
    result5 = interpreter.interpret('move the blue sphere to [0, 5, 0]')
    print(f"   Move blue sphere: {'✅' if result5['success'] else '❌'}")
    if result5['objects_modified']:
        print(f"   Modified: {result5['objects_modified']}")
    elif not result5['success']:
        print(f"   Error: {result5.get('message', 'Unknown error')}")
    
    # Test 4: Test vector distance calculation
    print("\n4. Testing vector distance calculation...")
    if len(interpreter.scene.objects) >= 2:
        from engraf.lexer.token_stream import TokenStream, tokenize
        from engraf.atn.subnet_sentence import run_sentence
        
        # Parse "the red cube"
        tokens = tokenize('the red cube')
        ts = TokenStream(tokens)
        parsed = run_sentence(ts)
        
        if parsed and parsed.predicate and parsed.predicate.noun_phrase:
            query_np = parsed.predicate.noun_phrase
            print(f"   Query: '{query_np.noun}' with determiner '{query_np.determiner}'")
            print(f"   Query vector: red={query_np.vector['red']:.2f}")
            
            # Test distance to each object
            for obj in interpreter.scene.objects:
                distance = interpreter._calculate_vector_distance(obj.vector, query_np.vector)
                matches = interpreter._object_matches_description(obj, query_np)
                print(f"   {obj.object_id}: distance={distance:.3f}, matches={matches}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_enhanced_matching()
