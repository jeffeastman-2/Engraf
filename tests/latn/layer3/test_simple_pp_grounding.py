#!/usr/bin/env python3
"""
Simple test for Layer 3 grounding with minimal PP attachment.
"""

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


@pytest.fixture
def scene_with_table_and_box():
    """Fixture providing a scene with table and box for spatial testing."""
    scene_model = SceneModel()
    
    # Table at origin
    table_vector = VectorSpace(word=None)
    table_vector['locX'] = 0.0
    table_vector['locY'] = 0.0
    table_vector['locZ'] = 0.0
    table = SceneObject("table", table_vector, object_id="table")
    scene_model.add_object(table)
    
    # Box positioned above table (for "above" relationship)
    box_vector = VectorSpace(word=None)
    box_vector['locX'] = 0.0
    box_vector['locY'] = 1.0
    box_vector['locZ'] = 0.0
    box = SceneObject("box", box_vector, object_id="box")
    scene_model.add_object(box)
    
    return scene_model, table, box


def test_simple_pp_grounding(scene_with_table_and_box):
    """Test Layer 3 PP grounding and Layer 4 verb phrase processing"""
    scene_model, table, box = scene_with_table_and_box
    
    print(f"📍 Object positions:")
    print(f"  Box: [{box.position['x']}, {box.position['y']}, {box.position['z']}] (above table)")
    print(f"  Table: [{table.position['x']}, {table.position['y']}, {table.position['z']}] (reference)")
    
    # Set up executor with the scene
    executor = LATNLayerExecutor(scene_model=scene_model)
    
    # Test sentence with verb to exercise Layer 4
    sentence = "move the box above the table to [5,5,5]"
    print(f"\n🔬 Testing sentence: '{sentence}'")
    
    # Execute Layer 3
    result = executor.execute_layer3(sentence, enable_semantic_grounding=True)
    
    print(f"\n✅ Layer 3 Result:")
    print(f"  Success: {result.success}")
    print(f"  Description: {getattr(result, 'description', 'No description')}")
    print(f"  Hypotheses: {len(result.hypotheses) if hasattr(result, 'hypotheses') else 0}")
    
    # Basic assertions
    assert result.success, "Layer 3 should succeed"
    assert hasattr(result, 'hypotheses'), "Layer 3 should have hypotheses"
    assert len(result.hypotheses) > 0, "Layer 3 should have at least one hypothesis"
    
    # Check if Layer 2 result is available
    if hasattr(result, 'layer2_result'):
        print(f"  Layer 2 Success: {result.layer2_result.success}")
        print(f"  Layer 2 Hypotheses: {len(result.layer2_result.hypotheses) if hasattr(result.layer2_result, 'hypotheses') else 0}")
    
    # Check grounding results  
    if hasattr(result, 'grounding_results'):
        print(f"  Grounding Results: {len(result.grounding_results)}")
        for i, gr in enumerate(result.grounding_results):
            print(f"    [{i}] Success: {getattr(gr, 'success', 'N/A')}, Description: {getattr(gr, 'description', 'N/A')}")
    
    # Check prepositional phrases
    if hasattr(result, 'prepositional_phrases'):
        print(f"  Prepositional Phrases: {len(result.prepositional_phrases)}")
        for i, pp in enumerate(result.prepositional_phrases):
            print(f"    [{i}] {pp}")

    if hasattr(result, 'hypotheses') and result.hypotheses:
        best_hypothesis = result.hypotheses[0]
        
        print(f"\n🔍 Layer 3 Best Hypothesis Analysis:")
        print(f"  Confidence: {getattr(best_hypothesis, 'confidence', 'N/A')}")
        print(f"  Token count: {len(best_hypothesis.tokens) if hasattr(best_hypothesis, 'tokens') else 0}")
        
        if hasattr(best_hypothesis, 'tokens'):
            print(f"  Tokens:")
            for i, token in enumerate(best_hypothesis.tokens):
                token_type = type(token).__name__
                
                # Check for grounded phrases
                has_grounded = hasattr(token, '_grounded_phrase')
                grounded_type = type(token._grounded_phrase).__name__ if has_grounded else None
                
                print(f"    [{i}] {token_type}: {token}")
                if has_grounded:
                    print(f"        _grounded_phrase: {grounded_type} - {token._grounded_phrase}")
                
                # Check for spatial relationships in SceneObjectPhrases
                if hasattr(token, 'spatial_relationships'):
                    print(f"        spatial_relationships: {len(token.spatial_relationships)}")
                    for j, rel in enumerate(token.spatial_relationships):
                        print(f"          [{j}] {rel}")

    # Return the result for use in integrated tests
    return result


def test_layer4_verb_phrase_processing(scene_with_table_and_box):
    """Test Layer 4 verb phrase processing separately."""
    scene_model, table, box = scene_with_table_and_box
    
    # Set up executor with the scene
    executor = LATNLayerExecutor(scene_model=scene_model)
    
    # Test sentence with verb to exercise Layer 4
    sentence = "move the box above the table to [5,5,5]"
    
    print(f"\n" + "="*60)
    print(f"🚀 TESTING LAYER 4")
    print(f"="*60)
    
    try:
        layer4_result = executor.execute_layer4(sentence)
        
        print(f"\n✅ Layer 4 Result:")
        print(f"  Success: {layer4_result.success}")
        print(f"  Description: {getattr(layer4_result, 'description', 'No description')}")
        print(f"  Hypotheses: {len(layer4_result.hypotheses) if hasattr(layer4_result, 'hypotheses') else 0}")
        
        # Basic assertions for Layer 4
        assert layer4_result.success, "Layer 4 should succeed"
        
        # Check Layer 4 specific results
        if hasattr(layer4_result, 'verb_phrases'):
            print(f"  Verb Phrases: {len(layer4_result.verb_phrases)}")
            for i, vp in enumerate(layer4_result.verb_phrases):
                print(f"    [{i}] {vp}")
        
        if hasattr(layer4_result, 'hypotheses') and layer4_result.hypotheses:
            best_l4_hypothesis = layer4_result.hypotheses[0]
            
            print(f"\n🔍 Layer 4 Best Hypothesis Analysis:")
            print(f"  Confidence: {getattr(best_l4_hypothesis, 'confidence', 'N/A')}")
            print(f"  Token count: {len(best_l4_hypothesis.tokens) if hasattr(best_l4_hypothesis, 'tokens') else 0}")
            
            if hasattr(best_l4_hypothesis, 'tokens'):
                print(f"  Tokens:")
                for i, token in enumerate(best_l4_hypothesis.tokens):
                    token_type = type(token).__name__
                    print(f"    [{i}] {token_type}: {token}")
                    
                    # Check for verb phrase replacements
                    if hasattr(best_l4_hypothesis, 'vp_replacements'):
                        print(f"  VP Replacements: {len(best_l4_hypothesis.vp_replacements)}")
                        for i, vp_repl in enumerate(best_l4_hypothesis.vp_replacements):
                            print(f"    [{i}] {vp_repl}")
        
        return layer4_result
    
    except Exception as e:
        print(f"❌ Layer 4 failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Layer 4 failed with exception: {e}")


def test_integrated_layer3_and_layer4(scene_with_table_and_box):
    """Test both Layer 3 and Layer 4 in sequence."""
    scene_model, table, box = scene_with_table_and_box
    
    # Set up executor with the scene
    executor = LATNLayerExecutor(scene_model=scene_model)
    
    # Test sentence with verb to exercise both layers
    sentence = "move the box above the table to [5,5,5]"
    
    # Execute Layer 3
    layer3_result = executor.execute_layer3(sentence, enable_semantic_grounding=True)
    assert layer3_result.success, "Layer 3 should succeed in integrated test"
    
    # Execute Layer 4  
    layer4_result = executor.execute_layer4(sentence)
    assert layer4_result.success, "Layer 4 should succeed in integrated test"
    
    # Verify both layers succeeded
    print(f"✅ Integrated test completed: Layer 3 and Layer 4 both succeeded!")
