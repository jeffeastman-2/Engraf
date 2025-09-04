#!/usr/bin/env python3
"""
LATN Layer 2 Grounding Test Suite

Tests semantic grounding of NP tokens to existing scene objects.
"""

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.vector_space import vector_from_features
from engraf.visualizer.scene.scene_model import SceneModel, SceneObject
from engraf.pos.noun_phrase import NounPhrase


class TestLayer2Grounding:
    
    def test_simple_exact_match(self):
        """Test basic NP grounding: 'the red box' should find existing red box in scene."""
        
        # Create a simple scene with one red box
        scene = SceneModel()
        
        # Create a vector for red box
        red_box_vector = vector_from_features("noun", color=[1.0, 0.0, 0.0], scale=[2.0, 0.0, 0.0])
        
        red_box = SceneObject(
            name="box",
            object_id="box-1",
            vector=red_box_vector
        )
        scene.add_object(red_box)
        
        # Execute Layer 2 with grounding enabled
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer2("the red box")
        
        # Verify basic success
        assert result.success, "Layer 2 should process successfully"
        assert len(result.hypotheses) > 0, "Should generate hypotheses"
        
        # Check that grounding occurred
        best_hypothesis = result.hypotheses[0]
        
        # Print debug info to understand the mechanism
        print(f"\nDEBUG: Result confidence: {best_hypothesis.confidence}")
        print(f"DEBUG: Number of hypotheses: {len(result.hypotheses)}")
        print(f"DEBUG: Hypothesis tokens: {[token.word for token in best_hypothesis.tokens]}")
        print(f"DEBUG: NP replacements: {best_hypothesis.replacements}")
        
        # Check grounding results
        assert len(result.grounding_results) > 0, "Should have grounding results"
        grounding_result = result.grounding_results[0]
        assert grounding_result.success, "Grounding should succeed"
        assert grounding_result.resolved_object is not None, "Should resolve to an object"
        assert grounding_result.grounded_noun_phrase is not None, "Should create grounded NounPhrase"
        
        # Check the grounded NounPhrase
        grounded_np_result = grounding_result.grounded_noun_phrase
        assert grounded_np_result is not None, "Should have grounded noun phrase"
        assert grounded_np_result.grounding is not None, "NounPhrase should have grounding info"
        assert grounded_np_result.grounding['scene_object'] == red_box, "Should resolve to red box"
        
        # With our improved architecture, the result.noun_phrases should contain grounded NPs
        grounded_np = result.noun_phrases[0]
        from engraf.pos.noun_phrase import NounPhrase
        assert isinstance(grounded_np, NounPhrase), "Grounded NP should be NounPhrase"
        assert grounded_np.grounding is not None, "NounPhrase should have grounding info"
        assert grounded_np.grounding['scene_object'] == red_box, "Should resolve to red box"
        
        # Check the NP token has grounding
        np_token = best_hypothesis.tokens[0]
        # The token should have a reference to the grounded NP
        assert hasattr(np_token, 'grounding') or hasattr(np_token, '_grounded_phrase'), "Token should have grounding info"
        
        # Basic token structure assertions
        assert len(best_hypothesis.tokens) == 1, "Should have one NP token"
        assert best_hypothesis.tokens[0].word == "NP(the red box)", "Should create NP token"
    
    def test_multiple_objects_same_query(self):
        """Test NP grounding with multiple matching objects: 'the box' should find multiple boxes."""
        
        # Create a scene with two different boxes
        scene = SceneModel()
        
        # Create vectors for different colored boxes
        red_box_vector = vector_from_features("noun", color=[1.0, 0.0, 0.0], scale=[2.0, 0.0, 0.0])
        blue_box_vector = vector_from_features("noun", color=[0.0, 0.0, 1.0], scale=[2.0, 0.0, 0.0])
        
        red_box = SceneObject(
            name="box",
            object_id="box-1",
            vector=red_box_vector
        )
        blue_box = SceneObject(
            name="box", 
            object_id="box-2",
            vector=blue_box_vector
        )
        scene.add_object(red_box)
        scene.add_object(blue_box)
        
        # Execute Layer 2 with ambiguous query "the box"
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer2("the box")
        
        # Verify basic success
        assert result.success, "Layer 2 should process successfully"
        
        # Expect 2 hypotheses for 2 matching objects (hypothesis multiplication)
        assert len(result.hypotheses) == 2, f"Should generate 2 hypotheses for 2 matching objects, got {len(result.hypotheses)}"
        
        # Check that grounding occurred
        best_hypothesis = result.hypotheses[0]
        
        # Print debug info to understand how multiple objects are handled
        print(f"\nDEBUG: Result confidence: {best_hypothesis.confidence}")
        print(f"DEBUG: Number of hypotheses: {len(result.hypotheses)}")
        print(f"DEBUG: Hypothesis tokens: {[token.word for token in best_hypothesis.tokens]}")
        
        # Verify the system created 2 hypotheses (hypothesis multiplication working!)
        assert len(result.grounding_results) >= 1, "Should have grounding results"
        
        # Verify we have 2 noun phrases (one per hypothesis)  
        assert len(result.noun_phrases) == 2, f"Should have 2 noun phrases (one per hypothesis), got {len(result.noun_phrases)}"
        
        # Verify each hypothesis has grounding information via tokens
        grounded_objects = []
        for hypothesis in result.hypotheses:
            for token in hypothesis.tokens:
                if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                    if token._grounded_phrase.grounding is not None:
                        grounded_objects.append(token._grounded_phrase.grounding['scene_object'].object_id)
        
        assert len(grounded_objects) == 2, "Both hypotheses should be grounded"
        assert "box-1" in grounded_objects, "Should have box-1"
        assert "box-2" in grounded_objects, "Should have box-2"
        
        # Verify each hypothesis has a different grounding
        # The test should check if the grounded objects are correct
        # since descriptions may not contain object IDs in the current implementation
        assert len(grounded_objects) == 2, f"Should have 2 grounded objects, got {len(grounded_objects)}"
        assert "box-1" in grounded_objects, f"Should have box-1 in grounded objects: {grounded_objects}"
        assert "box-2" in grounded_objects, f"Should have box-2 in grounded objects: {grounded_objects}"
        
        print(f"✅ Hypothesis multiplication successful!")
        print(f"   Hypothesis 1: {result.hypotheses[0].description}")
        print(f"   Hypothesis 2: {result.hypotheses[1].description}")
        print(f"   Grounded to: {grounded_objects}")
    
    def test_cartesian_product_multiple_nps(self):
        """Test Cartesian product: 2 ambiguous NPs should create 4 hypotheses (2×2)."""
        
        # Create a scene with 2 boxes and 2 spheres  
        scene = SceneModel()
        
        # Create vectors for objects
        red_box_vector = vector_from_features("noun", color=[1.0, 0.0, 0.0])
        blue_box_vector = vector_from_features("noun", color=[0.0, 0.0, 1.0])
        green_sphere_vector = vector_from_features("noun", color=[0.0, 1.0, 0.0])
        tall_sphere_vector = vector_from_features("noun", scale=[0.0, 3.0, 0.0])  # tall
        
        red_box = SceneObject(name="box", object_id="red-box", vector=red_box_vector)
        blue_box = SceneObject(name="box", object_id="blue-box", vector=blue_box_vector)
        green_sphere = SceneObject(name="sphere", object_id="green-sphere", vector=green_sphere_vector)
        tall_sphere = SceneObject(name="sphere", object_id="tall-sphere", vector=tall_sphere_vector)
        
        scene.add_object(red_box)
        scene.add_object(blue_box)
        scene.add_object(green_sphere)
        scene.add_object(tall_sphere)
        
        # Test query with 2 ambiguous NPs: "the box under the sphere"
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer2("the box under the sphere")
        
        # Verify basic success
        assert result.success, "Layer 2 should process successfully"
        
        # Expect 4 hypotheses (2 boxes × 2 spheres = 4 combinations)
        assert len(result.hypotheses) == 4, f"Should generate 4 hypotheses (2×2), got {len(result.hypotheses)}"
        
        # Verify we have 8 noun phrases (2 NPs per hypothesis × 4 hypotheses)
        assert len(result.noun_phrases) == 8, f"Should have 8 noun phrases (2 per hypothesis), got {len(result.noun_phrases)}"
        
        # Print debug info
        print(f"\nDEBUG: Cartesian product test results:")
        print(f"DEBUG: Number of hypotheses: {len(result.hypotheses)}")
        print(f"DEBUG: Number of noun phrases: {len(result.noun_phrases)}")
        
        # Collect all grounding combinations
        combinations = []
        for i, hypothesis in enumerate(result.hypotheses):
            print(f"DEBUG: Hypothesis {i+1}: {hypothesis.description}")
            
            # Extract the resolved objects for this hypothesis via grounded phrases
            hypothesis_nps = []
            for token in hypothesis.tokens:
                if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                    if token._grounded_phrase.grounding is not None:
                        hypothesis_nps.append(token._grounded_phrase.grounding['scene_object'].object_id)
            
            combinations.append(tuple(hypothesis_nps))
            print(f"       → Objects: {hypothesis_nps}")
        
        # Verify we have all 4 expected combinations
        expected_combinations = {
            ("red-box", "green-sphere"),
            ("red-box", "tall-sphere"),
            ("blue-box", "green-sphere"), 
            ("blue-box", "tall-sphere")
        }
        
        actual_combinations = set(combinations)
        print(f"DEBUG: Expected combinations: {expected_combinations}")
        print(f"DEBUG: Actual combinations: {actual_combinations}")
        
        assert actual_combinations == expected_combinations, f"Should have all 4 box/sphere combinations"
        
        print(f"✅ Cartesian product successful! Generated all {len(expected_combinations)} combinations.")
    
    def test_triple_cartesian_product_with_universal_matcher(self):
        """Test triple Cartesian product: 3 ambiguous NPs should create 8 hypotheses (2×2×2)."""
        
        # Create a scene with 2 boxes, 2 spheres, plus a green box for "green object" testing
        scene = SceneModel()
        
        # Create vectors for objects
        red_box_vector = vector_from_features("noun", color=[1.0, 0.0, 0.0])
        blue_box_vector = vector_from_features("noun", color=[0.0, 0.0, 1.0])
        green_box_vector = vector_from_features("noun", color=[0.0, 1.0, 0.0])  # Add green box
        green_sphere_vector = vector_from_features("noun", color=[0.0, 1.0, 0.0])
        tall_sphere_vector = vector_from_features("noun", scale=[0.0, 3.0, 0.0])
        
        red_box = SceneObject(name="box", object_id="red-box", vector=red_box_vector)
        blue_box = SceneObject(name="box", object_id="blue-box", vector=blue_box_vector)
        green_box = SceneObject(name="box", object_id="green-box", vector=green_box_vector)  # Add green box
        green_sphere = SceneObject(name="sphere", object_id="green-sphere", vector=green_sphere_vector)
        tall_sphere = SceneObject(name="sphere", object_id="tall-sphere", vector=tall_sphere_vector)
        
        scene.add_object(red_box)
        scene.add_object(blue_box)
        scene.add_object(green_box)  # Add green box to scene
        scene.add_object(green_sphere)
        scene.add_object(tall_sphere)
        
        # Test query with 3 NPs: "a box under a sphere above a green object"
        # - "box" matches: red-box, blue-box, green-box (3 options)
        # - "sphere" matches: green-sphere, tall-sphere (2 options)  
        # - "green object" matches: green-box, green-sphere (2 options)
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer2("the box under the sphere above the green object")

        # Verify basic success
        assert result.success, "Layer 2 should process successfully"
        
        # Print debug info first to see what we actually get
        print(f"\nDEBUG: Triple Cartesian product test results:")
        print(f"DEBUG: Number of hypotheses: {len(result.hypotheses)}")
        print(f"DEBUG: Number of noun phrases: {len(result.noun_phrases)}")
        
        # Check the first few hypotheses to understand the structure
        for i, hypothesis in enumerate(result.hypotheses[:5]):  # Show first 5
            print(f"DEBUG: Hypothesis {i+1}: {hypothesis.description}")
            
            # Extract the resolved objects for this hypothesis via grounded phrases
            hypothesis_nps = []
            for token in hypothesis.tokens:
                if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                    if token._grounded_phrase.grounding is not None:
                        hypothesis_nps.append(token._grounded_phrase.grounding['scene_object'].object_id)
            
            print(f"       → Objects: {hypothesis_nps}")
        
        # The actual result with green box added:
        # - "box" matches: red-box, blue-box, green-box (3 options)  
        # - "sphere" matches: green-sphere, tall-sphere (2 options)
        # - "green object" matches: green-box, green-sphere (2 options)
        # Result: 3 × 2 × 2 = 12 hypotheses
        expected_hypotheses = 12
        
        assert len(result.hypotheses) == expected_hypotheses, f"Should generate {expected_hypotheses} hypotheses, got {len(result.hypotheses)}"
        
        # Verify we have 3 NPs per hypothesis
        expected_nps = len(result.hypotheses) * 3
        assert len(result.noun_phrases) == expected_nps, f"Should have {expected_nps} noun phrases (3 per hypothesis), got {len(result.noun_phrases)}"
        
        # Verify that "object" matching worked - check if "green object" matches both green objects
        object_position_matches = set()
        for hypothesis in result.hypotheses:
            hypothesis_objects = []
            for token in hypothesis.tokens:
                if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
                    if token._grounded_phrase.grounding is not None:
                        hypothesis_objects.append(token._grounded_phrase.grounding['scene_object'].object_id)
            
            if len(hypothesis_objects) == 3:  # box, sphere, object
                object_position_matches.add(hypothesis_objects[2])  # Third NP should be the "green object"
        
        print(f"DEBUG: Objects matched by 'green object': {object_position_matches}")
        
        # Both green objects should be matched by "green object"
        assert "green-sphere" in object_position_matches, "Green sphere should match 'green object'"
        assert "green-box" in object_position_matches, "Green box should match 'green object'"
        
        print(f"✅ Triple Cartesian product with universal matcher successful!")
        print(f"   Generated {len(result.hypotheses)} hypotheses with 'object' matching {len(object_position_matches)} different objects")
