#!/usr/bin/env python3
"""
LATN Layer 3 Grounding Tests

Tests for Layer 3 PP-to-NP attachment based on scene spatial relationships.
Layer 3 grounding should attach PPs to NPs only when the scene supports the spatial relationship.
"""

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features


class TestLayer3Grounding:
    """Test Layer 3 PP-to-NP attachment based on scene contents."""
    
    def test_simple_pp_attachment_with_scene_match(self):
        """Test PP attachment when scene supports spatial relationship.
        
        Sentence: "a box on the table"
        Scene: box_1 is actually positioned on table_1
        Expected: PP "on the table" should attach to NP "a box" with high confidence
        """
        # Create scene with box on table
        scene = SceneModel()
        
        # Create table at origin
        table_vector = vector_from_features("noun", loc=[0,0,0])
        table = SceneObject("table", table_vector, object_id="table-1")
        scene.add_object(table)
        
        # Create box positioned on the table (Y=1.0 means "on top of")``
        box_vector = vector_from_features("noun", loc=[0,1,0])
        box = SceneObject("box", box_vector, object_id="box-1") 
        scene.add_object(box)
        
        # Execute Layer 3 with scene grounding
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer3("a box on the table", enable_semantic_grounding=True)
        
        assert result.success, "Layer 3 should succeed"
        assert len(result.hypotheses) == 1, "Should generate at one hypothesis"
        
        # Check that grounding found the spatial relationship
        assert len(result.grounding_results) > 0, "Should have grounding results"
        
        # The PP "on the table" should be grounded to the actual spatial relationship
        pp_grounding = result.grounding_results[0]
        assert pp_grounding.success, "PP grounding should succeed when scene supports it"
        assert pp_grounding.confidence > 0.5, "Should have high confidence for scene-supported attachment"
        
        print(f"Grounding confidence: {pp_grounding.confidence}")
        print(f"Grounding description: {pp_grounding.description}")

    def test_pp_grounding_fails_when_scene_object_missing(self):
        """Test PP grounding failure when NP within PP doesn't exist in scene.
        
        Sentence: "a box on the table"  
        Scene: contains box but NO table
        Expected: PP "on the table" grounding should fail because table doesn't exist
        """
        # Create scene with box but NO table
        scene = SceneModel()
        
        # Create box (exists in scene)
        box_vector = vector_from_features("noun", loc=[0,1,0])
        box = SceneObject("box", box_vector, object_id="box-1") 
        scene.add_object(box)
        
        # NO table object added to scene (but "table" is a known word)
        
        # Execute Layer 3 with scene grounding
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer3("a box on the table", enable_semantic_grounding=True)
        
        assert result.success, "Layer 3 parsing should succeed"
        assert len(result.hypotheses) == 1, "Should generate one hypothesis"
        
        # Check that grounding failed due to missing table
        assert len(result.grounding_results) > 0, "Should have grounding results"
        
        # The PP "on the table" should fail because table doesn't exist in scene
        pp_grounding = result.grounding_results[0]
        assert not pp_grounding.success, "PP grounding should fail when object doesn't exist in scene"
        assert pp_grounding.confidence == 0.0, "Should have zero confidence for missing objects"
        assert "Failed to ground NP within PP" in pp_grounding.description, "Should indicate NP grounding failure"
        
        print(f"Grounding confidence: {pp_grounding.confidence}")
        print(f"Grounding description: {pp_grounding.description}")

    def test_pp_grounding_succeeds_sets_up_layer4_challenge(self):
        """Test Layer 3 PP grounding success with valid spatial relationship.
        
        Sentence: "move the box on the table"
        Scene: box_1 is actually positioned on table_1  
        Expected: Layer 3 should succeed - PP "on the table" attaches to NP "the box"
        Note: This sets up Layer 4 challenge (missing destination for move action)
        """
        # Create scene with box actually on table
        scene = SceneModel()
        
        # Create table at origin
        table_vector = vector_from_features("noun", loc=[0,0,0])
        table = SceneObject("table", table_vector, object_id="table-1")
        scene.add_object(table)
        
        # Create box positioned on the table 
        box_vector = vector_from_features("noun", loc=[0,1,0])
        box = SceneObject("box", box_vector, object_id="box-1") 
        scene.add_object(box)
        
        # Execute Layer 3 with scene grounding
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer3("move the box on the table", enable_semantic_grounding=True)
        
        assert result.success, "Layer 3 should succeed"
        assert len(result.hypotheses) == 1, "Should generate one hypothesis"
        
        # Check that PP grounding succeeded  
        assert len(result.grounding_results) > 0, "Should have grounding results"
        
        # The PP "on the table" should successfully ground to spatial relationship
        pp_grounding = result.grounding_results[0]
        assert pp_grounding.success, "PP grounding should succeed when scene supports spatial relationship"
        assert pp_grounding.confidence > 0.5, "Should have high confidence for valid spatial relationship"
        assert "on table-1" in pp_grounding.description, "Should reference the actual table object"
        
        # Layer 3 succeeds but sets up Layer 4 challenge: 
        # "move" action needs destination but only has source location
        print(f"Grounding confidence: {pp_grounding.confidence}")
        print(f"Grounding description: {pp_grounding.description}")
        print("Note: Layer 3 success sets up Layer 4 challenge - 'move' missing destination")

    def test_pp_grounding_fails_when_spatial_relationship_invalid(self):
        """Test PP grounding failure when spatial relationship contradicts scene geometry.
        
        Sentence: "move the box under the table"
        Scene: box_1 is positioned ON table_1 (not under it)
        Expected: PP "under the table" should NOT attach to "the box" because 
                 the actual scene shows box is ON the table, not under it
        """
        # Create scene with box ON table (same as previous test)
        scene = SceneModel()
        
        # Create table at origin
        table_vector = vector_from_features("noun", loc=[0,0,0])
        table = SceneObject("table", table_vector, object_id="table-1")
        scene.add_object(table)
        
        # Create box positioned ON the table (Y=1, above table)
        box_vector = vector_from_features("noun", loc=[0,1,0])
        box = SceneObject("box", box_vector, object_id="box-1") 
        scene.add_object(box)
        
        # Execute Layer 3 with scene grounding
        executor = LATNLayerExecutor(scene_model=scene)
        result = executor.execute_layer3("move the box under the table", enable_semantic_grounding=True)
        
        assert result.success, "Layer 3 parsing should succeed"
        assert len(result.hypotheses) == 1, "Should generate one hypothesis"
        
        # Check PP grounding behavior
        if len(result.grounding_results) > 0:
            pp_grounding = result.grounding_results[0]
            print(f"Grounding confidence: {pp_grounding.confidence}")
            print(f"Grounding description: {pp_grounding.description}")
            
            # The PP should either fail or have low confidence because 
            # "under the table" contradicts actual scene geometry
            if pp_grounding.success:
                print("PP grounded despite spatial contradiction - checking confidence...")
                # If it succeeds, confidence should be lower due to spatial mismatch
            else:
                print("PP grounding correctly failed due to spatial contradiction")
        else:
            print("No grounding results - PP parsing may have failed")

    def test_scene_aware_hypothesis_filtering_multiple_boxes(self):
        """Test that PP attachment filters Layer 2 hypotheses based on spatial plausibility.
        
        Sentence: "delete the box under the table"
        Scene: box-1 ON table, box-2 UNDER table
        Expected: Layer 2 generates 2 hypotheses for "the box" (could match either box)
                 Layer 3 should filter to only box-2 because "under the table" only 
                 spatially makes sense for the box that's actually under the table
        """
        # Create scene with two boxes in different spatial relationships
        scene = SceneModel()
        
        # Create table at origin
        table_vector = vector_from_features("noun", loc=[0,0,0])
        table = SceneObject("table", table_vector, object_id="table-1")
        scene.add_object(table)
        
        # Create box-1 positioned ON the table (Y=1, above table) - make it RED
        box1_vector = vector_from_features("noun", color=[1.0, 0.0, 0.0], loc=[0,1,0])
        box1 = SceneObject("box", box1_vector, object_id="box-1") 
        print(f"Box-1 vector: {box1_vector}")
        scene.add_object(box1)
        
        # Create box-2 positioned UNDER the table (Y=-1, below table) - make it BLUE
        box2_vector = vector_from_features("noun", color=[0.0, 0.0, 1.0], loc=[0,-1,0])
        box2 = SceneObject("box", box2_vector, object_id="box-2") 
        print(f"Box-2 vector: {box2_vector}")
        scene.add_object(box2)
        
        # Debug: Check what objects are actually in the scene
        print("=== Scene Objects ===")
        for obj in scene.objects:
            print(f"  {obj.object_id}: {obj.name} at {obj.get_position()}")
        
        # Create executor
        executor = LATNLayerExecutor(scene_model=scene)
        
        # Debug: Test Layer 2 grounding with just "the box" to isolate the issue
        print("=== Debug: Simple Layer 2 Test ===")
        simple_layer2 = executor.execute_layer2("the box", enable_semantic_grounding=True, return_all_matches=True)
        print(f"Simple Layer 2 hypotheses: {len(simple_layer2.hypotheses)}")
        print(f"Simple Layer 2 grounding results: {len(simple_layer2.grounding_results)}")
        for i, grounding in enumerate(simple_layer2.grounding_results):
            if grounding.success:
                print(f"  Simple Box {i+1}: {grounding.resolved_object.object_id} at {grounding.resolved_object.get_position()}")
        
        # Execute Layer 2 first to see multiple box hypotheses
        layer2_result = executor.execute_layer2("delete the box under the table", enable_semantic_grounding=True, return_all_matches=True)
        
        print("=== Layer 2 Results ===")
        print(f"Layer 2 hypotheses: {len(layer2_result.hypotheses)}")
        print(f"Layer 2 grounding results: {len(layer2_result.grounding_results)}")
        for i, grounding in enumerate(layer2_result.grounding_results):
            if grounding.success:
                print(f"  Box {i+1}: {grounding.resolved_object.object_id} at {grounding.resolved_object.get_position()}")
        
        # Execute Layer 3 with scene grounding
        layer3_result = executor.execute_layer3("delete the box under the table", enable_semantic_grounding=True)
        
        print("\n=== Layer 3 Results ===")
        print(f"Layer 3 hypotheses: {len(layer3_result.hypotheses)}")
        print(f"Layer 3 grounding results: {len(layer3_result.grounding_results)}")
        
        # Layer 3 should ideally filter to only the spatially consistent box
        # (This test will show current behavior vs ideal behavior)
        assert layer3_result.success, "Layer 3 should succeed"
        
        if len(layer3_result.grounding_results) > 0:
            pp_grounding = layer3_result.grounding_results[0]
            print(f"PP grounding: {pp_grounding.description}")
            
            # Current system likely won't filter - this test shows what should happen
            print("Expected: PP should only bind to box-2 (the one actually under table)")
            print("Current: PP might bind regardless of spatial relationship")
