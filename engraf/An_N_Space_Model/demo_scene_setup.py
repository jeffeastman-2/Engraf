#!/usr/bin/env python3
"""
Common Scene Setup for LATN Layer 3 Demos

This module provides a standardized scene setup that all Layer 3 demos can use
to ensure consistent spatial relationships for testing.
"""

from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features


def setup_demo_scene():
    """Create a standardized demo scene with positioned objects for spatial relationships.
    
    Scene layout:
    - table_1: table at origin [0, 0, 0] (reference point)
    - red_cube_1: red cube above the table at [0, 1, 0]
    - blue_sphere_1: blue sphere to the right of the cube at [2, 1, 0]
    - green_cylinder_1: green cylinder behind the table at [0, 0, -2]
    
    Spatial relationships that should be valid:
    - "cube above table" (Y=1 > Y=0) ✓
    - "sphere right of cube" (X=2 > X=0) ✓
    - "cylinder behind table" (Z=-2 < Z=0) ✓
    - "cube left of sphere" (X=0 < X=2) ✓
    
    Returns:
        SceneModel: The configured scene with 4 objects
    """
    scene = SceneModel()
    
    # Table at origin (reference point)
    table_vector = vector_from_features("noun", locX=0, locY=0, locZ=0)
    table = SceneObject("table", table_vector, object_id="table_1")
    scene.add_object(table)
    
    # Red cube above the table
    red_cube_vector = vector_from_features("noun", red=1.0, locX=0, locY=1, locZ=0)
    red_cube = SceneObject("cube", red_cube_vector, object_id="red_cube_1")
    scene.add_object(red_cube)
    
    # Blue sphere to the right of the cube
    blue_sphere_vector = vector_from_features("noun", blue=1.0, locX=2, locY=1, locZ=0)
    blue_sphere = SceneObject("sphere", blue_sphere_vector, object_id="blue_sphere_1")
    scene.add_object(blue_sphere)
    
    # Green cylinder behind the table
    green_cylinder_vector = vector_from_features("noun", green=1.0, locX=0, locY=0, locZ=-2)
    green_cylinder = SceneObject("cylinder", green_cylinder_vector, object_id="green_cylinder_1")
    scene.add_object(green_cylinder)
    
    return scene


def print_scene_info(scene):
    """Print scene object positions for debugging."""
    print("Scene Setup:")
    print(f"- {len(scene.objects)} objects positioned for spatial testing")
    for obj in scene.objects:
        pos = [obj.vector['locX'], obj.vector['locY'], obj.vector['locZ']]
        print(f"  • {obj.object_id}: {obj.name} at {pos}")
    print()


def get_common_test_phrases():
    """Return common test phrases used across layer demos for consistency.
    
    Returns:
        dict: Categories of test phrases used across different layer demos
    """
    return {
        'spatial_validation': [
            "the cube above the table",
            "the sphere right of the cube", 
            "the cylinder behind the table",
            "the table above the cube",  # Should fail validation
            "the cube left of the sphere"
        ],
        
        'vector_coordinates': [
            "move the cube to [5, 0, 0]",
            "place the sphere at [1, 2, 3]", 
            "position the cylinder to [-1, -1, -1]",
            "put the table at the origin [0, 0, 0]"
        ],
        
        'complex_spatial_chains': [
            "move the cube above the table right of the sphere",
            "place the sphere below the cube behind the cylinder", 
            "position the cylinder left of the table above the sphere",
            "put the cube above the table below the sphere to [2, 2, 2]"
        ],
        
        'simple_grounding': [
            "the blue sphere",
            "the red cube",
            "the green cylinder",
            "the table"
        ],
        
        'layer4_verb_phrases': [
            "draw a very tall box at [-1, 0, 0] and a very tall box at [1, 0, 0]",
            "draw a very tall box at [0,1,0] and rotate it by 90 degrees",
            "group them",
            "move it to [5,5,5]",
            "draw a red cube at [0, 0, 0]",
            "move it to [2, 3, 4]",
            "make it bigger",
            "color it blue",
            "draw a big blue sphere at [3, 0, 0]",
            "move it above the cube",
            "draw a green cylinder at [-3, 0, 0]",
            "rotate it by 90 degrees"
        ]
    }


def get_all_test_phrases():
    """Return all test phrases as a flat list for comprehensive testing."""
    phrases = get_common_test_phrases()
    all_phrases = []
    for category in phrases.values():
        all_phrases.extend(category)
    return all_phrases


def process_test_phrase_category(executor_method, test_phrases_dict, enable_grounding=False):
    """Process all categories of test phrases and display the results.
    
    Args:
        executor_method: The specific layer executor method to call (e.g., executor.execute_layer1)
        test_phrases_dict: Dictionary of test phrase categories
    """
    # Process each category separately
    for category_name, phrases in test_phrases_dict.items():
        print(f"\n=== {category_name.replace('_', ' ').title()} ===")
        
        for phrase in phrases:
            print(f"\n📝 Input: \"{phrase}\"")
            print("-" * 30)
            
            # Call the passed executor method
            result = executor_method(phrase, enable_semantic_grounding=enable_grounding)
            
            if result.success:
                print(f"✅ Generated {len(result.hypotheses)} tokenization hypothesis(es)")
                
                for i, hyp in enumerate(result.hypotheses[:3], 1):  # Show top 3
                    tokens = [t.word for t in hyp.tokens]
                    print(f"  {i}. {tokens}")
                    print(f"     Confidence: {hyp.confidence:.3f}")
                    hyp.print_tokens()
                    
            else:
                print("❌ Tokenization failed")
