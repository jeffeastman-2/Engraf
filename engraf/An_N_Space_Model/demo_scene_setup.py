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
        print(f"  • {obj.object_id}: {obj.noun} at {pos}")
    print()
