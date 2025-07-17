#!/usr/bin/env python3
"""
Simple VPython test to debug the renderer issue
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from unittest.mock import Mock
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


def create_mock_vector_space(properties: dict) -> VectorSpace:
    """Create a mock VectorSpace with specific properties."""
    vector = Mock(spec=VectorSpace)
    vector.__getitem__ = Mock(side_effect=lambda key: properties.get(key, 0.0))
    return vector


def test_simple_objects():
    """Test simple objects one by one."""
    print("🔍 Testing VPython renderer with simple objects...")
    
    try:
        # Create VPython renderer
        renderer = VPythonRenderer(
            width=800, 
            height=600, 
            title="Simple Test",
            headless=False
        )
        
        print("✅ Renderer created successfully")
        
        # Test simple cube
        cube_vector = create_mock_vector_space({
            "locX": 0.0, "locY": 0.0, "locZ": 0.0,
            "scaleX": 2.0, "scaleY": 2.0, "scaleZ": 2.0,
            "red": 1.0, "green": 0.0, "blue": 0.0
        })
        cube = SceneObject("cube", cube_vector)
        
        print("🟥 Rendering cube...")
        renderer.render_object(cube)
        print(f"✅ Cube rendered: {cube.name in renderer.rendered_objects}")
        
        # Test simple sphere
        sphere_vector = create_mock_vector_space({
            "locX": 3.0, "locY": 0.0, "locZ": 0.0,
            "scaleX": 1.5, "scaleY": 1.5, "scaleZ": 1.5,
            "red": 0.0, "green": 1.0, "blue": 0.0
        })
        sphere = SceneObject("sphere", sphere_vector)
        
        print("🟢 Rendering sphere...")
        renderer.render_object(sphere)
        print(f"✅ Sphere rendered: {sphere.name in renderer.rendered_objects}")
        
        print(f"📊 Total objects rendered: {len(renderer.rendered_objects)}")
        for name in renderer.rendered_objects:
            print(f"  • {name}")
        
        input("Press Enter to exit...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_objects()
