#!/usr/bin/env python3
"""
VPython 3D Visualization Demo

This script demonstrates the ENGRAF VPython renderer by creating a colorful
3D scene with various geometric objects. Run this script to see the objects
rendered in your browser with interactive 3D controls.

Usage:
    python demo_vpython_3d.py

Controls in the browser:
    - Drag with mouse to rotate the view
    - Mouse wheel to zoom in/out
    - Right-click and drag to pan
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from unittest.mock import Mock
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


def create_mock_vector_space(properties: dict) -> VectorSpace:
    """Create a mock VectorSpace with specific properties."""
    vector = Mock(spec=VectorSpace)
    vector.__getitem__ = Mock(side_effect=lambda key: properties.get(key, 0.0))
    return vector


def create_demo_scene():
    """Create a demonstration scene with various colored objects."""
    scene = SceneModel()
    
    # Red cube on the left
    cube_vector = create_mock_vector_space({
        "locX": -3.0, "locY": 0.0, "locZ": 0.0,
        "scaleX": 1.0, "scaleY": 1.0, "scaleZ": 1.0,
        "red": 1.0, "green": 0.0, "blue": 0.0
    })
    cube = SceneObject("cube", cube_vector)
    scene.add_object(cube)
    
    # Green sphere in the center-left
    sphere_vector = create_mock_vector_space({
        "locX": -1.0, "locY": 0.0, "locZ": 0.0,
        "scaleX": 1.0, "scaleY": 1.0, "scaleZ": 1.0,
        "red": 0.0, "green": 1.0, "blue": 0.0
    })
    sphere = SceneObject("sphere", sphere_vector)
    scene.add_object(sphere)
    
    # Blue cylinder in the center-right
    cylinder_vector = create_mock_vector_space({
        "locX": 1.0, "locY": 0.0, "locZ": 0.0,
        "scaleX": 0.5, "scaleY": 2.0, "scaleZ": 0.5,
        "red": 0.0, "green": 0.0, "blue": 1.0
    })
    cylinder = SceneObject("cylinder", cylinder_vector)
    scene.add_object(cylinder)
    
    # Yellow cone on the right
    cone_vector = create_mock_vector_space({
        "locX": 3.0, "locY": 0.0, "locZ": 0.0,
        "scaleX": 1.0, "scaleY": 1.5, "scaleZ": 1.0,
        "red": 1.0, "green": 1.0, "blue": 0.0
    })
    cone = SceneObject("cone", cone_vector)
    scene.add_object(cone)
    
    # Purple pyramid above
    pyramid_vector = create_mock_vector_space({
        "locX": 0.0, "locY": 2.0, "locZ": 0.0,
        "scaleX": 1.2, "scaleY": 1.2, "scaleZ": 1.2,
        "red": 0.8, "green": 0.0, "blue": 0.8
    })
    pyramid = SceneObject("pyramid", pyramid_vector)
    scene.add_object(pyramid)
    
    # Orange arch below-left
    arch_vector = create_mock_vector_space({
        "locX": -1.0, "locY": -2.0, "locZ": 0.0,
        "scaleX": 1.5, "scaleY": 1.0, "scaleZ": 0.5,
        "red": 1.0, "green": 0.5, "blue": 0.0
    })
    arch = SceneObject("arch", arch_vector)
    scene.add_object(arch)
    
    # Brown table below-right
    table_vector = create_mock_vector_space({
        "locX": 1.0, "locY": -2.0, "locZ": 0.0,
        "scaleX": 2.0, "scaleY": 0.8, "scaleZ": 1.0,
        "red": 0.6, "green": 0.3, "blue": 0.1
    })
    table = SceneObject("table", table_vector)
    scene.add_object(table)
    
    return scene


def main():
    """Main function to create and display the 3D scene."""
    print("🎨 ENGRAF VPython 3D Visualization Demo")
    print("=" * 50)
    print()
    print("Creating 3D scene with colorful objects...")
    print("This will open a browser window with interactive 3D visualization.")
    print()
    
    try:
        # Create VPython renderer (will open browser window)
        renderer = VPythonRenderer(
            width=1000, 
            height=700, 
            title="ENGRAF 3D Scene Demo",
            headless=False  # This will show the browser window
        )
        
        # Create and render the demo scene
        scene = create_demo_scene()
        renderer.render_scene(scene)
        
        print("✅ Scene rendered successfully!")
        print(f"📊 Total objects rendered: {len(renderer.rendered_objects)}")
        print()
        print("Objects in the scene:")
        for name in renderer.rendered_objects:
            print(f"  • {name}")
        print()
        print("🎮 Browser Controls:")
        print("  • Drag with mouse to rotate the view")
        print("  • Mouse wheel to zoom in/out")
        print("  • Right-click and drag to pan")
        print("  • Close the browser tab to exit")
        print()
        print("🎯 What you should see:")
        print("  • Red cube (left) - actual cube shape")
        print("  • Green sphere (center-left) - actual sphere shape")
        print("  • Blue cylinder (center-right, tall) - actual cylinder shape")
        print("  • Yellow cone (right) - actual cone shape")
        print("  • Purple pyramid (above) - square base with triangular sides")
        print("  • Orange arch (below-left) - architectural arch structure")
        print("  • Brown table (below-right) - table with legs")
        print()
        
        # Keep the program running so the browser window stays open
        try:
            input("Press Enter to exit (or Ctrl+C)...")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        
        # Clean up
        renderer.clear_scene()
        print("✨ Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure VPython is installed: pip install vpython")
        print("💡 If you're in a virtual environment, activate it first")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
