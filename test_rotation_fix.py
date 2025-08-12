#!/usr/bin/env python3
"""Test rotation fix with our new architecture"""

import sys
import os

# Add the project root to path
project_root = os.path.abspath('/Users/jeff/Python/Engraf')
sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer

def test_rotation():
    """Test the green cylinder rotation functionality"""
    print("🔧 Testing Green Cylinder Rotation Fix")
    print("=" * 40)
    
    # Create renderer and interpreter
    renderer = VPythonRenderer(
        width=800,
        height=600,
        title="Rotation Test",
        headless=False
    )
    
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # Create and rotate a green cylinder
    print("📍 Creating green cylinder...")
    result1 = interpreter.interpret("draw a green cylinder at [0, 0, 0]")
    print(f"Result: {result1}")
    
    print("🔄 Rotating cylinder 45 degrees around z-axis...")
    result2 = interpreter.interpret("rotate it 45 degrees around the z-axis")
    print(f"Result: {result2}")
    
    print("✅ Test completed! Check if the green cylinder rotated in the 3D view.")
    
    # Keep the window open
    input("Press Enter to exit...")

if __name__ == "__main__":
    test_rotation()
    test_rotation()
