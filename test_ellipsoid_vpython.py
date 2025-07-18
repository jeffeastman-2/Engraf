#!/usr/bin/env python3
"""
Test ellipsoid with VPython renderer
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer

def test_ellipsoid_vpython():
    print("🎨 Testing Ellipsoid with VPython Renderer")
    print("=" * 50)
    
    # Create VPython renderer
    renderer = VPythonRenderer(
        width=800,
        height=600,
        title="Ellipsoid Test",
        headless=False
    )
    
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # Test sentences with ellipsoids
    test_sentences = [
        "draw a red ellipsoid at [0, 0, 0]",
        "draw a tall blue ellipsoid at [3, 0, 0]",
        "draw a very tall green ellipsoid at [6, 0, 0]", 
        "draw a wide yellow ellipsoid at [0, 3, 0]",
        "draw a small purple ellipsoid at [3, 3, 0]"
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i}. Processing: '{sentence}'")
        
        try:
            result = interpreter.interpret(sentence)
            
            if result['success']:
                print(f"   ✅ Success: {result['message']}")
                if result['objects_created']:
                    print(f"   📦 Created: {', '.join(result['objects_created'])}")
            else:
                print(f"   ❌ Failed: {result['message']}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
        
        print()
    
    # Scene summary
    summary = interpreter.get_scene_summary()
    print("📊 Scene Summary:")
    print(f"   Total objects: {summary['total_objects']}")
    print(f"   Object types: {summary['object_types']}")
    print()
    
    print("🎯 Check the VPython window - you should see:")
    print("   - Red ellipsoid (normal proportions)")
    print("   - Blue ellipsoid (tall - stretched vertically)")
    print("   - Green ellipsoid (very tall - stretched even more vertically)")
    print("   - Yellow ellipsoid (wide - stretched horizontally)")
    print("   - Purple ellipsoid (small - shrunk in all dimensions)")
    print()
    
    input("Press Enter to continue...")

if __name__ == "__main__":
    test_ellipsoid_vpython()
