#!/usr/bin/env python3
"""
Isolated test for the adjective scaling bug in ObjectModifier
"""

import os
import sys

# Add the project root to the Python path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def test_adjective_scaling_branch():
    """Test the specific adjective scaling branch in ObjectModifier."""
    print("🔍 Testing adjective scaling branch...")
    print("=" * 50)
    
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # Create the cube first
    result1 = interpreter.interpret("draw a red cube at [0, 0, 0]")
    print(f"1. Create result: {result1['success']}")
    
    # Test "make it bigger" with extra debug
    print(f"\n2. Testing 'make it bigger'...")
    result2 = interpreter.interpret("make it bigger")
    
    print(f"\nResult: {result2}")
    print(f"Success: {result2['success']}")
    print(f"Objects modified: {result2.get('objects_modified', [])}")

if __name__ == "__main__":
    test_adjective_scaling_branch()
