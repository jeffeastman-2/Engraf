"""
Test the grouping functionality with a simple demonstration.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer


def test_grouping():
    """Test the grouping functionality."""
    print("🧪 Testing Assembly Creation via Grouping")
    print("=" * 50)
    
    # Create interpreter with mock renderer for testing
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Step 1: Create some objects to group
    print("\n📦 Creating objects to group...")
    result1 = interpreter.interpret("draw a very tall box at [-1,0,0]")
    print(f"   Created: {result1.get('objects_created', [])}")
    
    result2 = interpreter.interpret("draw a very tall box at [1,0,0]")
    print(f"   Created: {result2.get('objects_created', [])}")
    
    result3 = interpreter.interpret("draw a very tall box at [0,2,0] and rotate it by 90 degrees")
    print(f"   Created: {result3.get('objects_created', [])}")
    
    # Step 2: Try to group them with simpler command first
    print("\n🔗 Attempting to group objects...")
    group_result = interpreter.interpret("group them")
    print(f"   Group result: {group_result}")
        
    # Step 3: Check the scene state
    print("\n📊 Scene state after grouping:")
    print(f"   Standalone objects: {len(interpreter.scene.objects)}")
    print(f"   Assemblies: {len(interpreter.scene.assemblies) if hasattr(interpreter.scene, 'assemblies') else 0}")
    
    if hasattr(interpreter.scene, 'assemblies') and interpreter.scene.assemblies:
        for assembly in interpreter.scene.assemblies:
            print(f"   Assembly '{assembly.assembly_id}': {len(assembly.objects)} objects")


if __name__ == "__main__":
    test_grouping()
