"""
Test the naming functionality for assemblies.
This is separate from basic grouping and can be debugged independently.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer


def test_assembly_naming():
    """Test naming assemblies after creation."""
    print("🧪 Testing Assembly Naming Functionality")
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
    
    # Step 2: Group them
    print("\n🔗 Grouping objects...")
    group_result = interpreter.interpret("group them")
    print(f"   Group result: {group_result}")
    
    # Step 3: Try to name the assembly - this is what we need to debug
    print("\n🏷️  Attempting to name the assembly...")
    
    # Test various naming syntaxes
    naming_commands = [
        "name that 'arch'",
        "call that an 'arch'",
        "name it 'arch'",
        "call it an 'arch'"
    ]
    
    for command in naming_commands:
        print(f"\n   Testing: {command}")
        try:
            name_result = interpreter.interpret(command)
            print(f"   Result: {name_result}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Step 4: Check final scene state
    print("\n📊 Final scene state:")
    print(f"   Standalone objects: {len(interpreter.scene.objects)}")
    print(f"   Assemblies: {len(interpreter.scene.assemblies) if hasattr(interpreter.scene, 'assemblies') else 0}")
    
    if hasattr(interpreter.scene, 'assemblies') and interpreter.scene.assemblies:
        for assembly in interpreter.scene.assemblies:
            assembly_name = getattr(assembly, 'name', 'unnamed')
            print(f"   Assembly '{assembly.assembly_id}' (name: {assembly_name}): {len(assembly.objects)} objects")


def test_direct_named_grouping():
    """Test creating named assemblies directly in one command."""
    print("\n🧪 Testing Direct Named Assembly Creation")
    print("=" * 50)
    
    # Create interpreter with mock renderer for testing
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Step 1: Create some objects to group
    print("\n📦 Creating objects to group...")
    result1 = interpreter.interpret("draw a very tall box at [-1,0,0]")
    result2 = interpreter.interpret("draw a very tall box at [1,0,0]")
    result3 = interpreter.interpret("draw a very tall box at [0,2,0]")
    
    # Step 2: Try to group them with a name in one command
    print("\n🔗 Attempting direct named grouping...")
    
    direct_naming_commands = [
        "group them as an 'arch'",
        "group them into an 'arch'",
        "make them into an 'arch'",
        "organize them as an 'arch'"
    ]
    
    for command in direct_naming_commands:
        print(f"\n   Testing: {command}")
        try:
            result = interpreter.interpret(command)
            print(f"   Result: {result}")
        except Exception as e:
            print(f"   Error: {e}")


if __name__ == "__main__":
    test_assembly_naming()
    test_direct_named_grouping()
