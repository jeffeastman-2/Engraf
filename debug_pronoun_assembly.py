#!/usr/bin/env python3
"""
Debug script to test pronoun resolution with assemblies
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer
from engraf.visualizer.scene.scene_model import resolve_pronoun

def test_pronoun_resolution():
    """Test pronoun resolution after assembly creation."""
    print("🔍 Testing pronoun resolution with assemblies...")
    print()
    
    # Create interpreter with mock renderer
    renderer = MockRenderer()
    interpreter = SentenceInterpreter(renderer=renderer)
    
    print("1. Creating objects...")
    result1 = interpreter.interpret("draw a tall box at [-1, 0, 0]")
    print(f"   Box1 result: {result1.get('success', False)}")
    
    result2 = interpreter.interpret("draw a tall box at [1, 0, 0]")
    print(f"   Box2 result: {result2.get('success', False)}")
    
    result3 = interpreter.interpret("draw a tall box at [0, 1, 0] and rotate it by 90 degrees")
    print(f"   Box3 result: {result3.get('success', False)}")
    
    print(f"\n   Scene objects: {len(interpreter.scene.objects)}")
    print(f"   Scene assemblies: {len(interpreter.scene.assemblies)}")
    print(f"   Recent items: {len(interpreter.scene.recent)}")
    if interpreter.scene.recent:
        print(f"   Most recent: {type(interpreter.scene.recent[-1]).__name__} - {getattr(interpreter.scene.recent[-1], 'name', 'no name')}")
    
    print("\n2. Testing pronoun resolution before grouping...")
    resolved_it = resolve_pronoun("it", interpreter.scene)
    print(f"   'it' resolves to: {len(resolved_it)} item(s)")
    if resolved_it:
        item = resolved_it[0]
        print(f"   Type: {type(item).__name__}")
        print(f"   Name: {getattr(item, 'name', 'no name')}")
        print(f"   ID: {getattr(item, 'object_id', getattr(item, 'assembly_id', 'no id'))}")
    
    print("\n3. Creating assembly...")
    group_result = interpreter.interpret("group them")
    print(f"   Group result: {group_result.get('success', False)}")
    print(f"   Message: {group_result.get('message', 'no message')}")
    
    print(f"\n   Scene objects after grouping: {len(interpreter.scene.objects)}")
    print(f"   Scene assemblies after grouping: {len(interpreter.scene.assemblies)}")
    print(f"   Recent items after grouping: {len(interpreter.scene.recent)}")
    if interpreter.scene.recent:
        print(f"   Most recent after grouping: {type(interpreter.scene.recent[-1]).__name__} - {getattr(interpreter.scene.recent[-1], 'name', 'no name')}")
        if hasattr(interpreter.scene.recent[-1], 'assembly_id'):
            print(f"   Assembly ID: {interpreter.scene.recent[-1].assembly_id}")
    
    print("\n4. Testing pronoun resolution after grouping...")
    resolved_it_after = resolve_pronoun("it", interpreter.scene)
    print(f"   'it' resolves to: {len(resolved_it_after)} item(s)")
    if resolved_it_after:
        item = resolved_it_after[0]
        print(f"   Type: {type(item).__name__}")
        print(f"   Name: {getattr(item, 'name', 'no name')}")
        print(f"   ID: {getattr(item, 'object_id', getattr(item, 'assembly_id', 'no id'))}")
        
        # Check if it's an assembly and has move methods
        if hasattr(item, 'move_to'):
            print(f"   ✅ Has move_to method")
        else:
            print(f"   ❌ Missing move_to method")
    
    print("\n5. Testing movement command...")
    move_result = interpreter.interpret("move it to [5, 5, 5]")
    print(f"   Move result: {move_result.get('success', False)}")
    print(f"   Message: {move_result.get('message', 'no message')}")
    
    if interpreter.scene.assemblies:
        assembly = interpreter.scene.assemblies[0]
        print(f"   Assembly position after move: [{assembly.vector['locX']}, {assembly.vector['locY']}, {assembly.vector['locZ']}]")

if __name__ == "__main__":
    test_pronoun_resolution()
