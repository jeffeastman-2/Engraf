#!/usr/bin/env python3

# Debug script to isolate the success flag issue

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer

def debug_assembly_creation():
    print("🔍 Debugging assembly creation result...")
    
    # Create interpreter with mock renderer
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create some objects
    print("\n1. Creating objects...")
    result1 = interpreter.interpret('draw a box at [0,0,0]')
    print(f"   Box result: success={result1.get('success')}")
    
    result2 = interpreter.interpret('draw a sphere at [1,0,0]') 
    print(f"   Sphere result: success={result2.get('success')}")
    
    result3 = interpreter.interpret('draw a cylinder at [2,0,0]')
    print(f"   Cylinder result: success={result3.get('success')}")
    
    # Try to group them
    print("\n2. Grouping objects...")
    try:
        result = interpreter.interpret('group them')
        print(f"   Group result type: {type(result)}")
        print(f"   Group result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"   Success: {result.get('success') if isinstance(result, dict) else 'N/A'}")
        print(f"   Message: {result.get('message') if isinstance(result, dict) else 'N/A'}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_assembly_creation()
