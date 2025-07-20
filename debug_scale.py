#!/usr/bin/env python3
import sys
import traceback
sys.path.insert(0, '/Users/jeff/Python/Engraf')

from engraf.interpreter.sentence_interpreter import SentenceInterpreter

# Mock renderer for testing
class MockVPythonRenderer:
    def __init__(self):
        self.objects = {}
        self.updates = []
    
    def update_object(self, obj_name, properties):
        self.updates.append((obj_name, properties))
        print(f"🔧 Renderer: Update {obj_name} with {properties}")

# Create interpreter with test renderer
interpreter = SentenceInterpreter()
interpreter.renderer = MockVPythonRenderer()

# Create object
try:
    result = interpreter.interpret('draw a red cube')
    print('Object created successfully')
    print('Objects:', list(interpreter.scene.objects.keys()))
except Exception as e:
    print('Error creating object:', e)
    traceback.print_exc()

# Test scale
try:
    result = interpreter.interpret('scale the cube by [2, 2, 2]')
    print('Scale result:', result)
except Exception as e:
    print('Error during scale:', e)
    traceback.print_exc()
