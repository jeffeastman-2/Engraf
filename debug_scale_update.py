#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

from engraf.interpreter.sentence_interpreter import SentenceInterpreter

class MockRenderer:
    def render_object(self, obj): pass
    def update_object(self, obj): pass

# Test the scaling issue more precisely
interpreter = SentenceInterpreter(renderer=MockRenderer())

# Create cube
result1 = interpreter.interpret('draw a red cube at [0, 0, 0]')
cube_obj = interpreter.scene.objects[0]

print("=== After creating cube ===")
print(f"Vector scaleX: {cube_obj.vector['scaleX']}")
print(f"'scaleX' in vector: {'scaleX' in cube_obj.vector}")
print(f"SceneObject scale.x: {cube_obj.scale['x']}")

# Apply "make it bigger"
result2 = interpreter.interpret('make it bigger')

print("\n=== After making it bigger ===")
print(f"Vector scaleX: {cube_obj.vector['scaleX']}")
print(f"'scaleX' in vector: {'scaleX' in cube_obj.vector}")
print(f"SceneObject scale.x: {cube_obj.scale['x']}")

# Force an update to see if that helps
cube_obj.update_transformations()
print("\n=== After manual update_transformations ===")
print(f"Vector scaleX: {cube_obj.vector['scaleX']}")
print(f"'scaleX' in vector: {'scaleX' in cube_obj.vector}")
print(f"SceneObject scale.x: {cube_obj.scale['x']}")
