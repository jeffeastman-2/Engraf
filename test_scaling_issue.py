#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

from engraf.interpreter.sentence_interpreter import SentenceInterpreter

class MockRenderer:
    def __init__(self):
        self.objects = {}
    
    def render_object(self, obj):
        print(f'🎨 Creating {obj.name}:')
        print(f'   Position: {obj.position}')
        print(f'   Scale: {obj.scale}')
        print(f'   Vector scales: scaleX={obj.vector["scaleX"]}, scaleY={obj.vector["scaleY"]}, scaleZ={obj.vector["scaleZ"]}')
    
    def update_object(self, obj):
        print(f'🔄 Updating {obj.name}:')
        print(f'   Position: {obj.position}')
        print(f'   Scale: {obj.scale}')
        print(f'   Vector scales: scaleX={obj.vector["scaleX"]}, scaleY={obj.vector["scaleY"]}, scaleZ={obj.vector["scaleZ"]}')

# Test the scaling issue
interpreter = SentenceInterpreter(renderer=MockRenderer())

print('=== Creating cube ===')
result1 = interpreter.interpret('draw a red cube at [0, 0, 0]')
print(f'Success: {result1["success"]}')

print('\n=== Making it bigger ===')
result2 = interpreter.interpret('make it bigger')
print(f'Success: {result2["success"]}')

# Let's also check the scene object directly
cube_obj = list(interpreter.scene.objects.values())[0]
print(f'\n=== Direct object inspection ===')
print(f'Object name: {cube_obj.name}')
print(f'Object scale property: {cube_obj.scale}')
print(f'Vector scaleX: {cube_obj.vector["scaleX"]}')
print(f'Vector scaleY: {cube_obj.vector["scaleY"]}')
print(f'Vector scaleZ: {cube_obj.vector["scaleZ"]}')
