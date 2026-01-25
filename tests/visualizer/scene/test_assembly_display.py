"""
Test assembly display and rendering functionality.
"""

import pytest
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer


class TestAssemblyDisplay:
    """Test assembly display and rendering functionality."""
    
    def test_assembly_appears_in_renderer(self):
        """Test that assemblies are properly sent to the renderer."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Create some objects
        interpreter.interpret("draw a box at [0,0,0]")
        interpreter.interpret("draw a sphere at [1,0,0]")
        interpreter.interpret("draw a cylinder at [2,0,0]")
        
        # Group them
        group_result = interpreter.interpret("group them")
        assert group_result['success'] is True
        
        # Trigger renderer update by calling render_scene
        interpreter.renderer.render_scene(interpreter.scene)
        
        # Check that the renderer received the assembly
        renderer = interpreter.renderer
        assert hasattr(renderer, 'assemblies')
        assert len(renderer.assemblies) == 1
        
        # The assembly should be visible in the scene
        assert len(interpreter.scene.assemblies) == 1
        assembly = interpreter.scene.assemblies[0]
        assert len(assembly.objects) == 3
    
    def test_assembly_vs_individual_objects_display(self):
        """Test that assemblies are displayed differently from individual objects."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Create individual objects first
        interpreter.interpret("draw a box at [0,0,0]")
        interpreter.interpret("draw a sphere at [1,0,0]")
        
        initial_objects = len(interpreter.scene.objects)
        assert initial_objects == 2
        
        # Group them
        interpreter.interpret("group them")
        
        # After grouping, objects should be moved to assembly
        assert len(interpreter.scene.objects) == 0  # Objects moved to assembly
        assert len(interpreter.scene.assemblies) == 1  # Assembly created
        assert len(interpreter.scene.assemblies[0].objects) == 2  # Contains the objects
    
    def test_multiple_assemblies_display(self):
        """Test displaying multiple assemblies in the same scene."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Create first group of objects
        interpreter.interpret("draw a box at [0,0,0]")
        interpreter.interpret("draw a sphere at [1,0,0]")
        interpreter.interpret("group them")
        
        first_assembly_count = len(interpreter.scene.assemblies)
        assert first_assembly_count == 1
        
        # Create second group of objects
        interpreter.interpret("draw a cylinder at [3,0,0]")
        interpreter.interpret("draw a cone at [4,0,0]")
        interpreter.interpret("group them")
        
        # Should now have two assemblies
        assert len(interpreter.scene.assemblies) == 2
        assert len(interpreter.scene.assemblies[0].objects) == 2
        assert len(interpreter.scene.assemblies[1].objects) == 2
    
    def test_assembly_properties_preservation(self):
        """Test that object properties are preserved when grouped into assemblies."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Create objects with specific properties
        result1 = interpreter.interpret("draw a box at [0,0,0]")
        result2 = interpreter.interpret("draw a sphere at [1,0,0]")
        result3 = interpreter.interpret("draw a cylinder at [2,0,0]")
        
        # Store original object IDs
        original_objects = [obj.object_id for obj in interpreter.scene.objects]
        assert len(original_objects) == 3
        
        # Group them
        group_result = interpreter.interpret("group them")
        assert group_result['success'] is True
        
        # Check that assembly contains the original objects
        assembly = interpreter.scene.assemblies[0]
        assembly_objects = assembly.objects  # This is a list, not a dict
        
        # Objects should be moved from scene to assembly
        assert len(assembly_objects) == 3
        
        # Check that the original object names/types are preserved in the assembly
        assembly_shapes = [obj.name for obj in assembly_objects]
        assert 'box' in assembly_shapes
        assert 'sphere' in assembly_shapes
        assert 'cylinder' in assembly_shapes
    
    def test_assembly_transformation_display(self):
        """Test that assemblies can be transformed and displayed correctly."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Create objects and group them
        interpreter.interpret("draw a box at [0,0,0]")
        interpreter.interpret("draw a sphere at [1,0,0]")
        interpreter.interpret("group them")
        
        # The assembly should be created
        assert len(interpreter.scene.assemblies) == 1
        assembly = interpreter.scene.assemblies[0]
        
        # Assembly should have an ID and contain objects
        assert assembly.assembly_id is not None
        assert len(assembly.objects) == 2
        
        # Future: Test transforming the assembly as a unit
        # This would require implementing assembly-level transformations
    
    def test_empty_assembly_handling(self):
        """Test how empty assemblies are handled in display."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Try to group when no objects exist
        group_result = interpreter.interpret("group them")
        
        # Should handle gracefully - either "no objects" or "failed to parse" 
        # (if pronoun can't resolve to any objects in empty scene)
        assert group_result['success'] is False
        msg = group_result.get('message', '').lower()
        assert 'no objects' in msg or 'failed to parse' in msg
        
        # No assemblies should be created
        assert len(interpreter.scene.assemblies) == 0


def test_assembly_display_integration():
    """Integration test for assembly display functionality."""
    print("🎨 Testing Assembly Display Integration")
    print("=" * 50)
    
    # Create interpreter with mock renderer for testing
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Step 1: Create objects for display testing
    print("\n📦 Creating objects for display...")
    result1 = interpreter.interpret("draw a box at [0,0,0]")
    result2 = interpreter.interpret("draw a sphere at [2,0,0]")
    result3 = interpreter.interpret("draw a cylinder at [4,0,0]")
    print(f"   Created {len(interpreter.scene.objects)} objects")
    
    # Step 2: Group them and check display
    print("\n🔗 Grouping objects...")
    group_result = interpreter.interpret("group them")
    print(f"   Grouping success: {group_result['success']}")
    
    # Step 3: Check display state
    print("\n🎨 Display state:")
    print(f"   Individual objects in scene: {len(interpreter.scene.objects)}")
    print(f"   Assemblies in scene: {len(interpreter.scene.assemblies)}")
    
    if interpreter.scene.assemblies:
        for i, assembly in enumerate(interpreter.scene.assemblies):
            print(f"   Assembly {i+1} ('{assembly.assembly_id}'): {len(assembly.objects)} objects")
    
    # Step 4: Test multiple assemblies
    print("\n📦 Creating second group...")
    interpreter.interpret("draw a cone at [6,0,0]")
    interpreter.interpret("draw a pyramid at [8,0,0]")
    interpreter.interpret("group them")
    
    print(f"   Total assemblies: {len(interpreter.scene.assemblies)}")
    for i, assembly in enumerate(interpreter.scene.assemblies):
        print(f"   Assembly {i+1}: {len(assembly.objects)} objects")
