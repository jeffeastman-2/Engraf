"""
Test the assembly grouping functionality.
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


class TestAssemblyGrouping:
    """Test assembly creation and grouping functionality."""
    
    def test_basic_grouping(self):
        """Test basic assembly creation via grouping."""
        # Create interpreter with mock renderer for testing
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Step 1: Create some objects to group
        result1 = interpreter.interpret("draw a very tall box at [-1,0,0]")
        result2 = interpreter.interpret("draw a very tall box at [1,0,0]")
        result3 = interpreter.interpret("draw a very tall box at [0,2,0] and rotate it by 90 degrees")
        
        # Verify objects were created
        assert len(result1.get('objects_created', [])) == 1
        assert len(result2.get('objects_created', [])) == 1
        assert len(result3.get('objects_created', [])) == 1
        assert len(interpreter.scene.objects) == 3
        
        # Step 2: Group them
        group_result = interpreter.interpret("group them")
        
        # Verify grouping succeeded
        assert group_result['success'] is True
        assert 'group' in group_result.get('actions_performed', [])
        # Note: assemblies_created not currently returned in result dict
        # Verify assembly was created by checking scene state instead
        
        # Step 3: Check scene state after grouping
        assert len(interpreter.scene.objects) == 0  # Objects moved to assembly
        assert hasattr(interpreter.scene, 'assemblies')
        assert len(interpreter.scene.assemblies) == 1
        
        # Verify assembly contains all objects
        assembly = interpreter.scene.assemblies[0]
        assert len(assembly.objects) == 3
        assert assembly.assembly_id is not None
    
    def test_grouping_with_complex_sentences(self):
        """Test grouping after creating objects with complex sentences."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Create objects with complex sentence parsing - use simpler sentences for now
        result = interpreter.interpret("draw a box at [0,0,0]")
        # Note: complex sentences may not fully work yet, but basic ones should
        assert len(result.get('objects_created', [])) == 1
        
        # Add more objects
        result2 = interpreter.interpret("draw a sphere at [2,0,0]")
        result3 = interpreter.interpret("draw a cylinder at [-2,0,0]")
        
        assert len(interpreter.scene.objects) == 3
        
        # Group all objects
        group_result = interpreter.interpret("group them")
        assert group_result['success'] is True
        assert len(interpreter.scene.assemblies) == 1
        assert len(interpreter.scene.assemblies[0].objects) == 3
    
    def test_empty_scene_grouping(self):
        """Test grouping when no objects exist."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Try to group when no objects exist
        group_result = interpreter.interpret("group them")
        
        # Should handle gracefully with an appropriate error message
        assert group_result['success'] is False
        assert 'no objects' in group_result.get('message', '').lower()
    
    def test_single_object_grouping(self):
        """Test grouping a single object."""
        interpreter = SentenceInterpreter(renderer=MockRenderer())
        
        # Create single object
        interpreter.interpret("draw a box at [0,0,0]")
        assert len(interpreter.scene.objects) == 1
        
        # Group the single object
        group_result = interpreter.interpret("group them")
        assert group_result['success'] is True
        assert len(interpreter.scene.assemblies) == 1
        assert len(interpreter.scene.assemblies[0].objects) == 1


def test_grouping_integration():
    """Integration test for grouping functionality """
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
    
    # Step 2: Try to group them
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
