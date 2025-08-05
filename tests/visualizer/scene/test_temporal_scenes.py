"""
Tests for TemporalScenes class
"""

import pytest
from engraf.visualizer.scene.temporal_scenes import TemporalScenes
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


class TestTemporalScenes:
    
    def test_initialization_empty(self):
        """Test initialization with no arguments."""
        temporal = TemporalScenes()
        
        assert len(temporal) == 1
        assert temporal.get_current_index() == 0
        assert temporal.get_scene_count() == 1
        assert isinstance(temporal.get_current_scene(), SceneModel)
        assert len(temporal.get_current_scene().objects) == 0
    
    def test_initialization_with_scene(self):
        """Test initialization with an initial scene."""
        initial_scene = SceneModel()
        obj = SceneObject(name="cube", vector=VectorSpace(word="test"), object_id="test_object")
        initial_scene.add_object(obj)

        temporal = TemporalScenes(initial_scene)

        assert len(temporal) == 1
        assert temporal.get_current_index() == 0
        assert len(temporal.get_current_scene().objects) == 1
        assert temporal.get_current_scene().objects[0].object_id == "test_object"

    def test_add_scene_snapshot(self):
        """Test adding scene snapshots."""
        temporal = TemporalScenes()

        # Add first snapshot
        scene1 = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace(word="test1"), object_id="obj1")
        scene1.add_object(obj1)
        temporal.add_scene_snapshot(scene1)

        assert len(temporal) == 2
        assert temporal.get_current_index() == 1
        assert len(temporal.get_current_scene().objects) == 1
        assert temporal.get_current_scene().objects[0].object_id == "obj1"

        # Add second snapshot
        scene2 = SceneModel()
        obj2 = SceneObject(name="sphere", vector=VectorSpace(word="test2"), object_id="obj2")
        scene2.add_object(obj2)
        temporal.add_scene_snapshot(scene2)

        assert len(temporal) == 3
        assert temporal.get_current_index() == 2
        assert len(temporal.get_current_scene().objects) == 1
        assert temporal.get_current_scene().objects[0].object_id == "obj2"

    def test_go_back_forward(self):
        """Test basic temporal navigation."""
        temporal = TemporalScenes()

        # Add some scenes
        for i in range(3):
            scene = SceneModel()
            obj = SceneObject(name="cube", vector=VectorSpace(word=f"test{i}"), object_id=f"obj{i}")
            scene.add_object(obj)
            temporal.add_scene_snapshot(scene)

        # Should be at scene 3 (index 3)
        assert temporal.get_current_index() == 3
        assert len(temporal.get_current_scene().objects) == 1
        assert temporal.get_current_scene().objects[0].object_id == "obj2"

        # Go back one step
        success = temporal.go_back()
        assert success is True
        assert temporal.get_current_index() == 2
        assert temporal.get_current_scene().objects[0].object_id == "obj1"

        # Go back another step
        success = temporal.go_back()
        assert success is True
        assert temporal.get_current_index() == 1
        assert temporal.get_current_scene().objects[0].object_id == "obj0"

        # Go forward one step
        success = temporal.go_forward()
        assert success is True
        assert temporal.get_current_index() == 2
        assert temporal.get_current_scene().objects[0].object_id == "obj1"

        # Go forward another step
        success = temporal.go_forward()
        assert success is True
        assert temporal.get_current_index() == 3
        assert temporal.get_current_scene().objects[0].object_id == "obj2"

    def test_can_go_back_forward(self):
        """Test the can_go_back and can_go_forward methods."""
        temporal = TemporalScenes()
        
        # Initially can't go back or forward
        assert temporal.can_go_back() == False
        assert temporal.can_go_forward() == False
        
        # Add a scene
        scene = SceneModel()
        temporal.add_scene_snapshot(scene)
        
        # Now can go back but not forward
        assert temporal.can_go_back() == True
        assert temporal.can_go_forward() == False
        
        # Go back
        temporal.go_back()
        assert temporal.can_go_back() == False
        assert temporal.can_go_forward() == True
    
    def test_history_truncation(self):
        """Test that adding a scene while not at the end truncates future history."""
        temporal = TemporalScenes()
        
        # Add 3 scenes
        for i in range(3):
            scene = SceneModel()
            obj = SceneObject(name="cube", vector=VectorSpace(word=f"test{i}"), object_id=f"obj{i}")
            scene.add_object(obj)
            temporal.add_scene_snapshot(scene)
        
        assert len(temporal) == 4  # Initial + 3 added

        # Go back 2 steps (from index 3 to index 1)
        temporal.go_back()
        temporal.go_back()
        assert temporal.get_current_index() == 1

        # Add a new scene - this should truncate everything after index 1
        new_scene = SceneModel()
        new_obj = SceneObject(name="sphere", vector=VectorSpace(word="new_test"), object_id="new_obj")
        new_scene.add_object(new_obj)
        temporal.add_scene_snapshot(new_scene)

        # Should have truncated and added new scene
        assert len(temporal) == 3  # Initial + 1 kept + 1 new  
        assert temporal.get_current_index() == 2
        assert temporal.get_current_scene().objects[0].object_id == "new_obj"        # Can't go forward anymore
        assert temporal.can_go_forward() == False
    
    def test_get_scene_at_index(self):
        """Test getting scenes by index."""
        temporal = TemporalScenes()
        
        # Add some scenes
        for i in range(3):
            scene = SceneModel()
            obj = SceneObject(name="cube", vector=VectorSpace(word=f"test{i}"), object_id=f"obj{i}")
            scene.add_object(obj)
            temporal.add_scene_snapshot(scene)

        # Test valid indices
        scene0 = temporal.get_scene_at_index(0)
        assert scene0 is not None
        assert len(scene0.objects) == 0  # Initial empty scene

        scene1 = temporal.get_scene_at_index(1)
        assert scene1 is not None
        assert scene1.objects[0].object_id == "obj0"

        scene3 = temporal.get_scene_at_index(3)
        assert scene3 is not None
        assert scene3.objects[0].object_id == "obj2"        # Test invalid indices
        assert temporal.get_scene_at_index(-1) is None
        assert temporal.get_scene_at_index(10) is None
    
    def test_clear_history_keep_current(self):
        """Test clearing history while keeping current scene."""
        temporal = TemporalScenes()
        
        # Add scenes and navigate
        for i in range(3):
            scene = SceneModel()
            obj = SceneObject(name="cube", vector=VectorSpace(word=f"test{i}"), object_id=f"obj{i}")
            scene.add_object(obj)
            temporal.add_scene_snapshot(scene)

        temporal.go_back()  # Go back to scene with obj1
        current_obj_id = temporal.get_current_scene().objects[0].object_id        # Clear history keeping current
        temporal.clear_history(keep_current=True)
        
        assert len(temporal) == 1
        assert temporal.get_current_index() == 0
        assert temporal.get_current_scene().objects[0].object_id == current_obj_id
        assert temporal.can_go_back() == False
        assert temporal.can_go_forward() == False
    
    def test_clear_history_reset_to_empty(self):
        """Test clearing history and resetting to empty scene."""
        temporal = TemporalScenes()
        
        # Add scenes
        for i in range(3):
            scene = SceneModel()
            obj = SceneObject(f"obj{i}", "cube", VectorSpace(word=f"test{i}"))
            scene.add_object(obj)
            temporal.add_scene_snapshot(scene)
        
        # Clear history completely
        temporal.clear_history(keep_current=False)
        
        assert len(temporal) == 1
        assert temporal.get_current_index() == 0
        assert len(temporal.get_current_scene().objects) == 0
        assert temporal.can_go_back() == False
        assert temporal.can_go_forward() == False
    
    def test_scene_independence(self):
        """Test that scene snapshots are independent (deep copied)."""
        temporal = TemporalScenes()
        
        # Create initial scene
        scene = SceneModel()
        obj = SceneObject(name="cube", vector=VectorSpace(word="test"), object_id="test_obj")
        scene.add_object(obj)
        temporal.add_scene_snapshot(scene)

        # Modify the original scene
        obj2 = SceneObject(name="sphere", vector=VectorSpace(word="test2"), object_id="test_obj2")
        scene.add_object(obj2)

        # The snapshot should not be affected
        snapshot_scene = temporal.get_current_scene()
        assert len(snapshot_scene.objects) == 1
        assert snapshot_scene.objects[0].object_id == "test_obj"        # Original scene should have 2 objects
        assert len(scene.objects) == 2
    
    def test_repr(self):
        """Test string representation."""
        temporal = TemporalScenes()
        temporal.add_scene_snapshot(SceneModel())
        
        repr_str = repr(temporal)
        assert "TemporalScenes" in repr_str
        assert "scenes=2" in repr_str
        assert "current=1" in repr_str
