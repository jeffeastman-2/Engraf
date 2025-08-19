import pytest
from engraf.lexer.vector_space import VectorSpace
from engraf.visualizer.scene.scene_object import SceneObject, scene_object_from_np


class TestSceneObject:
    def test_init_basic(self):
        """Test basic SceneObject initialization."""
        vector = VectorSpace()
        obj = SceneObject(name="cube", vector=vector)
        
        assert obj.name == "cube"
        assert obj.vector == vector
        assert obj.object_id == "cube"  # defaults to name

    def test_init_with_object_id(self):
        """Test SceneObject initialization with object_id."""
        vector = VectorSpace()
        obj = SceneObject(name="cube", vector=vector, object_id="red_cube_1")
        
        assert obj.name == "cube"
        assert obj.object_id == "red_cube_1"
        assert obj.vector == vector

    def test_repr(self):
        """Test string representation of SceneObject."""
        vector = VectorSpace()
        obj = SceneObject(name="cube", vector=vector)
        
        repr_str = repr(obj)
        assert "cube" in repr_str
        assert "pos=[0.0,0.0,0.0]" in repr_str

    def test_position_update(self):
        """Test that position updates work correctly."""
        vector = VectorSpace()
        obj = SceneObject(name="cube", vector=vector)
        
        obj.move_to(1.0, 2.0, 3.0)
        
        assert obj.position['x'] == 1.0
        assert obj.position['y'] == 2.0
        assert obj.position['z'] == 3.0
        assert obj.vector['locX'] == 1.0
        assert obj.vector['locY'] == 2.0
        assert obj.vector['locZ'] == 3.0


class TestSceneObjectFromNP:
    def test_simple_noun_phrase(self):
        """Test creating SceneObject from simple noun phrase."""
        # Mock a simple noun phrase
        class MockNP:
            def __init__(self, noun, vector):
                self.noun = noun
                self.vector = vector
        
        vector = VectorSpace()
        np = MockNP("cube", vector)
        
        obj = scene_object_from_np(np)
        
        assert isinstance(obj, SceneObject)
        assert obj.name == "cube"
        assert obj.vector == vector
