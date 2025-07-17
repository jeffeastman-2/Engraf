"""
Unit tests for SceneObject class.
Tests the core scene object functionality without parsing dependencies.
"""

import pytest
from engraf.visualizer.scene.scene_object import SceneObject, scene_object_from_np
from engraf.lexer.vector_space import VectorSpace


class TestSceneObject:
    """Test the SceneObject class functionality."""

    def test_init_basic(self):
        """Test basic SceneObject initialization."""
        vector = VectorSpace()
        obj = SceneObject(name="cube", vector=vector)
        
        assert obj.name == "cube"
        assert obj.vector == vector
        assert obj.modifiers == []

    def test_init_with_modifiers(self):
        """Test SceneObject initialization with modifiers."""
        vector = VectorSpace()
        modifier = SceneObject(name="sphere", vector=VectorSpace())
        
        obj = SceneObject(name="cube", vector=vector, modifiers=[modifier])
        
        assert obj.name == "cube"
        assert obj.vector == vector
        assert obj.modifiers == [modifier]

    def test_repr(self):
        """Test string representation of SceneObject."""
        vector = VectorSpace()
        obj = SceneObject(name="cube", vector=vector)
        
        repr_str = repr(obj)
        assert "cube" in repr_str
        assert "modifiers=[]" in repr_str

    def test_repr_with_modifiers(self):
        """Test string representation with modifiers."""
        vector = VectorSpace()
        modifier = SceneObject(name="sphere", vector=VectorSpace())
        obj = SceneObject(name="cube", vector=vector, modifiers=[modifier])
        
        repr_str = repr(obj)
        assert "cube" in repr_str
        assert "sphere" in repr_str


class TestSceneObjectFromNP:
    """Test the scene_object_from_np function."""

    def test_simple_noun_phrase(self):
        """Test creating SceneObject from simple noun phrase."""
        # Mock a simple noun phrase
        class MockNP:
            def __init__(self, noun, vector):
                self.noun = noun
                self.vector = vector
                self.preps = []
        
        vector = VectorSpace()
        np = MockNP("cube", vector)
        
        obj = scene_object_from_np(np)
        
        assert isinstance(obj, SceneObject)
        assert obj.name == "cube"
        assert obj.vector == vector
        assert obj.modifiers == []

    def test_noun_phrase_with_prepositional_phrase(self):
        """Test creating SceneObject from noun phrase with prepositional phrase."""
        # Mock structures
        class MockNP:
            def __init__(self, noun, vector, preps=None):
                self.noun = noun
                self.vector = vector
                self.preps = preps or []
        
        class MockPP:
            def __init__(self, noun_phrase):
                self.noun_phrase = noun_phrase
        
        # Create nested structure: "cube over the sphere"
        sphere_vector = VectorSpace()
        sphere_np = MockNP("sphere", sphere_vector)
        pp = MockPP(sphere_np)
        
        cube_vector = VectorSpace()
        cube_np = MockNP("cube", cube_vector, preps=[pp])
        
        obj = scene_object_from_np(cube_np)
        
        assert isinstance(obj, SceneObject)
        assert obj.name == "cube"
        assert obj.vector == cube_vector
        assert len(obj.modifiers) == 1
        
        modifier = obj.modifiers[0]
        assert isinstance(modifier, SceneObject)
        assert modifier.name == "sphere"
        assert modifier.vector == sphere_vector
        assert modifier.modifiers == []

    def test_noun_phrase_with_multiple_preps(self):
        """Test creating SceneObject from noun phrase with multiple prepositional phrases."""
        class MockNP:
            def __init__(self, noun, vector, preps=None):
                self.noun = noun
                self.vector = vector
                self.preps = preps or []
        
        class MockPP:
            def __init__(self, noun_phrase):
                self.noun_phrase = noun_phrase
        
        # Create structure: "cube over the sphere by the arch"
        sphere_vector = VectorSpace()
        sphere_np = MockNP("sphere", sphere_vector)
        pp1 = MockPP(sphere_np)
        
        arch_vector = VectorSpace()
        arch_np = MockNP("arch", arch_vector)
        pp2 = MockPP(arch_np)
        
        cube_vector = VectorSpace()
        cube_np = MockNP("cube", cube_vector, preps=[pp1, pp2])
        
        obj = scene_object_from_np(cube_np)
        
        assert isinstance(obj, SceneObject)
        assert obj.name == "cube"
        assert len(obj.modifiers) == 2
        
        # Check first modifier
        modifier1 = obj.modifiers[0]
        assert modifier1.name == "sphere"
        assert modifier1.vector == sphere_vector
        
        # Check second modifier
        modifier2 = obj.modifiers[1]
        assert modifier2.name == "arch"
        assert modifier2.vector == arch_vector

    def test_noun_phrase_with_nested_preps(self):
        """Test creating SceneObject from noun phrase with nested prepositional phrases."""
        class MockNP:
            def __init__(self, noun, vector, preps=None):
                self.noun = noun
                self.vector = vector
                self.preps = preps or []
        
        class MockPP:
            def __init__(self, noun_phrase):
                self.noun_phrase = noun_phrase
        
        # Create structure: "cube over the sphere by the arch"
        # where "sphere by the arch" means sphere has its own prep
        arch_vector = VectorSpace()
        arch_np = MockNP("arch", arch_vector)
        arch_pp = MockPP(arch_np)
        
        sphere_vector = VectorSpace()
        sphere_np = MockNP("sphere", sphere_vector, preps=[arch_pp])
        sphere_pp = MockPP(sphere_np)
        
        cube_vector = VectorSpace()
        cube_np = MockNP("cube", cube_vector, preps=[sphere_pp])
        
        obj = scene_object_from_np(cube_np)
        
        assert isinstance(obj, SceneObject)
        assert obj.name == "cube"
        assert len(obj.modifiers) == 2  # Flattened: sphere and arch
        
        # Should have both sphere and arch as modifiers (flattened)
        modifier_names = [mod.name for mod in obj.modifiers]
        assert "sphere" in modifier_names
        assert "arch" in modifier_names

    def test_noun_phrase_with_none_noun_in_prep(self):
        """Test handling prepositional phrases with None noun."""
        class MockNP:
            def __init__(self, noun, vector, preps=None):
                self.noun = noun
                self.vector = vector
                self.preps = preps or []
        
        class MockPP:
            def __init__(self, noun_phrase):
                self.noun_phrase = noun_phrase
        
        # Create a PP with None noun (should be ignored)
        none_np = MockNP(None, VectorSpace())
        pp = MockPP(none_np)
        
        cube_vector = VectorSpace()
        cube_np = MockNP("cube", cube_vector, preps=[pp])
        
        obj = scene_object_from_np(cube_np)
        
        assert isinstance(obj, SceneObject)
        assert obj.name == "cube"
        assert obj.modifiers == []  # None noun should be ignored
