"""
Unit tests for SceneModel class.
Tests the core scene management functionality without parsing dependencies.
"""

import pytest
from engraf.visualizer.scene.scene_model import SceneModel, resolve_pronoun
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


class TestSceneModel:
    """Test the SceneModel class functionality."""

    def test_init(self):
        """Test SceneModel initialization."""
        scene = SceneModel()
        assert scene.objects == []
        assert scene.recent == []

    def test_add_object(self):
        """Test adding objects to the scene."""
        scene = SceneModel()
        obj = SceneObject(name="cube", vector=VectorSpace())
        
        scene.add_object(obj)
        
        assert len(scene.objects) == 1
        assert scene.objects[0] == obj
        assert scene.recent == [obj]

    def test_add_multiple_objects(self):
        """Test adding multiple objects updates recent correctly."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace())
        obj2 = SceneObject(name="sphere", vector=VectorSpace())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        assert len(scene.objects) == 2
        assert scene.objects == [obj1, obj2]
        assert scene.recent == [obj2]  # Only most recent

    def test_get_recent_objects_default(self):
        """Test getting recent objects with default count."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace())
        obj2 = SceneObject(name="sphere", vector=VectorSpace())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        recent = scene.get_recent_objects()
        assert recent == [obj2]

    def test_get_recent_objects_with_count(self):
        """Test getting recent objects with specific count."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace())
        obj2 = SceneObject(name="sphere", vector=VectorSpace())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        # Test with count
        recent = scene.get_recent_objects(count=1)
        assert recent == [obj2]

    def test_find_noun_phrase_by_name(self):
        """Test finding objects by noun name."""
        scene = SceneModel()
        cube_vector = VectorSpace()
        sphere_vector = VectorSpace()
        
        cube = SceneObject(name="cube", vector=cube_vector)
        sphere = SceneObject(name="sphere", vector=sphere_vector)
        
        scene.add_object(cube)
        scene.add_object(sphere)
        
        # Mock noun phrase with just name
        class MockNP:
            def __init__(self, noun, vector=None):
                self.noun = noun
                self.vector = vector
        
        result = scene.find_noun_phrase(MockNP("cube"), return_all_matches=False)
        assert result == cube
        
        result = scene.find_noun_phrase(MockNP("sphere"), return_all_matches=False)
        assert result == sphere

    def test_find_noun_phrase_not_found(self):
        """Test finding non-existent object returns None."""
        scene = SceneModel()
        cube = SceneObject(name="cube", vector=VectorSpace())
        scene.add_object(cube)
        
        class MockNP:
            def __init__(self, noun, vector=None):
                self.noun = noun
                self.vector = vector
        
        result = scene.find_noun_phrase(MockNP("nonexistent"), return_all_matches=False)
        assert result is None

    def test_find_noun_phrase_by_vector_similarity(self):
        """Test finding objects by vector similarity."""
        scene = SceneModel()
        
        # Create objects with different vectors
        red_vector = VectorSpace()
        red_vector["red"] = 1.0
        red_cube = SceneObject(name="cube", vector=red_vector)
        
        blue_vector = VectorSpace()
        blue_vector["blue"] = 1.0
        blue_cube = SceneObject(name="cube", vector=blue_vector)
        
        scene.add_object(red_cube)
        scene.add_object(blue_cube)
        
        # Search for red cube
        search_vector = VectorSpace()
        search_vector["red"] = 1.0
        
        class MockNP:
            def __init__(self, noun, vector):
                self.noun = noun
                self.vector = vector
        
        result = scene.find_noun_phrase(MockNP("cube", search_vector), return_all_matches=False)
        assert result == red_cube  # Should find the red cube due to vector similarity

    def test_repr(self):
        """Test string representation of SceneModel."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace())
        obj2 = SceneObject(name="sphere", vector=VectorSpace())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        repr_str = repr(scene)
        assert "cube" in repr_str
        assert "sphere" in repr_str
        assert "\n" in repr_str  # Should join with newlines


class TestResolvePronoun:
    """Test the resolve_pronoun function."""

    def test_resolve_it_empty_scene(self):
        """Test resolving 'it' in empty scene."""
        scene = SceneModel()
        result = resolve_pronoun("it", scene)
        assert result == []

    def test_resolve_it_single_object(self):
        """Test resolving 'it' with single object."""
        scene = SceneModel()
        obj = SceneObject(name="cube", vector=VectorSpace())
        scene.add_object(obj)
        
        result = resolve_pronoun("it", scene)
        assert result == [obj]

    def test_resolve_it_multiple_objects(self):
        """Test resolving 'it' with multiple objects returns most recent."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace())
        obj2 = SceneObject(name="sphere", vector=VectorSpace())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        result = resolve_pronoun("it", scene)
        assert result == [obj2]  # Most recent

    def test_resolve_they_multiple_objects(self):
        """Test resolving 'they' returns all objects."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace())
        obj2 = SceneObject(name="sphere", vector=VectorSpace())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        result = resolve_pronoun("they", scene)
        assert result == [obj1, obj2]

    def test_resolve_them_multiple_objects(self):
        """Test resolving 'them' returns all objects."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace())
        obj2 = SceneObject(name="sphere", vector=VectorSpace())
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        result = resolve_pronoun("them", scene)
        assert result == [obj1, obj2]

    def test_resolve_case_insensitive(self):
        """Test that pronoun resolution is case insensitive."""
        scene = SceneModel()
        obj = SceneObject(name="cube", vector=VectorSpace())
        scene.add_object(obj)
        
        assert resolve_pronoun("IT", scene) == [obj]
        assert resolve_pronoun("It", scene) == [obj]
        assert resolve_pronoun("THEY", scene) == [obj]

    def test_resolve_unknown_pronoun(self):
        """Test that unknown pronouns raise ValueError."""
        scene = SceneModel()
        
        with pytest.raises(ValueError, match="Unrecognized pronoun"):
            resolve_pronoun("unknown", scene)


class TestSceneModelCopy:
    """Test the SceneModel copy functionality."""
    
    def test_copy_empty_scene(self):
        """Test copying an empty scene."""
        scene = SceneModel()
        copied_scene = scene.copy()
        
        assert copied_scene is not scene
        assert copied_scene.objects == []
        assert copied_scene.recent == []
        
        # Verify they are independent
        obj = SceneObject(name="cube", vector=VectorSpace(word="test"))
        scene.add_object(obj)
        
        assert len(scene.objects) == 1
        assert len(copied_scene.objects) == 0
    
    def test_copy_scene_with_objects(self):
        """Test copying a scene with objects."""
        scene = SceneModel()
        
        # Add some objects
        obj1 = SceneObject(name="cube", vector=VectorSpace(word="test1"), object_id="obj1")
        obj2 = SceneObject(name="sphere", vector=VectorSpace(word="test2"), object_id="obj2")
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        # Copy the scene
        copied_scene = scene.copy()
        
        # Verify structure
        assert copied_scene is not scene
        assert len(copied_scene.objects) == 2
        assert len(copied_scene.recent) == 1  # recent contains last added object
        
        # Verify objects are deep copied
        assert copied_scene.objects[0] is not scene.objects[0]
        assert copied_scene.objects[1] is not scene.objects[1]
        assert copied_scene.recent[0] is not scene.recent[0]
        
        # Verify object data is preserved
        assert copied_scene.objects[0].object_id == "obj1"
        assert copied_scene.objects[0].name == "cube"
        assert copied_scene.objects[1].object_id == "obj2"
        assert copied_scene.objects[1].name == "sphere"
        assert copied_scene.recent[0].object_id == "obj2"
    
    def test_copy_independence(self):
        """Test that copied scenes are completely independent."""
        scene = SceneModel()
        obj1 = SceneObject(name="cube", vector=VectorSpace(word="test1"), object_id="obj1")
        scene.add_object(obj1)
        
        # Copy the scene
        copied_scene = scene.copy()
        
        # Modify original scene
        obj2 = SceneObject(name="sphere", vector=VectorSpace(word="test2"), object_id="obj2")
        scene.add_object(obj2)
        
        # Modify object in original scene
        scene.objects[0].name = "modified_cube"
        
        # Copied scene should be unaffected
        assert len(copied_scene.objects) == 1
        assert copied_scene.objects[0].object_id == "obj1"
        assert copied_scene.objects[0].name == "cube"  # Not modified
        
        # Modify copied scene
        obj3 = SceneObject(name="cylinder", vector=VectorSpace(word="test3"), object_id="obj3")
        copied_scene.add_object(obj3)
        
        # Original scene should be unaffected by changes to copy
        assert len(scene.objects) == 2
        assert scene.objects[0].name == "modified_cube"
        assert scene.objects[1].object_id == "obj2"
    
    def test_copy_vector_space_independence(self):
        """Test that vector spaces in copied objects are independent."""
        scene = SceneModel()
        
        # Create object with vector space
        vector = VectorSpace(word="test")
        vector["red"] = 1.0
        vector["scaleX"] = 2.0
        obj = SceneObject(name="cube", vector=vector, object_id="obj1")
        scene.add_object(obj)
        
        # Copy the scene
        copied_scene = scene.copy()
        
        # Modify vector in original
        scene.objects[0].vector["red"] = 0.5
        scene.objects[0].vector["blue"] = 1.0
        
        # Copied object's vector should be unaffected
        copied_vector = copied_scene.objects[0].vector
        assert copied_vector["red"] == 1.0
        assert copied_vector["blue"] == 0.0
        assert copied_vector["scaleX"] == 2.0
    
    def test_copy_recent_object_mapping(self):
        """Test that recent objects list properly maps to copied objects."""
        scene = SceneModel()
        
        # Add multiple objects
        obj1 = SceneObject(name="cube", vector=VectorSpace(word="test1"), object_id="obj1")
        obj2 = SceneObject(name="sphere", vector=VectorSpace(word="test2"), object_id="obj2")
        obj3 = SceneObject(name="cylinder", vector=VectorSpace(word="test3"), object_id="obj3")
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        scene.add_object(obj3)
        
        # Copy the scene
        copied_scene = scene.copy()
        
        # Verify recent object is properly mapped to copied version
        assert copied_scene.recent[0] is copied_scene.objects[2]  # Should reference copied obj3
        assert copied_scene.recent[0] is not scene.recent[0]     # Should not reference original
        assert copied_scene.recent[0].object_id == "obj3"        # Should have same object_id
