"""
Unit tests for SceneModel with SceneAssembly support.
Tests the enhanced scene model functionality including assembly management.
"""

import pytest
from engraf.visualizer.scene.scene_model import SceneModel, resolve_pronoun
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.scene.scene_assembly import SceneAssembly
from engraf.lexer.vector_space import VectorSpace


class TestSceneModelWithAssemblies:
    """Test the enhanced SceneModel class with assembly support."""

    def test_scene_model_initialization(self):
        """Test that SceneModel initializes with empty collections."""
        scene = SceneModel()
        
        assert scene.objects == []
        assert scene.assemblies == []
        assert scene.recent == []

    def test_add_object(self):
        """Test adding a standalone object to the scene."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        scene.add_object(cube)
        
        assert len(scene.objects) == 1
        assert cube in scene.objects
        assert len(scene.assemblies) == 0
        assert scene.recent == [cube]

    def test_add_assembly(self):
        """Test adding an assembly to the scene."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        assembly = SceneAssembly("house", objects=[cube], assembly_id="house_1")
        
        scene.add_assembly(assembly)
        
        assert len(scene.assemblies) == 1
        assert assembly in scene.assemblies
        assert len(scene.objects) == 0  # Objects are in assembly, not standalone
        assert scene.recent == [assembly]

    def test_get_all_scene_objects(self):
        """Test getting all objects from both standalone and assemblies."""
        scene = SceneModel()
        
        # Add standalone object
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        # Add assembly with objects
        sphere_vector = VectorSpace()
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        pyramid_vector = VectorSpace()
        pyramid = SceneObject("pyramid", pyramid_vector, object_id="pyramid_1")
        assembly = SceneAssembly("house", objects=[sphere, pyramid], assembly_id="house_1")
        scene.add_assembly(assembly)
        
        all_objects = scene.get_all_scene_objects()
        
        assert len(all_objects) == 3
        assert cube in all_objects
        assert sphere in all_objects
        assert pyramid in all_objects

    def test_find_object_by_id_standalone(self):
        """Test finding a standalone object by ID."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        found = scene.find_object_by_id("cube_1")
        assert found is cube
        
        not_found = scene.find_object_by_id("nonexistent")
        assert not_found is None

    def test_find_object_by_id_in_assembly(self):
        """Test finding an object inside an assembly by ID."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        assembly = SceneAssembly("house", objects=[cube], assembly_id="house_1")
        scene.add_assembly(assembly)
        
        found = scene.find_object_by_id("cube_1")
        assert found is cube

    def test_find_assembly_by_id(self):
        """Test finding an assembly by ID."""
        scene = SceneModel()
        
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        found = scene.find_assembly_by_id("house_1")
        assert found is assembly
        
        not_found = scene.find_assembly_by_id("nonexistent")
        assert not_found is None

    def test_find_assembly_by_name(self):
        """Test finding an assembly by name."""
        scene = SceneModel()
        
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        found = scene.find_assembly_by_name("house")
        assert found is assembly
        
        not_found = scene.find_assembly_by_name("nonexistent")
        assert not_found is None

    def test_remove_standalone_object(self):
        """Test removing a standalone object."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        # Verify object is there
        assert len(scene.objects) == 1
        
        # Remove it
        result = scene.remove_object("cube_1")
        
        assert result is True
        assert len(scene.objects) == 0

    def test_remove_object_from_assembly(self):
        """Test removing an object from within an assembly."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        sphere_vector = VectorSpace()
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube, sphere], assembly_id="house_1")
        scene.add_assembly(assembly)
        
        # Verify objects are there
        assert len(assembly.objects) == 2
        
        # Remove one object
        result = scene.remove_object("cube_1")
        
        assert result is True
        assert len(assembly.objects) == 1
        assert cube not in assembly.objects
        assert sphere in assembly.objects

    def test_remove_nonexistent_object(self):
        """Test removing an object that doesn't exist."""
        scene = SceneModel()
        
        result = scene.remove_object("nonexistent")
        
        assert result is False

    def test_remove_assembly(self):
        """Test removing an entire assembly."""
        scene = SceneModel()
        
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        # Verify assembly is there
        assert len(scene.assemblies) == 1
        
        # Remove it
        result = scene.remove_assembly("house_1")
        
        assert result is True
        assert len(scene.assemblies) == 0

    def test_remove_nonexistent_assembly(self):
        """Test removing an assembly that doesn't exist."""
        scene = SceneModel()
        
        result = scene.remove_assembly("nonexistent")
        
        assert result is False

    def test_move_object_to_assembly(self):
        """Test moving a standalone object into an assembly."""
        scene = SceneModel()
        
        # Create standalone object
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        # Create assembly
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        # Initially
        assert len(scene.objects) == 1
        assert len(assembly.objects) == 0
        
        # Move object to assembly
        result = scene.move_object_to_assembly("cube_1", "house_1")
        
        assert result is True
        assert len(scene.objects) == 0
        assert len(assembly.objects) == 1
        assert cube in assembly.objects

    def test_move_object_to_assembly_failures(self):
        """Test failure cases for moving object to assembly."""
        scene = SceneModel()
        
        # Test with nonexistent object
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        result = scene.move_object_to_assembly("nonexistent", "house_1")
        assert result is False
        
        # Test with nonexistent assembly
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        result = scene.move_object_to_assembly("cube_1", "nonexistent")
        assert result is False

    def test_extract_object_from_assembly(self):
        """Test extracting an object from assembly to make it standalone."""
        scene = SceneModel()
        
        # Create assembly with object
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        assembly = SceneAssembly("house", objects=[cube], assembly_id="house_1")
        scene.add_assembly(assembly)
        
        # Initially
        assert len(scene.objects) == 0
        assert len(assembly.objects) == 1
        
        # Extract object
        result = scene.extract_object_from_assembly("cube_1")
        
        assert result is True
        assert len(scene.objects) == 1
        assert len(assembly.objects) == 0
        assert cube in scene.objects

    def test_extract_nonexistent_object_from_assembly(self):
        """Test extracting an object that doesn't exist in any assembly."""
        scene = SceneModel()
        
        result = scene.extract_object_from_assembly("nonexistent")
        
        assert result is False

    def test_find_noun_phrase_assembly_priority(self):
        """Test that assemblies have priority in noun phrase matching."""
        scene = SceneModel()
        
        # Create standalone object
        cube_vector = VectorSpace()
        cube_vector['noun'] = 1.0
        cube = SceneObject("house", cube_vector, object_id="house_object")  # Named "house" too
        scene.add_object(cube)
        
        # Create assembly with same name
        assembly_vector = VectorSpace()
        assembly_vector['noun'] = 1.0
        assembly = SceneAssembly("house", assembly_id="house_1")
        assembly.vector = assembly_vector  # Override computed vector for test
        scene.add_assembly(assembly)
        
        # Mock noun phrase
        class MockNP:
            def __init__(self, noun, vector=None):
                self.noun = noun
                self.vector = vector
        
        np = MockNP("house")
        found = scene.find_noun_phrase(np)
        
        # Should find assembly first (assemblies are searched before objects)
        assert found is assembly

    def test_find_noun_phrase_object_in_assembly(self):
        """Test finding individual objects within assemblies."""
        scene = SceneModel()
        
        # Create assembly with cube
        cube_vector = VectorSpace()
        cube_vector['noun'] = 1.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        assembly = SceneAssembly("house", objects=[cube], assembly_id="house_1")
        scene.add_assembly(assembly)
        
        # Mock noun phrase
        class MockNP:
            def __init__(self, noun, vector=None):
                self.noun = noun
                self.vector = vector
        
        np = MockNP("cube")
        found = scene.find_noun_phrase(np)
        
        # Should find the cube inside the assembly
        assert found is cube

    def test_scene_repr_with_assemblies(self):
        """Test string representation of scene with assemblies."""
        scene = SceneModel()
        
        # Add standalone object
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        # Add assembly
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        repr_str = repr(scene)
        
        assert "Objects:" in repr_str
        assert "Assemblies:" in repr_str
        assert "cube" in repr_str
        assert "house" in repr_str

    def test_empty_scene_repr(self):
        """Test string representation of empty scene."""
        scene = SceneModel()
        
        repr_str = repr(scene)
        
        assert repr_str == "Empty scene"

    def test_scene_copy_with_assemblies(self):
        """Test deep copying a scene with assemblies."""
        scene = SceneModel()
        
        # Add standalone object
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        # Add assembly with object
        sphere_vector = VectorSpace()
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        assembly = SceneAssembly("house", objects=[sphere], assembly_id="house_1")
        scene.add_assembly(assembly)
        
        # Set recent to mix of objects and assemblies
        scene.recent = [cube, assembly]
        
        # Copy scene
        copied_scene = scene.copy()
        
        # Verify structure is preserved
        assert len(copied_scene.objects) == 1
        assert len(copied_scene.assemblies) == 1
        assert len(copied_scene.recent) == 2
        
        # Verify objects are different instances
        copied_cube = copied_scene.objects[0]
        copied_assembly = copied_scene.assemblies[0]
        copied_sphere = copied_assembly.objects[0]
        
        assert copied_cube is not cube
        assert copied_assembly is not assembly
        assert copied_sphere is not sphere
        
        # But have same properties
        assert copied_cube.object_id == cube.object_id
        assert copied_assembly.assembly_id == assembly.assembly_id
        assert copied_sphere.object_id == sphere.object_id


class TestResolvePronounWithAssemblies:
    """Test pronoun resolution with assembly support."""
    
    def test_resolve_it_with_recent_object(self):
        """Test resolving 'it' to most recent object."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        result = resolve_pronoun("it", scene)
        
        assert len(result) == 1
        assert result[0] is cube

    def test_resolve_it_with_recent_assembly(self):
        """Test resolving 'it' to most recent assembly."""
        scene = SceneModel()
        
        # Add object first
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        # Then add assembly (becomes most recent)
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        result = resolve_pronoun("it", scene)
        
        assert len(result) == 1
        assert result[0] is assembly

    def test_resolve_they_with_mixed_items(self):
        """Test resolving 'they' to all objects and assemblies."""
        scene = SceneModel()
        
        # Add standalone object
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        # Add assembly
        assembly = SceneAssembly("house", assembly_id="house_1")
        scene.add_assembly(assembly)
        
        result = resolve_pronoun("they", scene)
        
        assert len(result) == 2
        assert cube in result
        assert assembly in result

    def test_resolve_them_same_as_they(self):
        """Test that 'them' resolves the same as 'they'."""
        scene = SceneModel()
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        scene.add_object(cube)
        
        result_they = resolve_pronoun("they", scene)
        result_them = resolve_pronoun("them", scene)
        
        assert result_they == result_them

    def test_resolve_it_empty_scene(self):
        """Test resolving 'it' in empty scene."""
        scene = SceneModel()
        
        result = resolve_pronoun("it", scene)
        
        assert result == []

    def test_resolve_invalid_pronoun(self):
        """Test resolving invalid pronoun raises error."""
        scene = SceneModel()
        
        with pytest.raises(ValueError, match="Unrecognized pronoun"):
            resolve_pronoun("xyz", scene)
