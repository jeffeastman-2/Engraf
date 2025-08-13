"""
Unit tests for SceneAssembly class.
Tests the core scene assembly functionality including object containment,
transformations, and hierarchical operations.
"""

import pytest
import math
from engraf.visualizer.scene.scene_assembly import SceneAssembly
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


class TestSceneAssembly:
    """Test the SceneAssembly class functionality."""

    def test_empty_assembly_creation(self):
        """Test creating an empty assembly."""
        assembly = SceneAssembly("house")
        
        assert assembly.name == "house"
        assert assembly.assembly_id == "house"
        assert assembly.objects == []
        assert assembly.position == {'x': 0.0, 'y': 0.0, 'z': 0.0}
        assert assembly.rotation == {'x': 0.0, 'y': 0.0, 'z': 0.0}
        assert assembly.scale == {'x': 1.0, 'y': 1.0, 'z': 1.0}
        assert assembly.vector.isa('assembly')
        assert assembly.vector.isa('noun')

    def test_assembly_creation_with_objects(self):
        """Test creating an assembly with initial objects."""
        # Create test objects
        cube_vector = VectorSpace()
        cube_vector['locX'] = 1.0
        cube_vector['locY'] = 2.0
        cube_vector['locZ'] = 3.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        sphere_vector = VectorSpace()
        sphere_vector['locX'] = 4.0
        sphere_vector['locY'] = 5.0
        sphere_vector['locZ'] = 6.0
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        # Create assembly
        assembly = SceneAssembly("house", objects=[cube, sphere], assembly_id="house_1")
        
        assert assembly.name == "house"
        assert assembly.assembly_id == "house_1"
        assert len(assembly.objects) == 2
        assert cube in assembly.objects
        assert sphere in assembly.objects
        
        # Check centroid calculation
        expected_center_x = (1.0 + 4.0) / 2.0  # 2.5
        expected_center_y = (2.0 + 5.0) / 2.0  # 3.5
        expected_center_z = (3.0 + 6.0) / 2.0  # 4.5
        
        assert assembly.vector['locX'] == expected_center_x
        assert assembly.vector['locY'] == expected_center_y
        assert assembly.vector['locZ'] == expected_center_z

    def test_add_object(self):
        """Test adding an object to an assembly."""
        assembly = SceneAssembly("house")
        
        cube_vector = VectorSpace()
        cube_vector['locX'] = 1.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        # Initially empty
        assert len(assembly.objects) == 0
        
        # Add object
        assembly.add_object(cube)
        
        assert len(assembly.objects) == 1
        assert cube in assembly.objects
        assert assembly.vector['locX'] == 1.0  # Centroid updated

    def test_add_duplicate_object(self):
        """Test that adding the same object twice doesn't duplicate it."""
        assembly = SceneAssembly("house")
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        assembly.add_object(cube)
        assembly.add_object(cube)  # Add same object again
        
        assert len(assembly.objects) == 1  # Should still be 1

    def test_remove_object(self):
        """Test removing an object from an assembly."""
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        assembly = SceneAssembly("house", objects=[cube])
        
        # Initially has object
        assert len(assembly.objects) == 1
        assert cube in assembly.objects
        
        # Remove object
        result = assembly.remove_object(cube)
        
        assert result is True
        assert len(assembly.objects) == 0
        assert cube not in assembly.objects

    def test_remove_nonexistent_object(self):
        """Test removing an object that's not in the assembly."""
        assembly = SceneAssembly("house")
        
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        # Remove from empty assembly
        result = assembly.remove_object(cube)
        
        assert result is False
        assert len(assembly.objects) == 0

    def test_get_object_by_name(self):
        """Test finding objects by name within assembly."""
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        sphere_vector = VectorSpace()
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube, sphere])
        
        # Find existing objects
        found_cube = assembly.get_object_by_name("cube")
        found_sphere = assembly.get_object_by_name("sphere")
        
        assert found_cube is cube
        assert found_sphere is sphere
        
        # Try to find non-existent object
        not_found = assembly.get_object_by_name("pyramid")
        assert not_found is None

    def test_get_objects_by_type(self):
        """Test finding multiple objects of the same type."""
        # Create multiple cubes
        cube1_vector = VectorSpace()
        cube1 = SceneObject("cube", cube1_vector, object_id="cube_1")
        
        cube2_vector = VectorSpace()
        cube2 = SceneObject("cube", cube2_vector, object_id="cube_2")
        
        sphere_vector = VectorSpace()
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube1, cube2, sphere])
        
        # Find all cubes
        cubes = assembly.get_objects_by_type("cube")
        assert len(cubes) == 2
        assert cube1 in cubes
        assert cube2 in cubes
        assert sphere not in cubes
        
        # Find spheres
        spheres = assembly.get_objects_by_type("sphere")
        assert len(spheres) == 1
        assert sphere in spheres

    def test_get_object_by_id(self):
        """Test finding objects by ID within assembly."""
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        assembly = SceneAssembly("house", objects=[cube])
        
        found = assembly.get_object_by_id("cube_1")
        assert found is cube
        
        not_found = assembly.get_object_by_id("nonexistent")
        assert not_found is None

    def test_move_by(self):
        """Test moving assembly by deltas."""
        # Create objects at different positions
        cube_vector = VectorSpace()
        cube_vector['locX'] = 1.0
        cube_vector['locY'] = 1.0
        cube_vector['locZ'] = 1.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        sphere_vector = VectorSpace()
        sphere_vector['locX'] = 3.0
        sphere_vector['locY'] = 3.0
        sphere_vector['locZ'] = 3.0
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube, sphere])
        
        # Initial positions
        assert cube.vector['locX'] == 1.0
        assert sphere.vector['locX'] == 3.0
        
        # Move assembly
        assembly.move_by(5.0, 10.0, 15.0)
        
        # Check that both objects moved by the same delta
        assert cube.vector['locX'] == 6.0   # 1.0 + 5.0
        assert cube.vector['locY'] == 11.0  # 1.0 + 10.0
        assert cube.vector['locZ'] == 16.0  # 1.0 + 15.0
        
        assert sphere.vector['locX'] == 8.0   # 3.0 + 5.0
        assert sphere.vector['locY'] == 13.0  # 3.0 + 10.0
        assert sphere.vector['locZ'] == 18.0  # 3.0 + 15.0

    def test_move_to(self):
        """Test moving assembly to absolute position."""
        # Create objects
        cube_vector = VectorSpace()
        cube_vector['locX'] = 1.0
        cube_vector['locY'] = 1.0
        cube_vector['locZ'] = 1.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        sphere_vector = VectorSpace()
        sphere_vector['locX'] = 3.0
        sphere_vector['locY'] = 3.0
        sphere_vector['locZ'] = 3.0
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube, sphere])
        
        # Initial center should be (2.0, 2.0, 2.0)
        assert assembly.vector['locX'] == 2.0
        assert assembly.vector['locY'] == 2.0
        assert assembly.vector['locZ'] == 2.0
        
        # Move center to (10.0, 20.0, 30.0)
        assembly.move_to(10.0, 20.0, 30.0)
        
        # Objects should move by delta of (8.0, 18.0, 28.0)
        assert cube.vector['locX'] == 9.0   # 1.0 + 8.0
        assert cube.vector['locY'] == 19.0  # 1.0 + 18.0
        assert cube.vector['locZ'] == 29.0  # 1.0 + 28.0
        
        assert sphere.vector['locX'] == 11.0  # 3.0 + 8.0
        assert sphere.vector['locY'] == 21.0  # 3.0 + 18.0
        assert sphere.vector['locZ'] == 31.0  # 3.0 + 28.0
        
        # Assembly center should now be at target
        assert assembly.vector['locX'] == 10.0
        assert assembly.vector['locY'] == 20.0
        assert assembly.vector['locZ'] == 30.0

    def test_scale_by(self):
        """Test scaling assembly."""
        # Create objects at different positions with different scales
        cube_vector = VectorSpace()
        cube_vector['locX'] = 1.0
        cube_vector['scaleX'] = 2.0
        cube_vector['scaleY'] = 2.0
        cube_vector['scaleZ'] = 2.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        sphere_vector = VectorSpace()
        sphere_vector['locX'] = 3.0
        sphere_vector['scaleX'] = 1.0
        sphere_vector['scaleY'] = 1.0
        sphere_vector['scaleZ'] = 1.0
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube, sphere])
        
        # Assembly center at (2.0, 0.0, 0.0)
        # Scale by factor of 2.0
        assembly.scale_by(2.0, 2.0, 2.0)
        
        # Object scales should be doubled
        assert cube.vector['scaleX'] == 4.0   # 2.0 * 2.0
        assert sphere.vector['scaleX'] == 2.0  # 1.0 * 2.0
        
        # Object positions should be scaled relative to center
        # cube: (1.0 - 2.0) * 2.0 + 2.0 = -1.0 * 2.0 + 2.0 = 0.0
        # sphere: (3.0 - 2.0) * 2.0 + 2.0 = 1.0 * 2.0 + 2.0 = 4.0
        assert cube.vector['locX'] == 0.0
        assert sphere.vector['locX'] == 4.0

    def test_rotate_around_center_z_axis(self):
        """Test rotating assembly around Z-axis."""
        # Create two objects so assembly center is not at object position
        cube1_vector = VectorSpace()
        cube1_vector['locX'] = 1.0
        cube1_vector['locY'] = 0.0
        cube1_vector['locZ'] = 0.0
        cube1 = SceneObject("cube", cube1_vector, object_id="cube_1")
        
        cube2_vector = VectorSpace()
        cube2_vector['locX'] = -1.0
        cube2_vector['locY'] = 0.0
        cube2_vector['locZ'] = 0.0
        cube2 = SceneObject("cube", cube2_vector, object_id="cube_2")
        
        assembly = SceneAssembly("house", objects=[cube1, cube2])
        
        # Assembly center should be at (0, 0, 0)
        assert assembly.vector['locX'] == 0.0
        assert assembly.vector['locY'] == 0.0
        assert assembly.vector['locZ'] == 0.0
        
        # Rotate 90 degrees around Z-axis
        assembly.rotate_around_center(0.0, 0.0, 90.0)
        
        # cube1 should move from (1, 0, 0) to approximately (0, 1, 0)
        assert abs(cube1.vector['locX'] - 0.0) < 0.001
        assert abs(cube1.vector['locY'] - 1.0) < 0.001
        assert cube1.vector['locZ'] == 0.0
        
        # cube2 should move from (-1, 0, 0) to approximately (0, -1, 0)
        assert abs(cube2.vector['locX'] - 0.0) < 0.001
        assert abs(cube2.vector['locY'] - (-1.0)) < 0.001
        assert cube2.vector['locZ'] == 0.0
        
        # Objects' own rotation should be updated
        assert cube1.vector['rotZ'] == 90.0
        assert cube2.vector['rotZ'] == 90.0

    def test_bounding_box_calculation(self):
        """Test bounding box calculation."""
        # Create objects with known positions and sizes
        cube_vector = VectorSpace()
        cube_vector['locX'] = 0.0
        cube_vector['locY'] = 0.0
        cube_vector['locZ'] = 0.0
        cube_vector['scaleX'] = 2.0
        cube_vector['scaleY'] = 2.0
        cube_vector['scaleZ'] = 2.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        sphere_vector = VectorSpace()
        sphere_vector['locX'] = 5.0
        sphere_vector['locY'] = 5.0
        sphere_vector['locZ'] = 5.0
        sphere_vector['scaleX'] = 1.0
        sphere_vector['scaleY'] = 1.0
        sphere_vector['scaleZ'] = 1.0
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube, sphere])
        
        bbox = assembly.bounding_box
        
        # Cube extends from -1 to +1 in all axes (center 0, scale 2)
        # Sphere extends from 4.5 to 5.5 in all axes (center 5, scale 1)
        assert bbox['min_x'] == -1.0
        assert bbox['max_x'] == 5.5
        assert bbox['min_y'] == -1.0
        assert bbox['max_y'] == 5.5
        assert bbox['min_z'] == -1.0
        assert bbox['max_z'] == 5.5
        
        assert bbox['width'] == 6.5   # 5.5 - (-1.0)
        assert bbox['height'] == 6.5
        assert bbox['depth'] == 6.5

    def test_empty_assembly_bounding_box(self):
        """Test bounding box of empty assembly."""
        assembly = SceneAssembly("house")
        
        bbox = assembly.bounding_box
        
        assert bbox['min_x'] == 0
        assert bbox['max_x'] == 0
        assert bbox['width'] == 0
        assert bbox['height'] == 0
        assert bbox['depth'] == 0

    def test_update_transformations(self):
        """Test that update_transformations propagates to contained objects."""
        cube_vector = VectorSpace()
        cube_vector['locX'] = 1.0
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        assembly = SceneAssembly("house", objects=[cube])
        
        # Modify assembly vector directly
        assembly.vector['locX'] = 5.0
        assembly.vector['locY'] = 10.0
        
        # Call update transformations
        assembly.update_transformations()
        
        # Assembly position should be updated
        assert assembly.position['x'] == 5.0
        assert assembly.position['y'] == 10.0

    def test_assembly_repr(self):
        """Test string representation of assembly."""
        cube_vector = VectorSpace()
        cube = SceneObject("cube", cube_vector, object_id="cube_1")
        
        sphere_vector = VectorSpace()
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere_1")
        
        assembly = SceneAssembly("house", objects=[cube, sphere], assembly_id="house_1")
        
        repr_str = repr(assembly)
        
        assert "SceneAssembly" in repr_str
        assert "house" in repr_str
        assert "house_1" in repr_str
        assert "cube" in repr_str
        assert "sphere" in repr_str
        assert "center=" in repr_str
