"""
Unit tests for TransformMatrix class.
Tests the 4×4 homogeneous matrix transformation system.
"""

import pytest
import numpy as np
from engraf.visualizer.transforms.transform_matrix import TransformMatrix


class TestTransformMatrix:
    """Test the TransformMatrix class functionality."""

    def test_init_default(self):
        """Test default initialization creates identity matrix."""
        transform = TransformMatrix()
        expected = np.eye(4)
        np.testing.assert_array_equal(transform.matrix, expected)

    def test_init_with_matrix(self):
        """Test initialization with custom matrix."""
        matrix = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [0, 0, 0, 1]
        ])
        transform = TransformMatrix(matrix)
        np.testing.assert_array_equal(transform.matrix, matrix.astype(np.float64))

    def test_init_invalid_shape(self):
        """Test initialization with invalid matrix shape raises error."""
        invalid_matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Matrix must be 4×4"):
            TransformMatrix(invalid_matrix)

    def test_identity(self):
        """Test identity matrix creation."""
        transform = TransformMatrix.identity()
        expected = np.eye(4)
        np.testing.assert_array_equal(transform.matrix, expected)

    def test_translation(self):
        """Test translation matrix creation."""
        transform = TransformMatrix.translation(1, 2, 3)
        expected = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(transform.matrix, expected)

    def test_rotation_x(self):
        """Test rotation around X-axis."""
        transform = TransformMatrix.rotation_x(90)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(transform.matrix, expected)

    def test_rotation_y(self):
        """Test rotation around Y-axis."""
        transform = TransformMatrix.rotation_y(90)
        expected = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(transform.matrix, expected)

    def test_rotation_z(self):
        """Test rotation around Z-axis."""
        transform = TransformMatrix.rotation_z(90)
        expected = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(transform.matrix, expected)

    def test_rotation_xyz(self):
        """Test combined XYZ rotation."""
        transform = TransformMatrix.rotation_xyz(90, 0, 0)
        expected_x = TransformMatrix.rotation_x(90)
        np.testing.assert_array_almost_equal(transform.matrix, expected_x.matrix)

    def test_scale(self):
        """Test scaling matrix creation."""
        transform = TransformMatrix.scale(2, 3, 4)
        expected = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(transform.matrix, expected)

    def test_uniform_scale(self):
        """Test uniform scaling matrix creation."""
        transform = TransformMatrix.uniform_scale(2)
        expected = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(transform.matrix, expected)

    def test_compose(self):
        """Test matrix composition."""
        translate = TransformMatrix.translation(1, 2, 3)
        scale = TransformMatrix.scale(2, 2, 2)
        
        # Compose: first scale, then translate
        result = translate.compose(scale)
        
        # Apply to a point to verify
        point = np.array([1, 1, 1])
        transformed = result.apply_to_point(point)
        
        # Should be: scale by 2, then translate by (1,2,3)
        expected = np.array([3, 4, 5])  # (1*2 + 1, 1*2 + 2, 1*2 + 3)
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_apply_to_point(self):
        """Test applying transformation to a point."""
        transform = TransformMatrix.translation(1, 2, 3)
        point = (4, 5, 6)
        
        result = transform.apply_to_point(point)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_to_point_numpy(self):
        """Test applying transformation to a numpy point."""
        transform = TransformMatrix.translation(1, 2, 3)
        point = np.array([4, 5, 6])
        
        result = transform.apply_to_point(point)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_to_vector(self):
        """Test applying transformation to a vector (no translation)."""
        transform = TransformMatrix.translation(1, 2, 3)
        vector = (4, 5, 6)
        
        result = transform.apply_to_vector(vector)
        expected = np.array([4, 5, 6])  # Translation should not affect vectors
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_to_vector_with_scale(self):
        """Test applying transformation with scale to a vector."""
        transform = TransformMatrix.scale(2, 3, 4)
        vector = (1, 1, 1)
        
        result = transform.apply_to_vector(vector)
        expected = np.array([2, 3, 4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_inverse(self):
        """Test matrix inversion."""
        transform = TransformMatrix.translation(1, 2, 3)
        inverse = transform.inverse()
        
        # Composing with inverse should give identity
        identity = transform.compose(inverse)
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(identity.matrix, expected)

    def test_inverse_rotation(self):
        """Test inverse of rotation matrix."""
        transform = TransformMatrix.rotation_z(45)
        inverse = transform.inverse()
        
        # Applying rotation then inverse should return original point
        point = np.array([1, 0, 0])
        rotated = transform.apply_to_point(point)
        restored = inverse.apply_to_point(rotated)
        
        np.testing.assert_array_almost_equal(restored, point)

    def test_decompose_translation(self):
        """Test decomposition of translation matrix."""
        transform = TransformMatrix.translation(1, 2, 3)
        translation, rotation, scale = transform.decompose()
        
        np.testing.assert_array_almost_equal(translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(rotation, [0, 0, 0])
        np.testing.assert_array_almost_equal(scale, [1, 1, 1])

    def test_decompose_scale(self):
        """Test decomposition of scale matrix."""
        transform = TransformMatrix.scale(2, 3, 4)
        translation, rotation, scale = transform.decompose()
        
        np.testing.assert_array_almost_equal(translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(rotation, [0, 0, 0])
        np.testing.assert_array_almost_equal(scale, [2, 3, 4])

    def test_decompose_rotation_z(self):
        """Test decomposition of Z rotation matrix."""
        transform = TransformMatrix.rotation_z(90)
        translation, rotation, scale = transform.decompose()
        
        np.testing.assert_array_almost_equal(translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(rotation, [0, 0, 90])
        np.testing.assert_array_almost_equal(scale, [1, 1, 1])

    def test_get_translation(self):
        """Test getting translation component."""
        transform = TransformMatrix.translation(1, 2, 3)
        translation = transform.get_translation()
        
        np.testing.assert_array_almost_equal(translation, [1, 2, 3])

    def test_get_scale(self):
        """Test getting scale component."""
        transform = TransformMatrix.scale(2, 3, 4)
        scale = transform.get_scale()
        
        np.testing.assert_array_almost_equal(scale, [2, 3, 4])

    def test_multiplication_operator(self):
        """Test matrix multiplication operator."""
        translate = TransformMatrix.translation(1, 2, 3)
        scale = TransformMatrix.scale(2, 2, 2)
        
        result = translate * scale
        expected = translate.compose(scale)
        
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_equality(self):
        """Test equality comparison."""
        transform1 = TransformMatrix.translation(1, 2, 3)
        transform2 = TransformMatrix.translation(1, 2, 3)
        transform3 = TransformMatrix.translation(1, 2, 4)
        
        assert transform1 == transform2
        assert transform1 != transform3
        assert transform1 != "not a transform"

    def test_repr(self):
        """Test string representation."""
        transform = TransformMatrix.identity()
        repr_str = repr(transform)
        
        assert "TransformMatrix(" in repr_str
        assert "[[1. 0. 0. 0.]" in repr_str

    def test_str(self):
        """Test human-readable string representation."""
        transform = TransformMatrix.translation(1, 2, 3)
        str_repr = str(transform)
        
        assert "Transform(" in str_repr
        assert "translation=[1. 2. 3.]" in str_repr

    def test_complex_transformation(self):
        """Test complex transformation combining multiple operations."""
        # Create a complex transformation: translate, then rotate, then scale
        translate = TransformMatrix.translation(1, 0, 0)
        rotate = TransformMatrix.rotation_z(90)
        scale = TransformMatrix.scale(2, 2, 2)
        
        # Compose in order: scale, then rotate, then translate
        complex_transform = translate.compose(rotate.compose(scale))
        
        # Apply to a point
        point = np.array([1, 0, 0])
        result = complex_transform.apply_to_point(point)
        
        # Expected: scale (2,0,0) -> rotate 90° -> (0,2,0) -> translate -> (1,2,0)
        expected = np.array([1, 2, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_chain_composition(self):
        """Test chaining multiple transformations."""
        t1 = TransformMatrix.translation(1, 0, 0)
        t2 = TransformMatrix.translation(0, 1, 0)
        t3 = TransformMatrix.translation(0, 0, 1)
        
        # Chain them together
        result = t1 * t2 * t3
        
        # Should be equivalent to translation(1, 1, 1)
        expected = TransformMatrix.translation(1, 1, 1)
        
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_rotation_order(self):
        """Test that rotation order matters."""
        # Rotate 90° around X, then 90° around Y
        rx_then_ry = TransformMatrix.rotation_y(90).compose(TransformMatrix.rotation_x(90))
        
        # Rotate 90° around Y, then 90° around X  
        ry_then_rx = TransformMatrix.rotation_x(90).compose(TransformMatrix.rotation_y(90))
        
        # These should be different
        assert not np.allclose(rx_then_ry.matrix, ry_then_rx.matrix)

    def test_rotation_by_zero(self):
        """Test rotation by zero degrees gives identity."""
        transform = TransformMatrix.rotation_x(0)
        identity = TransformMatrix.identity()
        
        np.testing.assert_array_almost_equal(transform.matrix, identity.matrix)

    def test_scale_by_one(self):
        """Test scaling by 1 gives identity."""
        transform = TransformMatrix.scale(1, 1, 1)
        identity = TransformMatrix.identity()
        
        np.testing.assert_array_almost_equal(transform.matrix, identity.matrix)

    def test_translation_by_zero(self):
        """Test translation by zero gives identity."""
        transform = TransformMatrix.translation(0, 0, 0)
        identity = TransformMatrix.identity()
        
        np.testing.assert_array_almost_equal(transform.matrix, identity.matrix)
