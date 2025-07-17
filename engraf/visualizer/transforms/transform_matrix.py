"""
4×4 Homogeneous Transformation Matrix utilities for ENGRAF Visualizer.

This module provides utilities for creating and manipulating 4×4 homogeneous
transformation matrices for 3D graphics. All transformations (translation, 
rotation, scaling) are represented using numpy arrays.

Matrix format:
| R11 R12 R13 Tx |
| R21 R22 R23 Ty |
| R31 R32 R33 Tz |
|  0   0   0   1 |

Where:
- R11-R33: 3×3 rotation/scale matrix
- Tx, Ty, Tz: translation components
- Bottom row: [0, 0, 0, 1] for homogeneous coordinates
"""

import numpy as np
from typing import Tuple, Union


class TransformMatrix:
    """
    Represents a 4×4 homogeneous transformation matrix.
    
    Supports translation, rotation, scaling, and matrix composition.
    All angles are in degrees for user convenience.
    """
    
    def __init__(self, matrix: np.ndarray = None):
        """
        Initialize a transformation matrix.
        
        Args:
            matrix: 4×4 numpy array. If None, creates identity matrix.
        """
        if matrix is None:
            self.matrix = np.eye(4, dtype=np.float64)
        else:
            if matrix.shape != (4, 4):
                raise ValueError("Matrix must be 4×4")
            self.matrix = matrix.astype(np.float64)
    
    @classmethod
    def identity(cls) -> 'TransformMatrix':
        """Create an identity transformation matrix."""
        return cls()
    
    @classmethod
    def translation(cls, x: float, y: float, z: float) -> 'TransformMatrix':
        """
        Create a translation matrix.
        
        Args:
            x, y, z: Translation amounts along each axis
            
        Returns:
            TransformMatrix representing the translation
        """
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return cls(matrix)
    
    @classmethod
    def rotation_x(cls, degrees: float) -> 'TransformMatrix':
        """
        Create a rotation matrix around the X-axis.
        
        Args:
            degrees: Rotation angle in degrees
            
        Returns:
            TransformMatrix representing the rotation
        """
        radians = np.radians(degrees)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        
        matrix = np.eye(4, dtype=np.float64)
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_y(cls, degrees: float) -> 'TransformMatrix':
        """
        Create a rotation matrix around the Y-axis.
        
        Args:
            degrees: Rotation angle in degrees
            
        Returns:
            TransformMatrix representing the rotation
        """
        radians = np.radians(degrees)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_z(cls, degrees: float) -> 'TransformMatrix':
        """
        Create a rotation matrix around the Z-axis.
        
        Args:
            degrees: Rotation angle in degrees
            
        Returns:
            TransformMatrix representing the rotation
        """
        radians = np.radians(degrees)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_xyz(cls, x_degrees: float, y_degrees: float, z_degrees: float) -> 'TransformMatrix':
        """
        Create a combined rotation matrix for X, Y, and Z axes.
        
        Applied in order: Z, Y, X (which is the typical Euler angle order)
        
        Args:
            x_degrees, y_degrees, z_degrees: Rotation angles in degrees
            
        Returns:
            TransformMatrix representing the combined rotation
        """
        # Apply rotations in Z, Y, X order
        rot_z = cls.rotation_z(z_degrees)
        rot_y = cls.rotation_y(y_degrees)
        rot_x = cls.rotation_x(x_degrees)
        
        # Combine: R = Rx * Ry * Rz
        return rot_x.compose(rot_y.compose(rot_z))
    
    @classmethod
    def scale(cls, x: float, y: float, z: float) -> 'TransformMatrix':
        """
        Create a scaling matrix.
        
        Args:
            x, y, z: Scale factors along each axis
            
        Returns:
            TransformMatrix representing the scaling
        """
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = x
        matrix[1, 1] = y
        matrix[2, 2] = z
        return cls(matrix)
    
    @classmethod
    def uniform_scale(cls, factor: float) -> 'TransformMatrix':
        """
        Create a uniform scaling matrix.
        
        Args:
            factor: Scale factor for all axes
            
        Returns:
            TransformMatrix representing the uniform scaling
        """
        return cls.scale(factor, factor, factor)
    
    def compose(self, other: 'TransformMatrix') -> 'TransformMatrix':
        """
        Compose this transformation with another.
        
        The result applies 'other' first, then this transformation.
        Equivalent to: self * other in matrix multiplication.
        
        Args:
            other: The transformation to compose with
            
        Returns:
            New TransformMatrix representing the composition
        """
        result_matrix = np.dot(self.matrix, other.matrix)
        return TransformMatrix(result_matrix)
    
    def apply_to_point(self, point: Union[Tuple[float, float, float], np.ndarray]) -> np.ndarray:
        """
        Apply this transformation to a 3D point.
        
        Args:
            point: 3D point as tuple or numpy array
            
        Returns:
            Transformed point as numpy array
        """
        if isinstance(point, tuple):
            point = np.array(point)
        
        # Convert to homogeneous coordinates
        homogeneous_point = np.append(point, 1.0)
        
        # Apply transformation
        transformed = np.dot(self.matrix, homogeneous_point)
        
        # Convert back to 3D coordinates
        return transformed[:3] / transformed[3]
    
    def apply_to_vector(self, vector: Union[Tuple[float, float, float], np.ndarray]) -> np.ndarray:
        """
        Apply this transformation to a 3D vector (ignoring translation).
        
        Args:
            vector: 3D vector as tuple or numpy array
            
        Returns:
            Transformed vector as numpy array
        """
        if isinstance(vector, tuple):
            vector = np.array(vector)
        
        # For vectors, we only apply the rotation/scale part (3×3 upper-left)
        # and ignore translation
        rotation_scale = self.matrix[:3, :3]
        return np.dot(rotation_scale, vector)
    
    def inverse(self) -> 'TransformMatrix':
        """
        Compute the inverse transformation.
        
        Returns:
            New TransformMatrix representing the inverse transformation
            
        Raises:
            np.linalg.LinAlgError: If matrix is not invertible
        """
        inverse_matrix = np.linalg.inv(self.matrix)
        return TransformMatrix(inverse_matrix)
    
    def decompose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose the transformation into translation, rotation, and scale.
        
        Note: This is a simplified decomposition that assumes the matrix
        was built from basic transformations. For arbitrary matrices,
        decomposition can be more complex.
        
        Returns:
            Tuple of (translation, rotation_degrees, scale) as numpy arrays
        """
        # Extract translation (last column, first 3 elements)
        translation = self.matrix[:3, 3]
        
        # Extract the 3×3 rotation/scale matrix
        upper_left = self.matrix[:3, :3]
        
        # Extract scale (length of each column vector)
        scale = np.array([
            np.linalg.norm(upper_left[:, 0]),
            np.linalg.norm(upper_left[:, 1]),
            np.linalg.norm(upper_left[:, 2])
        ])
        
        # Remove scale to get pure rotation matrix
        rotation_matrix = upper_left.copy()
        rotation_matrix[:, 0] /= scale[0]
        rotation_matrix[:, 1] /= scale[1]
        rotation_matrix[:, 2] /= scale[2]
        
        # Extract rotation angles (simplified - assumes no gimbal lock)
        # This is a basic implementation for typical use cases
        rotation_y = np.arcsin(-rotation_matrix[2, 0])
        
        if np.cos(rotation_y) > 1e-6:
            rotation_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            rotation_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            # Gimbal lock case
            rotation_x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            rotation_z = 0.0
        
        rotation_degrees = np.degrees([rotation_x, rotation_y, rotation_z])
        
        return translation, rotation_degrees, scale
    
    def get_translation(self) -> np.ndarray:
        """Get the translation component of the transformation."""
        return self.matrix[:3, 3].copy()
    
    def get_scale(self) -> np.ndarray:
        """Get the scale component of the transformation."""
        upper_left = self.matrix[:3, :3]
        return np.array([
            np.linalg.norm(upper_left[:, 0]),
            np.linalg.norm(upper_left[:, 1]),
            np.linalg.norm(upper_left[:, 2])
        ])
    
    def __mul__(self, other: 'TransformMatrix') -> 'TransformMatrix':
        """
        Matrix multiplication operator.
        
        Args:
            other: The transformation to multiply with
            
        Returns:
            New TransformMatrix representing the composition
        """
        return self.compose(other)
    
    def __eq__(self, other: 'TransformMatrix') -> bool:
        """Check if two transformation matrices are equal."""
        if not isinstance(other, TransformMatrix):
            return False
        return np.allclose(self.matrix, other.matrix)
    
    def __repr__(self) -> str:
        """String representation of the transformation matrix."""
        return f"TransformMatrix(\n{self.matrix}\n)"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        translation, rotation, scale = self.decompose()
        return (f"Transform(translation={translation}, "
                f"rotation={rotation}°, scale={scale})")
