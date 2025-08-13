"""
Scene Entity Interface

This module defines the abstract base class that both SceneObject and SceneAssembly
implement, providing a unified API for all scene entities.
"""

from abc import ABC, abstractmethod
from typing import Optional


class SceneEntity(ABC):
    """
    Abstract base class for all scene entities (objects and assemblies).
    Provides a unified interface for transformation operations.
    """
    
    @property
    @abstractmethod
    def entity_id(self) -> str:
        """Unique identifier for this entity."""
        pass
    
    @abstractmethod
    def move_to(self, new_x: float, new_y: float, new_z: float) -> None:
        """Move the entity to the specified coordinates."""
        pass
    
    @abstractmethod
    def scale_by(self, factor_x: float, factor_y: float, factor_z: float) -> None:
        """Scale the entity by the specified factors."""
        pass
    
    @abstractmethod
    def rotate_around_center(self, angle_x: float, angle_y: float, angle_z: float) -> None:
        """Rotate the entity around its center by the specified angles (in degrees)."""
        pass
    
    @abstractmethod
    def get_position(self) -> tuple[float, float, float]:
        """Get the current position of the entity."""
        pass
    
    @abstractmethod
    def get_rotation(self) -> tuple[float, float, float]:
        """Get the current rotation of the entity (in degrees)."""
        pass
    
    @abstractmethod
    def get_scale(self) -> tuple[float, float, float]:
        """Get the current scale of the entity."""
        pass
