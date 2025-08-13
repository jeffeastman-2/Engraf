"""
Scene Assembly

This module defines SceneAssembly, a hierarchical container that groups multiple
SceneObjects into a single unit that can be manipulated as one entity.
"""

from typing import List, Optional, Dict, Any
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.scene.scene_entity import SceneEntity
from engraf.lexer.vector_space import VectorSpace
import math


class SceneAssembly(SceneEntity):
    """
    A hierarchical container that groups multiple SceneObjects into a single unit.
    Contains the actual SceneObject instances rather than just references.
    Can be treated as a compound noun in the vocabulary and manipulated as a single entity.
    """
    
    def __init__(self, name: str, objects: Optional[List[SceneObject]] = None, assembly_id: Optional[str] = None):
        """
        Initialize a SceneAssembly.
        
        Args:
            name: The assembly type name (e.g., 'house', 'car', 'table_setting')
            objects: List of SceneObject instances to include in this assembly
            assembly_id: Unique identifier for this assembly instance
        """
        self.name = name                          # e.g., 'house', 'car', 'table_setting'
        self.assembly_id = assembly_id or name    # unique identifier
        self.objects = objects or []              # list of actual SceneObject instances
        self.metadata = {}                        # Store additional metadata for matching
        
        # Assembly-level transformations (applied to all contained objects)
        self.position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.rotation = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.scale = {'x': 1.0, 'y': 1.0, 'z': 1.0}
        
        # Compute assembly properties from constituent objects
        self.vector = self._compute_assembly_vector()
        self.bounding_box = {}
        self._update_bounding_box()
    
    def add_object(self, scene_object: SceneObject) -> None:
        """Add a SceneObject to this assembly."""
        if scene_object not in self.objects:
            self.objects.append(scene_object)
            self._update_assembly_vector()
            self._update_bounding_box()
    
    def remove_object(self, scene_object: SceneObject) -> bool:
        """Remove a SceneObject from this assembly. Returns True if removed, False if not found."""
        if scene_object in self.objects:
            self.objects.remove(scene_object)
            self._update_assembly_vector()
            self._update_bounding_box()
            return True
        return False
    
    def get_object_by_name(self, name: str) -> Optional[SceneObject]:
        """Find the first object within this assembly by name."""
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None
    
    def get_objects_by_type(self, object_type: str) -> List[SceneObject]:
        """Get all objects of a specific type within this assembly."""
        return [obj for obj in self.objects if obj.name == object_type]
    
    def get_object_by_id(self, object_id: str) -> Optional[SceneObject]:
        """Find an object within this assembly by its object_id."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None
    
    def _compute_assembly_vector(self) -> VectorSpace:
        """Compute the assembly's semantic vector from its constituent objects."""
        # Start with a base vector space for the assembly
        assembly_vector = VectorSpace()
        
        # Mark it as an assembly noun
        assembly_vector['noun'] = 1.0
        assembly_vector['assembly'] = 1.0
        
        # Compute centroid position from all objects
        if self.objects:
            total_x = sum(obj.vector['locX'] for obj in self.objects)
            total_y = sum(obj.vector['locY'] for obj in self.objects)
            total_z = sum(obj.vector['locZ'] for obj in self.objects)
            count = len(self.objects)
            
            assembly_vector['locX'] = total_x / count
            assembly_vector['locY'] = total_y / count
            assembly_vector['locZ'] = total_z / count
        
        return assembly_vector
    
    def _update_assembly_vector(self) -> None:
        """Update the assembly's vector after changes to constituent objects."""
        self.vector = self._compute_assembly_vector()
    
    def _update_bounding_box(self) -> None:
        """Calculate the bounding box that encompasses all objects."""
        if not self.objects:
            self.bounding_box = {
                'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0, 'min_z': 0, 'max_z': 0,
                'width': 0, 'height': 0, 'depth': 0
            }
            return
        
        # Initialize with first object's bounds
        first_obj = self.objects[0]
        min_x = first_obj.position['x'] - first_obj.scale['x']/2
        max_x = first_obj.position['x'] + first_obj.scale['x']/2
        min_y = first_obj.position['y'] - first_obj.scale['y']/2
        max_y = first_obj.position['y'] + first_obj.scale['y']/2
        min_z = first_obj.position['z'] - first_obj.scale['z']/2
        max_z = first_obj.position['z'] + first_obj.scale['z']/2
        
        # Expand bounds to include all objects
        for obj in self.objects[1:]:
            obj_min_x = obj.position['x'] - obj.scale['x']/2
            obj_max_x = obj.position['x'] + obj.scale['x']/2
            obj_min_y = obj.position['y'] - obj.scale['y']/2
            obj_max_y = obj.position['y'] + obj.scale['y']/2
            obj_min_z = obj.position['z'] - obj.scale['z']/2
            obj_max_z = obj.position['z'] + obj.scale['z']/2
            
            min_x = min(min_x, obj_min_x)
            max_x = max(max_x, obj_max_x)
            min_y = min(min_y, obj_min_y)
            max_y = max(max_y, obj_max_y)
            min_z = min(min_z, obj_min_z)
            max_z = max(max_z, obj_max_z)
        
        self.bounding_box = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'min_z': min_z, 'max_z': max_z,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'depth': max_z - min_z
        }
    
    def move_by(self, delta_x: float, delta_y: float, delta_z: float) -> None:
        """Move all objects in the assembly by the specified deltas."""
        for obj in self.objects:
            obj.vector['locX'] += delta_x
            obj.vector['locY'] += delta_y
            obj.vector['locZ'] += delta_z
            obj.update_transformations()
        
        # Update assembly tracking
        self.position['x'] += delta_x
        self.position['y'] += delta_y
        self.position['z'] += delta_z
        
        self._update_assembly_vector()
        self._update_bounding_box()
    
    def move_to(self, new_x: float, new_y: float, new_z: float) -> None:
        """Move the assembly so its centroid is at the specified position."""
        current_center_x = self.vector['locX']
        current_center_y = self.vector['locY']
        current_center_z = self.vector['locZ']
        
        delta_x = new_x - current_center_x
        delta_y = new_y - current_center_y
        delta_z = new_z - current_center_z
        
        self.move_by(delta_x, delta_y, delta_z)
    
    def scale_by(self, factor_x: float, factor_y: float, factor_z: float) -> None:
        """Scale all objects in the assembly by the specified factors."""
        center_x = self.vector['locX']
        center_y = self.vector['locY']
        center_z = self.vector['locZ']
        
        for obj in self.objects:
            # Scale object size
            obj.vector['scaleX'] *= factor_x
            obj.vector['scaleY'] *= factor_y
            obj.vector['scaleZ'] *= factor_z
            
            # Scale position relative to assembly center
            rel_x = obj.vector['locX'] - center_x
            rel_y = obj.vector['locY'] - center_y
            rel_z = obj.vector['locZ'] - center_z
            
            obj.vector['locX'] = center_x + (rel_x * factor_x)
            obj.vector['locY'] = center_y + (rel_y * factor_y)
            obj.vector['locZ'] = center_z + (rel_z * factor_z)
            
            obj.update_transformations()
        
        # Update assembly tracking
        self.scale['x'] *= factor_x
        self.scale['y'] *= factor_y
        self.scale['z'] *= factor_z
        
        self._update_assembly_vector()
        self._update_bounding_box()
    
    def rotate_around_center(self, angle_x: float, angle_y: float, angle_z: float) -> None:
        """
        Rotate all objects around the assembly's center point.
        Note: This is a simplified rotation that works per-axis.
        For VPython compatibility, we rotate objects individually.
        """
        center_x = self.vector['locX']
        center_y = self.vector['locY']
        center_z = self.vector['locZ']
        
        # Convert angles to radians
        rad_x = math.radians(angle_x)
        rad_y = math.radians(angle_y)
        rad_z = math.radians(angle_z)
        
        for obj in self.objects:
            # Update object's own rotation
            obj.vector['rotX'] += angle_x
            obj.vector['rotY'] += angle_y
            obj.vector['rotZ'] += angle_z
            
            # Rotate object's position around assembly center
            # Start with position relative to center
            rel_x = obj.vector['locX'] - center_x
            rel_y = obj.vector['locY'] - center_y
            rel_z = obj.vector['locZ'] - center_z
            
            # Apply Z-axis rotation (most common case)
            if angle_z != 0:
                cos_z = math.cos(rad_z)
                sin_z = math.sin(rad_z)
                new_rel_x = rel_x * cos_z - rel_y * sin_z
                new_rel_y = rel_x * sin_z + rel_y * cos_z
                rel_x, rel_y = new_rel_x, new_rel_y
            
            # Apply Y-axis rotation
            if angle_y != 0:
                cos_y = math.cos(rad_y)
                sin_y = math.sin(rad_y)
                new_rel_x = rel_x * cos_y + rel_z * sin_y
                new_rel_z = -rel_x * sin_y + rel_z * cos_y
                rel_x, rel_z = new_rel_x, new_rel_z
            
            # Apply X-axis rotation
            if angle_x != 0:
                cos_x = math.cos(rad_x)
                sin_x = math.sin(rad_x)
                new_rel_y = rel_y * cos_x - rel_z * sin_x
                new_rel_z = rel_y * sin_x + rel_z * cos_x
                rel_y, rel_z = new_rel_y, new_rel_z
            
            # Update object position
            obj.vector['locX'] = center_x + rel_x
            obj.vector['locY'] = center_y + rel_y
            obj.vector['locZ'] = center_z + rel_z
            
            obj.update_transformations()
        
        # Update assembly tracking
        self.rotation['x'] += angle_x
        self.rotation['y'] += angle_y
        self.rotation['z'] += angle_z
        
        self._update_assembly_vector()
        self._update_bounding_box()
    
    def update_transformations(self) -> None:
        """Update transformation properties from vector space (call after vector changes)."""
        # Update position from vector
        self.position['x'] = self.vector['locX']
        self.position['y'] = self.vector['locY']
        self.position['z'] = self.vector['locZ']
        
        # Update all contained objects
        for obj in self.objects:
            obj.update_transformations()
        
        self._update_bounding_box()
    
    # SceneEntity interface implementation
    @property
    def entity_id(self) -> str:
        """Unique identifier for this entity."""
        return self.assembly_id
    
    def get_position(self) -> tuple[float, float, float]:
        """Get the current position of the assembly center."""
        return (self.vector['locX'], self.vector['locY'], self.vector['locZ'])
    
    def get_rotation(self) -> tuple[float, float, float]:
        """Get the current rotation of the assembly (in degrees)."""
        return (self.rotation['x'], self.rotation['y'], self.rotation['z'])
    
    def get_scale(self) -> tuple[float, float, float]:
        """Get the current scale of the assembly."""
        return (self.scale['x'], self.scale['y'], self.scale['z'])
    
    def __repr__(self) -> str:
        """String representation of the assembly."""
        object_names = [obj.name for obj in self.objects]
        center_x = self.vector['locX']
        center_y = self.vector['locY']
        center_z = self.vector['locZ']
        return f"<SceneAssembly '{self.name}' ({self.assembly_id}) objects={object_names} center=[{center_x:.1f},{center_y:.1f},{center_z:.1f}]>"
