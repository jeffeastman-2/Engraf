"""
VPython renderer for 3D visualization.

This module provides a VPython-based renderer that can display 3D objects
created by the ENGRAF system. It supports basic geometric shapes with
colors, textures, and transformations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

try:
    import vpython as vp
    VPYTHON_AVAILABLE = True
except ImportError:
    VPYTHON_AVAILABLE = False

from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.transforms.transform_matrix import TransformMatrix


class RendererBase(ABC):
    """Abstract base class for all renderers."""
    
    @abstractmethod
    def render_scene(self, scene: SceneModel) -> None:
        """Render the entire scene."""
        pass
    
    @abstractmethod
    def render_object(self, obj: SceneObject) -> None:
        """Render a single object."""
        pass
    
    @abstractmethod
    def clear_scene(self) -> None:
        """Clear the current scene."""
        pass


class VPythonRenderer(RendererBase):
    """
    VPython-based 3D renderer.
    
    This renderer creates 3D visualizations using VPython, supporting
    basic geometric shapes with colors, scaling, and positioning.
    """
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "ENGRAF 3D Visualizer", headless: bool = False):
        """
        Initialize the VPython renderer.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            title: Window title
            headless: If True, disable browser window (for testing)
        """
        if not VPYTHON_AVAILABLE:
            raise ImportError("VPython is required for VPythonRenderer. Install with: pip install vpython")
        
        self.width = width
        self.height = height
        self.title = title
        self.headless = headless
        
        # Initialize VPython scene
        self._init_scene()
        
        # Keep track of rendered objects
        self.rendered_objects: Dict[str, vp.compound] = {}
        
        # Shape creation methods
        self.shape_creators = {
            "cube": self._create_cube,
            "box": self._create_cube,
            "sphere": self._create_sphere,
            "ellipsoid": self._create_ellipsoid,
            "cylinder": self._create_cylinder,
            "cone": self._create_cone,
            "pyramid": self._create_pyramid,
            "arch": self._create_arch,
            "table": self._create_table,
        }
    
    def _init_scene(self) -> None:
        """Initialize the VPython scene with basic settings."""
        if self.headless:
            # For headless mode (testing), create a minimal scene
            self.scene = None
            return
            
        self.scene = vp.canvas(
            width=self.width,
            height=self.height,
            title=self.title,
            background=vp.color.gray(0.2),
            center=vp.vector(0, 0, 0),
            forward=vp.vector(0, 0, -1),
            up=vp.vector(0, 1, 0),
            range=10
        )
        
        # VPython now uses default lighting - we can't set custom lights
        # The scene will automatically have appropriate lighting
        
        # Note: Disabled default object clearing to debug the issue
        # self._clear_default_objects()
    
    def _clear_default_objects(self) -> None:
        """Clear any default objects that VPython might have created."""
        if self.headless or self.scene is None:
            return
            
        # For now, disable automatic clearing to avoid removing legitimate objects
        # VPython's default object creation seems to happen at unpredictable times
        # TODO: Find a better way to detect and remove only VPython's default objects
        pass
    
    def render_scene(self, scene: SceneModel) -> None:
        """
        Render the entire scene incrementally.
        
        Args:
            scene: The scene model to render
        """
        # Only render objects that haven't been rendered yet
        for obj in scene.objects:
            if obj.object_id not in self.rendered_objects:
                self.render_object(obj)
    
    def render_object(self, obj: SceneObject) -> None:
        """
        Render a single object.
        
        Args:
            obj: The scene object to render
        """
        # Get the shape name from the object (extract from names like "cube_1", "sphere_2", etc.)
        shape_name = obj.name.split('_')[0].lower() if '_' in obj.name else obj.name.lower()
        
        # Determine the appropriate shape creator
        creator = self.shape_creators.get(shape_name, self._create_cube)
        
        # Create the VPython object
        vpython_obj = creator(obj)
        
        # Store the rendered object using object_id as key
        self.rendered_objects[obj.object_id] = vpython_obj
    
    def clear_scene(self) -> None:
        """Clear all objects from the scene."""
        for obj in self.rendered_objects.values():
            obj.visible = False
            del obj
        self.rendered_objects.clear()
        
        # Note: Disabled unwanted object clearing to debug the issue
        # self._clear_unwanted_objects()
    
    def _clear_unwanted_objects(self) -> None:
        """Clear any unwanted objects that VPython might have created after scene completion."""
        if self.headless or self.scene is None:
            return
            
        try:
            # Get all objects in the scene
            scene_objects = self.scene.objects
            
            # Remove any objects that are not in our rendered_objects dict
            objects_to_remove = []
            for obj in scene_objects:
                # Check if this object is one we created
                is_our_object = False
                for rendered_obj in self.rendered_objects.values():
                    if obj == rendered_obj:
                        is_our_object = True
                        break
                
                # If it's not our object, mark it for removal
                if not is_our_object:
                    objects_to_remove.append(obj)
            
            # Remove the unwanted objects
            for obj in objects_to_remove:
                if hasattr(obj, 'visible'):
                    obj.visible = False
                    del obj
                    
        except Exception:
            # If there's any issue clearing unwanted objects, just continue
            pass
    
    def _create_cube(self, obj: SceneObject) -> vp.compound:
        """Create a cube/box object."""
        # Extract properties from the object
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        # Create the cube
        cube = vp.box(
            pos=position,
            size=size,
            color=color
        )
        
        # Apply any transformations
        self._apply_transformations(cube, obj)
        
        return cube
    
    def _create_sphere(self, obj: SceneObject) -> vp.compound:
        """Create a sphere object."""
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        # Use the average of the size components as radius
        radius = np.mean([size.x, size.y, size.z])
        
        sphere = vp.sphere(
            pos=position,
            radius=radius,
            color=color
        )
        
        self._apply_transformations(sphere, obj)
        
        return sphere
    
    def _create_ellipsoid(self, obj: SceneObject) -> vp.compound:
        """Create an ellipsoid object that can be scaled non-uniformly."""
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        # Create ellipsoid with proper scaling
        ellipsoid = vp.ellipsoid(
            pos=position,
            size=size,  # VPython ellipsoid uses size directly
            color=color
        )
        
        self._apply_transformations(ellipsoid, obj)
        
        return ellipsoid
    
    def _create_cylinder(self, obj: SceneObject) -> vp.compound:
        """Create a cylinder object."""
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        cylinder = vp.cylinder(
            pos=position,
            radius=size.x,
            height=size.y,
            color=color
        )
        
        self._apply_transformations(cylinder, obj)
        
        return cylinder
    
    def _create_cone(self, obj: SceneObject) -> vp.compound:
        """Create a cone object."""
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        # Create cone with circular base
        cone = vp.cone(
            pos=position + vp.vector(0, -size.y/2, 0),  # Position at bottom
            axis=vp.vector(0, size.y, 0),  # Point upward
            radius=max(size.x, size.z) / 2,  # Use larger dimension for radius
            color=color
        )
        
        self._apply_transformations(cone, obj)
        
        return cone
    
    def _create_pyramid(self, obj: SceneObject) -> vp.pyramid:
        """Create a pyramid object with square base and triangular sides."""
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        # Create pyramid with square base
        pyramid = vp.pyramid(
            pos=position,
            size=vp.vector(size.x, size.y, size.z),
            color=color
        )
        
        self._apply_transformations(pyramid, obj)
        
        return pyramid
    
    def _create_arch(self, obj: SceneObject) -> vp.compound:
        """Create an arch object using compound shapes."""
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        # Create arch as combination of boxes and a cylinder
        # Base pillars
        left_pillar = vp.box(
            pos=position + vp.vector(-size.x/2, 0, 0),
            size=vp.vector(size.x/4, size.y, size.z),
            color=color
        )
        
        right_pillar = vp.box(
            pos=position + vp.vector(size.x/2, 0, 0),
            size=vp.vector(size.x/4, size.y, size.z),
            color=color
        )
        
        # Top arch (half cylinder)
        arch_top = vp.cylinder(
            pos=position + vp.vector(-size.x/2, size.y/2, 0),
            axis=vp.vector(size.x, 0, 0),
            radius=size.y/4,
            color=color
        )
        
        # Combine into compound object
        arch = vp.compound([left_pillar, right_pillar, arch_top])
        
        self._apply_transformations(arch, obj)
        
        return arch
    
    def _create_table(self, obj: SceneObject) -> vp.compound:
        """Create a table object using compound shapes."""
        position = self._extract_position(obj)
        size = self._extract_size(obj)
        color = self._extract_color(obj)
        
        # Table top
        table_top = vp.box(
            pos=position + vp.vector(0, size.y/2, 0),
            size=vp.vector(size.x, size.y/10, size.z),
            color=color
        )
        
        # Table legs
        leg_size = vp.vector(size.x/20, size.y, size.z/20)
        leg_positions = [
            position + vp.vector(-size.x/2.5, 0, -size.z/2.5),
            position + vp.vector(size.x/2.5, 0, -size.z/2.5),
            position + vp.vector(-size.x/2.5, 0, size.z/2.5),
            position + vp.vector(size.x/2.5, 0, size.z/2.5)
        ]
        
        legs = []
        for leg_pos in leg_positions:
            leg = vp.box(
                pos=leg_pos,
                size=leg_size,
                color=color
            )
            legs.append(leg)
        
        # Combine all parts
        table = vp.compound([table_top] + legs)
        
        self._apply_transformations(table, obj)
        
        return table
    
    def _extract_position(self, obj: SceneObject) -> vp.vector:
        """Extract position from object's vector or default to origin."""
        if obj.vector:
            try:
                x = obj.vector["locX"]
                y = obj.vector["locY"]
                z = obj.vector["locZ"]
                return vp.vector(x, y, z)
            except (KeyError, TypeError):
                pass
        
        return vp.vector(0, 0, 0)
    
    def _extract_size(self, obj: SceneObject) -> vp.vector:
        """Extract size from object's vector or default to unit size."""
        if obj.vector:
            try:
                x = obj.vector["scaleX"]
                y = obj.vector["scaleY"]
                z = obj.vector["scaleZ"]
                return vp.vector(abs(x), abs(y), abs(z))
            except (KeyError, TypeError):
                pass
        
        return vp.vector(1, 1, 1)
    
    def _extract_color(self, obj: SceneObject) -> vp.vector:
        """Extract color from object's vector or default to white."""
        if obj.vector:
            try:
                r = obj.vector["red"]
                g = obj.vector["green"]
                b = obj.vector["blue"]
                
                # Handle black color specially - VPython (0,0,0) may not render properly
                if r == 0 and g == 0 and b == 0:
                    return vp.vector(0.1, 0.1, 0.1)  # Very dark gray instead of pure black
                
                return vp.vector(r, g, b)
            except (KeyError, TypeError):
                pass
        
        return vp.vector(1, 1, 1)  # Default to white
    
    def _apply_transformations(self, vpython_obj: vp.compound, obj: SceneObject) -> None:
        """Apply any transformations stored in the object's modifiers."""
        if not obj.modifiers:
            return
        
        # Look for transformation matrices in modifiers
        for modifier in obj.modifiers:
            if hasattr(modifier, 'matrix'):
                # Apply the transformation matrix
                self._apply_transform_matrix(vpython_obj, modifier.matrix)
    
    def _apply_transform_matrix(self, vpython_obj: vp.compound, matrix: np.ndarray) -> None:
        """Apply a 4x4 transformation matrix to a VPython object."""
        # Extract translation
        translation = matrix[:3, 3]
        
        # Apply translation
        vpython_obj.pos += vp.vector(translation[0], translation[1], translation[2])
        
        # For now, skip rotation and scaling transformations
        # VPython doesn't directly support matrix transformations
        # This could be enhanced to decompose the matrix and apply individual transformations
    
    def update_object(self, obj: SceneObject) -> None:
        """Update an existing object in the scene."""
        if obj.name in self.rendered_objects:
            # Remove the old object
            self.rendered_objects[obj.name].visible = False
            del self.rendered_objects[obj.name]
        
        # Render the updated object
        self.render_object(obj)
    
    def get_object_info(self, obj_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a rendered object."""
        if obj_name in self.rendered_objects:
            vpython_obj = self.rendered_objects[obj_name]
            return {
                "name": obj_name,
                "position": [vpython_obj.pos.x, vpython_obj.pos.y, vpython_obj.pos.z],
                "visible": vpython_obj.visible,
                "color": [vpython_obj.color.x, vpython_obj.color.y, vpython_obj.color.z]
            }
        return None
    
    def set_camera(self, position: Tuple[float, float, float], target: Tuple[float, float, float]) -> None:
        """Set the camera position and target."""
        self.scene.camera.pos = vp.vector(*position)
        self.scene.center = vp.vector(*target)
    
    def set_background_color(self, color: Tuple[float, float, float]) -> None:
        """Set the background color."""
        self.scene.background = vp.vector(*color)


class MockVPythonRenderer(RendererBase):
    """
    Mock renderer for testing when VPython is not available.
    
    This renderer simulates the VPython renderer without actually displaying anything,
    useful for testing and development when VPython is not installed.
    """
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "ENGRAF 3D Visualizer"):
        """Initialize the mock renderer."""
        self.width = width
        self.height = height
        self.title = title
        self.rendered_objects: Dict[str, Dict[str, Any]] = {}
    
    def render_scene(self, scene: SceneModel) -> None:
        """Mock render the entire scene."""
        self.clear_scene()
        for obj in scene.objects:
            self.render_object(obj)
    
    def render_object(self, obj: SceneObject) -> None:
        """Mock render a single object."""
        mock_obj = {
            "name": obj.name,
            "position": [0, 0, 0],
            "size": [1, 1, 1],
            "color": [1, 1, 1],
            "visible": True
        }
        self.rendered_objects[obj.name] = mock_obj
    
    def clear_scene(self) -> None:
        """Clear all mock objects."""
        self.rendered_objects.clear()
    
    def update_object(self, obj: SceneObject) -> None:
        """Update a mock object."""
        self.render_object(obj)
    
    def get_object_info(self, obj_name: str) -> Optional[Dict[str, Any]]:
        """Get info about a mock object."""
        return self.rendered_objects.get(obj_name)


def create_renderer(backend: str = "vpython", **kwargs) -> RendererBase:
    """
    Factory function to create a renderer.
    
    Args:
        backend: The rendering backend ("vpython" or "mock")
        **kwargs: Additional arguments passed to the renderer
        
    Returns:
        A renderer instance
    """
    if backend == "vpython":
        if VPYTHON_AVAILABLE:
            return VPythonRenderer(**kwargs)
        else:
            print("Warning: VPython not available, using mock renderer")
            return MockVPythonRenderer(**kwargs)
    elif backend == "mock":
        return MockVPythonRenderer(**kwargs)
    else:
        raise ValueError(f"Unknown renderer backend: {backend}")
