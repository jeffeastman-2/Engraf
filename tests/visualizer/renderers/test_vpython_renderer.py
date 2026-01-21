"""
Tests for the VPython renderer.

Tests both the real VPython renderer and the mock renderer for compatibility.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from engraf.visualizer.renderers.vpython_renderer import (
    VPythonRenderer, MockVPythonRenderer, create_renderer, VPYTHON_AVAILABLE
)
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.transforms.transform_matrix import TransformMatrix
from engraf.lexer.vector_space import VectorSpace


class TestMockVPythonRenderer:
    """Test suite for the MockVPythonRenderer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = MockVPythonRenderer(width=800, height=600, title="Test")
    
    def create_mock_vector(self, **kwargs):
        """Create a properly mocked VectorSpace object that supports SceneObject requirements."""
        vector = Mock(spec=VectorSpace)
        # Mock the __contains__ method to check for specific keys
        def contains_mock(self_mock, key):
            return key in kwargs
        vector.__contains__ = contains_mock
        
        # Mock __getitem__ to return values from kwargs or 0.0 as default
        def getitem_mock(self_mock, key):
            return kwargs.get(key, 0.0)
        vector.__getitem__ = getitem_mock
        
        return vector
    
    def test_initialization(self):
        """Test that the mock renderer initializes correctly."""
        assert self.renderer.width == 800
        assert self.renderer.height == 600
        assert self.renderer.title == "Test"
        assert isinstance(self.renderer.rendered_objects, dict)
        assert len(self.renderer.rendered_objects) == 0
    
    def test_render_object(self):
        """Test rendering a single object."""
        # Create a mock scene object with proper vector behavior
        vector = self.create_mock_vector()
        
        obj = SceneObject("cube", vector)
        
        # Render the object
        self.renderer.render_object(obj)
        
        # Check that the object was rendered
        assert "cube" in self.renderer.rendered_objects
        rendered_obj = self.renderer.rendered_objects["cube"]
        assert rendered_obj["name"] == "cube"
        assert rendered_obj["visible"] == True
        assert rendered_obj["position"] == [0, 0, 0]
        assert rendered_obj["size"] == [1, 1, 1]
        assert rendered_obj["color"] == [1, 1, 1]
    
    def test_render_scene(self):
        """Test rendering an entire scene."""
        # Create a mock scene with multiple objects
        scene = SceneModel()
        
        vector1 = self.create_mock_vector()
        vector2 = self.create_mock_vector()
        
        obj1 = SceneObject("cube", vector1)
        obj2 = SceneObject("sphere", vector2)
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        # Render the scene
        self.renderer.render_scene(scene)
        
        # Check that both objects were rendered
        assert len(self.renderer.rendered_objects) == 2
        assert "cube" in self.renderer.rendered_objects
        assert "sphere" in self.renderer.rendered_objects
    
    def test_clear_scene(self):
        """Test clearing the scene."""
        # Add some objects
        vector = self.create_mock_vector()
        obj = SceneObject("cube", vector)
        self.renderer.render_object(obj)
        
        assert len(self.renderer.rendered_objects) == 1
        
        # Clear the scene
        self.renderer.clear_scene()
        
        # Check that all objects were removed
        assert len(self.renderer.rendered_objects) == 0
    
    def test_update_object(self):
        """Test updating an object."""
        # Create and render an object
        vector = self.create_mock_vector()
        obj = SceneObject("cube", vector)
        self.renderer.render_object(obj)
        
        # Update the object
        obj.name = "updated_cube"
        self.renderer.update_object(obj)
        
        # Check that the object was updated
        assert "updated_cube" in self.renderer.rendered_objects
        assert self.renderer.rendered_objects["updated_cube"]["name"] == "updated_cube"
    
    def test_get_object_info(self):
        """Test getting object information."""
        # Create and render an object
        vector = self.create_mock_vector()
        obj = SceneObject("cube", vector)
        self.renderer.render_object(obj)
        
        # Get object info
        info = self.renderer.get_object_info("cube")
        
        assert info is not None
        assert info["name"] == "cube"
        assert info["visible"] == True
        
        # Test with non-existent object
        info = self.renderer.get_object_info("nonexistent")
        assert info is None


@pytest.mark.skipif(not VPYTHON_AVAILABLE, reason="VPython not available")
class TestVPythonRenderer:
    """Test suite for the VPythonRenderer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a persistent patch to avoid opening windows during tests
        self.vp_patcher = patch('engraf.visualizer.renderers.vpython_renderer.vp')
        self.mock_vp = self.vp_patcher.start()
        
        # Mock the canvas and other VPython objects
        mock_canvas = Mock()
        mock_canvas.scene = Mock()
        self.mock_vp.canvas.return_value = mock_canvas
        self.mock_vp.vector = Mock(side_effect=lambda x, y, z: Mock(x=x, y=y, z=z))
        self.mock_vp.color = Mock()
        self.mock_vp.color.gray = Mock(return_value=Mock())
        self.mock_vp.box = Mock()
        self.mock_vp.sphere = Mock()
        self.mock_vp.cylinder = Mock()
        self.mock_vp.cone = Mock()
        self.mock_vp.compound = Mock()
        
        self.renderer = VPythonRenderer(width=800, height=600, title="Test", headless=True)
    
    def create_mock_vector(self, **kwargs):
        """Create a properly mocked VectorSpace object that supports SceneObject requirements."""
        vector = Mock(spec=VectorSpace)
        # Mock the __contains__ method to check for specific keys
        def contains_mock(self_mock, key):
            return key in kwargs
        vector.__contains__ = contains_mock
        
        # Mock __getitem__ to return values from kwargs or 0.0 as default
        def getitem_mock(self_mock, key):
            return kwargs.get(key, 0.0)
        vector.__getitem__ = getitem_mock
        
        return vector
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.vp_patcher.stop()
    
    def test_initialization(self):
        """Test that the VPython renderer initializes correctly."""
        assert self.renderer.width == 800
        assert self.renderer.height == 600
        assert self.renderer.title == "Test"
        assert isinstance(self.renderer.rendered_objects, dict)
        assert len(self.renderer.rendered_objects) == 0
        
        # Check that headless mode was set
        assert self.renderer.headless == True
        
        # In headless mode, no canvas should be created
        self.mock_vp.canvas.assert_not_called()
    
    def test_render_cube(self):
        """Test rendering a cube object."""
        # Create a cube object with vector
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(side_effect=lambda key: {
            "locX": 1.0, "locY": 2.0, "locZ": 3.0,
            "scaleX": 2.0, "scaleY": 2.0, "scaleZ": 2.0,
            "red": 1.0, "green": 0.0, "blue": 0.0
        }.get(key, 0.0))
        
        obj = SceneObject("cube", vector)
        
        # Render the object
        self.renderer.render_object(obj)
        
        # Check that VPython box was created
        self.mock_vp.box.assert_called_once()
        
        # Check that the object was stored
        assert "cube" in self.renderer.rendered_objects
    
    def test_render_sphere(self):
        """Test rendering a sphere object."""
        # Create a sphere object
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(side_effect=lambda key: {
            "locX": 0.0, "locY": 0.0, "locZ": 0.0,
            "scaleX": 1.0, "scaleY": 1.0, "scaleZ": 1.0,
            "red": 0.0, "green": 1.0, "blue": 0.0
        }.get(key, 0.0))
        
        obj = SceneObject("sphere", vector)
        
        # Render the object
        self.renderer.render_object(obj)
        
        # Check that VPython sphere was created
        self.mock_vp.sphere.assert_called_once()
        
        # Check that the object was stored
        assert "sphere" in self.renderer.rendered_objects
    
    def test_extract_position(self):
        """Test extracting position from object vector."""
        # Create object with position
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(side_effect=lambda key: {
            "locX": 1.0, "locY": 2.0, "locZ": 3.0
        }.get(key, 0.0))
        
        obj = SceneObject("cube", vector)
        
        # Extract position
        position = self.renderer._extract_position(obj)
        
        # Check that position was extracted correctly
        assert position.x == 1.0
        assert position.y == 2.0
        assert position.z == 3.0
    
    def test_extract_size(self):
        """Test extracting size from object vector."""
        # Create object with size
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(side_effect=lambda key: {
            "scaleX": 2.0, "scaleY": 3.0, "scaleZ": 4.0
        }.get(key, 1.0))
        
        obj = SceneObject("cube", vector)
        
        # Extract size
        size = self.renderer._extract_size(obj)
        
        # Check that size was extracted correctly
        assert size.x == 2.0
        assert size.y == 3.0
        assert size.z == 4.0
    
    def test_extract_color(self):
        """Test extracting color from object vector."""
        # Create object with color
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(side_effect=lambda key: {
            "red": 1.0, "green": 0.5, "blue": 0.2
        }.get(key, 1.0))
        
        obj = SceneObject("cube", vector)
        
        # Extract color
        color = self.renderer._extract_color(obj)
        
        # Check that color was extracted correctly
        assert color.x == 1.0
        assert color.y == 0.5
        assert color.z == 0.2
    
    def test_render_scene_with_multiple_objects(self):
        """Test rendering a scene with multiple objects."""
        # Create a scene with multiple objects
        scene = SceneModel()
        
        vector1 = self.create_mock_vector()
        vector1.__getitem__ = Mock(return_value=0.0)
        vector2 = self.create_mock_vector()
        vector2.__getitem__ = Mock(return_value=0.0)
        
        obj1 = SceneObject("cube", vector1)
        obj2 = SceneObject("sphere", vector2)
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        # Render the scene
        self.renderer.render_scene(scene)
        
        # Check that both objects were rendered
        assert len(self.renderer.rendered_objects) == 2
        assert "cube" in self.renderer.rendered_objects
        assert "sphere" in self.renderer.rendered_objects
    
    def test_clear_scene(self):
        """Test clearing the scene."""
        # Add some objects
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(return_value=0.0)
        obj = SceneObject("cube", vector)
        self.renderer.render_object(obj)
        
        assert len(self.renderer.rendered_objects) == 1
        
        # Clear the scene
        self.renderer.clear_scene()
        
        # Check that all objects were removed
        assert len(self.renderer.rendered_objects) == 0


class TestRendererFactory:
    """Test suite for the renderer factory function."""
    
    def test_create_mock_renderer(self):
        """Test creating a mock renderer."""
        renderer = create_renderer(backend="mock", width=1024, height=768, title="Test Mock")
        
        assert isinstance(renderer, MockVPythonRenderer)
        assert renderer.width == 1024
        assert renderer.height == 768
        assert renderer.title == "Test Mock"
    
    @pytest.mark.skipif(not VPYTHON_AVAILABLE, reason="VPython not available")
    def test_create_vpython_renderer(self):
        """Test creating a VPython renderer."""
        with patch('engraf.visualizer.renderers.vpython_renderer.vp') as mock_vp:
            # Mock VPython components
            mock_canvas = Mock()
            mock_canvas.scene = Mock()
            mock_vp.canvas.return_value = mock_canvas
            mock_vp.vector = Mock(side_effect=lambda x, y, z: Mock(x=x, y=y, z=z))
            mock_vp.color = Mock()
            mock_vp.color.gray = Mock(return_value=Mock())
            
            renderer = create_renderer(backend="vpython", width=1024, height=768, title="Test VPython")
            
            assert isinstance(renderer, VPythonRenderer)
            assert renderer.width == 1024
            assert renderer.height == 768
            assert renderer.title == "Test VPython"
    
    def test_create_unknown_renderer(self):
        """Test creating an unknown renderer raises an error."""
        with pytest.raises(ValueError, match="Unknown renderer backend"):
            create_renderer(backend="unknown")
    
    def test_fallback_to_mock_when_vpython_unavailable(self):
        """Test fallback to mock renderer when VPython is not available."""
        with patch('engraf.visualizer.renderers.vpython_renderer.VPYTHON_AVAILABLE', False):
            with patch('builtins.print') as mock_print:
                renderer = create_renderer(backend="vpython")
                
                assert isinstance(renderer, MockVPythonRenderer)
                mock_print.assert_called_once_with("Warning: VPython not available, using mock renderer")


class TestRendererIntegration:
    """Integration tests for the renderer with the scene system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = MockVPythonRenderer()
        self.scene = SceneModel()
    
    def create_mock_vector(self, **kwargs):
        """Create a properly mocked VectorSpace object that supports SceneObject requirements."""
        vector = Mock(spec=VectorSpace)
        # Mock the __contains__ method to check for specific keys
        def contains_mock(self_mock, key):
            return key in kwargs
        vector.__contains__ = contains_mock
        
        # Mock __getitem__ to return values from kwargs or 0.0 as default
        def getitem_mock(self_mock, key):
            return kwargs.get(key, 0.0)
        vector.__getitem__ = getitem_mock
        
        return vector
    
    def test_render_scene_with_transforms(self):
        """Test rendering a scene with transformed objects."""
        # Create object with transformation via vector dimensions
        vector = self.create_mock_vector(
            locX=1.0, locY=2.0, locZ=3.0,
            scaleX=2.0, scaleY=2.0, scaleZ=2.0,
            red=1.0, green=0.0, blue=0.0,
            rotX=45.0, rotY=0.0, rotZ=0.0  # Add rotation transform
        )
        
        obj = SceneObject("cube", vector)
        self.scene.add_object(obj)
        
        # Render the scene
        self.renderer.render_scene(self.scene)
        
        # Check that the object was rendered
        assert "cube" in self.renderer.rendered_objects
        assert self.renderer.rendered_objects["cube"]["name"] == "cube"
    
    def test_render_different_object_types(self):
        """Test rendering different types of objects."""
        object_types = ["cube", "sphere", "cylinder", "cone", "pyramid", "arch", "table"]
        
        for obj_type in object_types:
            # Create object of specific type
            vector = self.create_mock_vector()
            vector.__getitem__ = Mock(return_value=0.0)
            
            obj = SceneObject(obj_type, vector)
            
            # Render the object
            self.renderer.render_object(obj)
            
            # Check that the object was rendered
            assert obj_type in self.renderer.rendered_objects
            assert self.renderer.rendered_objects[obj_type]["name"] == obj_type
    
    def test_update_object_properties(self):
        """Test updating object properties."""
        # Create and render initial object
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(return_value=0.0)
        
        obj = SceneObject("cube", vector)
        self.renderer.render_object(obj)
        
        # Update object properties
        obj.name = "updated_cube"
        self.renderer.update_object(obj)
        
        # Check that the object was updated
        assert "updated_cube" in self.renderer.rendered_objects
        assert self.renderer.rendered_objects["updated_cube"]["name"] == "updated_cube"
    
    def test_scene_with_multiple_updates(self):
        """Test multiple updates to the same scene."""
        # Create initial scene
        vector = self.create_mock_vector()
        vector.__getitem__ = Mock(return_value=0.0)
        
        obj1 = SceneObject("cube", vector)
        obj2 = SceneObject("sphere", vector)
        
        self.scene.add_object(obj1)
        self.scene.add_object(obj2)
        
        # Render initial scene
        self.renderer.render_scene(self.scene)
        assert len(self.renderer.rendered_objects) == 2
        
        # Add another object
        obj3 = SceneObject("cylinder", vector)
        self.scene.add_object(obj3)
        
        # Re-render scene
        self.renderer.render_scene(self.scene)
        assert len(self.renderer.rendered_objects) == 3
        assert "cylinder" in self.renderer.rendered_objects
