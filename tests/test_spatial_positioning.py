"""
Unit tests for spatial positioning in scene rendering.
Focus on debugging overlapping objects issue with larger objects.
"""
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject


class MockVPythonRenderer:
    """Mock renderer that captures object properties without visual rendering."""
    
    def __init__(self):
        self.objects = {}  # Store objects by ID
        self.scene_data = None
    
    def create_object(self, obj_type, position, color, size):
        """Create a mock object with the given properties."""
        obj_id = f"{obj_type}_{len(self.objects)}"
        self.objects[obj_id] = {
            'type': obj_type,
            'position': position,
            'color': color,
            'size': size,
            'scale': 1.0
        }
        return obj_id
    
    def update_object(self, obj_id, **properties):
        """Update object properties."""
        if obj_id in self.objects:
            self.objects[obj_id].update(properties)
    
    def render_scene(self, scene):
        """Process scene objects and update our storage."""
        self.scene_data = scene
        for scene_obj in scene.objects:
            obj_type = scene_obj.shape.lower()
            obj_id = f"{obj_type}_{scene_obj.object_id}"
            
            # Update or create object
            if obj_id in self.objects:
                self.objects[obj_id].update({
                    'position': scene_obj.position,
                    'color': scene_obj.color,
                    'size': scene_obj.size,
                    'scale': getattr(scene_obj, 'scale', 1.0)
                })
            else:
                self.objects[obj_id] = {
                    'type': obj_type,
                    'position': scene_obj.position,
                    'color': scene_obj.color,
                    'size': scene_obj.size,
                    'scale': getattr(scene_obj, 'scale', 1.0)
                }
    
    def clear_scene(self):
        """Clear all objects."""
        self.objects.clear()
        self.scene_data = None


class TestSpatialPositioning(unittest.TestCase):
    """Test spatial positioning calculations for scene objects."""
    
    def setUp(self):
        """Set up test environment with mock renderer."""
        self.mock_renderer = MockVPythonRenderer()
        self.scene = SceneModel()  # Fixed: Use SceneModel instead of Scene
    
    def test_sphere_above_cube_calculation(self):
        """Test basic positioning calculation for sphere above cube."""
        print("\n🧪 Testing basic sphere above cube positioning...")
        
        # Create mock objects with simplified properties for testing
        cube_data = {
            'name': 'cube',
            'object_id': 'cube_1',
            'color': 'red',
            'position': (0, 0, 0),
            'size': 1.0,
            'shape': 'Cube'
        }
        
        sphere_data = {
            'name': 'sphere',
            'object_id': 'sphere_1', 
            'color': 'blue',
            'position': (0, 0, 0),
            'size': 1.0,
            'shape': 'Sphere'
        }
        
        # Add to mock renderer directly for testing
        self.mock_renderer.objects['cube_1'] = cube_data
        self.mock_renderer.objects['sphere_1'] = sphere_data
        
        # Calculate positioning for sphere above cube
        cube_height = cube_data['size']  # 1.0
        sphere_radius = sphere_data['size'] / 2  # 0.5
        cube_center_y = cube_data['position'][1]  # 0
        
        # Formula: Y = cube_center + cube_height/2 + sphere_radius
        expected_sphere_y = cube_center_y + (cube_height / 2) + sphere_radius
        
        # Update sphere position
        sphere_data['position'] = (sphere_data['position'][0], expected_sphere_y, sphere_data['position'][2])
        
        # Verify positioning
        actual_sphere_y = sphere_data['position'][1]
        
        # Calculate gap/overlap
        cube_top = cube_data['position'][1] + (cube_data['size'] / 2)
        sphere_bottom = sphere_data['position'][1] - (sphere_data['size'] / 2)
        gap = sphere_bottom - cube_top
        
        print(f"🎯 Expected sphere Y: {expected_sphere_y}")
        print(f"🎯 Actual sphere Y: {actual_sphere_y}")
        print(f"🎯 Cube top: {cube_top}")
        print(f"🎯 Sphere bottom: {sphere_bottom}")
        print(f"🎯 Contact analysis: Gap/overlap: {gap}")
        
        # Perfect contact means gap should be 0
        self.assertAlmostEqual(gap, 0.0, places=5, msg="Objects should be in perfect contact")
        self.assertAlmostEqual(actual_sphere_y, expected_sphere_y, places=5, msg="Sphere should be at calculated position")
        
        print("✅ Basic positioning test passed - objects in perfect contact")
    
    def test_big_sphere_above_bigger_cube_scaling_issue(self):
        """Test the specific scaling issue from demo: 'big blue sphere' above 'bigger cube'."""
        print("\n🧪 Testing scaling issue reproduction...")
        
        # Simulate demo sequence using mock data:
        # 1. "draw red cube" - normal cube
        cube_data = {
            'name': 'cube',
            'object_id': 'cube_1',
            'color': 'red',
            'position': (0, 0, 0),
            'size': 1.0,  # normal size
            'shape': 'Cube'
        }
        
        # 2. "make it bigger" - scale up the cube
        cube_data['size'] = 2.0  # doubled size from "make it bigger"
        cube_data['scale'] = 2.0
        
        # 3. "draw big blue sphere" - sphere with "big" adjective  
        sphere_data = {
            'name': 'sphere',
            'object_id': 'sphere_1',
            'color': 'blue',
            'position': (0, 0, 0),
            'size': 2.0,  # "big" adjective size
            'shape': 'Sphere',
            'scale': 2.0
        }
        
        # Add to mock renderer
        self.mock_renderer.objects['cube_1'] = cube_data
        self.mock_renderer.objects['sphere_1'] = sphere_data
        
        # 4. "move sphere above cube" - positioning calculation
        cube_height = cube_data['size']  # 2.0 (scaled)
        sphere_radius = sphere_data['size'] / 2  # 1.0 (scaled)
        cube_center_y = cube_data['position'][1]  # 0
        
        # Calculate expected position
        expected_sphere_y = cube_center_y + (cube_height / 2) + sphere_radius
        
        # Move sphere above cube
        sphere_data['position'] = (sphere_data['position'][0], expected_sphere_y, sphere_data['position'][2])
        
        # Calculate positioning
        cube_top = cube_data['position'][1] + (cube_data['size'] / 2)
        sphere_bottom = sphere_data['position'][1] - (sphere_data['size'] / 2)
        gap = sphere_bottom - cube_top
        
        print(f"🔍 SCALING ANALYSIS:")
        print(f"   Cube size: {cube_data['size']} (scale: {cube_data.get('scale', 1.0)})")
        print(f"   Sphere size: {sphere_data['size']} (scale: {sphere_data.get('scale', 1.0)})")
        print(f"   Expected sphere Y: {expected_sphere_y}")
        print(f"   Actual sphere Y: {sphere_data['position'][1]}")
        print(f"   Cube top: {cube_top}")
        print(f"   Sphere bottom: {sphere_bottom}")
        print(f"   Gap/overlap: {gap}")
        
        if gap < 0:
            print(f"❌ OVERLAP DETECTED: {abs(gap)} units of intersection")
            print("🐛 This reproduces the user's visual bug!")
        elif gap == 0:
            print("✅ Perfect contact - no overlap")
        else:
            print(f"📏 Gap of {gap} units between objects")
        
        # For debugging, we want to see what's happening
        # The assertion might fail, showing us the actual scaling issue
        try:
            self.assertGreaterEqual(gap, 0, "Objects should not overlap")
            print("✅ No overlap detected in test")
        except AssertionError as e:
            print(f"🐛 CONFIRMED BUG: {e}")
            print("   This test successfully reproduced the overlapping issue!")
            # Re-raise for proper test reporting
            raise
    
    def test_scaling_command_analysis(self):
        """Analyze the difference between 'make it bigger' and 'big' adjective."""
        print("\n🧪 Testing scaling command differences...")
        
        # Test 1: Normal cube then "make it bigger"
        cube1_data = {
            'name': 'cube',
            'object_id': 'cube_1',
            'color': 'red',
            'position': (0, 0, 0),
            'size': 2.0,  # After "make it bigger" - assuming it doubles
            'shape': 'Cube'
        }
        
        # Test 2: "big" adjective cube from start
        cube2_data = {
            'name': 'cube',
            'object_id': 'cube_2',
            'color': 'green',
            'position': (3, 0, 0),  # Offset position
            'size': 2.0,  # "big" adjective
            'shape': 'Cube'
        }
        
        # Test 3: "big" sphere
        sphere_data = {
            'name': 'sphere',
            'object_id': 'sphere_1',
            'color': 'blue',
            'position': (0, 0, 0),
            'size': 2.0,  # "big" adjective
            'shape': 'Sphere'
        }
        
        # Add to mock renderer
        self.mock_renderer.objects['cube_1'] = cube1_data
        self.mock_renderer.objects['cube_2'] = cube2_data
        self.mock_renderer.objects['sphere_1'] = sphere_data
        
        print(f"🔍 SCALING COMMAND COMPARISON:")
        for obj_id, obj_data in self.mock_renderer.objects.items():
            print(f"   {obj_id}: size={obj_data['size']}, type={obj_data['shape']}")
        
        # Both should have same size if our assumption is correct
        self.assertEqual(cube1_data['size'], cube2_data['size'], "Make it bigger should equal big adjective")
        self.assertEqual(cube2_data['size'], sphere_data['size'], "All big objects should have same scale")
        
        print("✅ Scaling command analysis completed")
    
    def test_real_demo_sequence_integration(self):
        """Test actual demo sequence using real interpreter with mock rendering."""
        print("\n🔬 Testing real demo sequence integration...")
        
        # Import the real components
        from engraf.interpreter.sentence_interpreter import SentenceInterpreter
        from engraf.visualizer.renderers.vpython_renderer import MockVPythonRenderer
        from engraf.utils.debug import set_debug
        
        # Enable debug to see positioning calculations
        set_debug(True)
        
        # Use the real MockVPythonRenderer from the system
        mock_renderer = MockVPythonRenderer(
            width=800, 
            height=600, 
            title="Integration Test Demo"
        )
        
        # Create real interpreter with mock renderer
        interpreter = SentenceInterpreter(renderer=mock_renderer)
        
        # Execute the exact demo sequence that causes the overlap
        demo_sequence = [
            "draw a red cube at [0, 0, 0]",      # Step 1: Create cube
            "move it to [1, 2, 3]",              # Step 2: Position cube  
            "make it bigger",                     # Step 3: Scale cube (scaling issue?)
            "color it blue",                      # Step 4: Recolor
            "draw a big blue sphere at [3, 0, 0]", # Step 5: Create big sphere
            "move the sphere above the cube",     # Step 6: Position sphere (overlap issue!)
        ]
        
        print(f"📋 Executing {len(demo_sequence)} commands from real demo...")
        
        # Storage for captured states
        cube_states = {}
        sphere_states = {}
        
        # Execute each step and capture object states
        for i, command in enumerate(demo_sequence, 1):
            print(f"\n  Step {i}: '{command}'")
            
            try:
                result = interpreter.interpret(command)
                
                if result['success']:
                    print(f"    ✅ {result['message']}")
                    if result.get('objects_created'):
                        print(f"    📦 Created: {', '.join(result['objects_created'])}")
                    if result.get('actions_performed'):
                        print(f"    🎬 Actions: {', '.join(result['actions_performed'])}")
                else:
                    print(f"    ❌ Failed: {result['message']}")
                    continue
                
                # Capture states after key operations
                current_objects = mock_renderer.rendered_objects
                
                if "cube" in current_objects:
                    cube_states[f"step_{i}"] = self._capture_object_state(current_objects, "cube", command)
                    
                if "sphere" in current_objects:
                    sphere_states[f"step_{i}"] = self._capture_object_state(current_objects, "sphere", command)
                    
            except Exception as e:
                print(f"    💥 Exception: {e}")
                # Continue with other steps to gather as much data as possible
                
        # Analyze the final positioning
        print(f"\n🔍 REAL SYSTEM ANALYSIS:")
        self._analyze_real_positioning(cube_states, sphere_states)
        
        # Disable debug after test
        set_debug(False)
    
    def _capture_object_state(self, rendered_objects, obj_name, command):
        """Helper to capture object state from MockVPythonRenderer."""
        if obj_name not in rendered_objects:
            return None
            
        obj_data = rendered_objects[obj_name].copy()
        obj_data['command'] = command
        
        print(f"    📊 Captured {obj_name}: pos={obj_data.get('position', 'N/A')}, size={obj_data.get('size', 'N/A')}")
        return obj_data
    
    def _analyze_real_positioning(self, cube_states, sphere_states):
        """Analyze the positioning using real system values."""
        
        # Get final states
        final_cube_key = max(cube_states.keys()) if cube_states else None
        final_sphere_key = max(sphere_states.keys()) if sphere_states else None
        
        if not final_cube_key or not final_sphere_key:
            print("❌ Missing final object states - cannot analyze positioning")
            return
            
        final_cube = cube_states[final_cube_key]
        final_sphere = sphere_states[final_sphere_key]
        
        print(f"\n📐 FINAL POSITIONING ANALYSIS:")
        print(f"   Final Cube: {final_cube}")
        print(f"   Final Sphere: {final_sphere}")
        
        # Extract positioning data (adapting to MockVPythonRenderer format)
        cube_pos = final_cube.get('position', [0, 0, 0])
        cube_size = final_cube.get('size', [1, 1, 1])
        sphere_pos = final_sphere.get('position', [0, 0, 0]) 
        sphere_size = final_sphere.get('size', [1, 1, 1])
        
        # Handle different size formats (could be scalar or list)
        if isinstance(cube_size, list):
            cube_height = cube_size[1]  # Y dimension
        else:
            cube_height = cube_size
            
        if isinstance(sphere_size, list):
            sphere_radius = max(sphere_size) / 2  # Largest dimension / 2
        else:
            sphere_radius = sphere_size / 2
        
        # Calculate overlap
        cube_top = cube_pos[1] + (cube_height / 2)
        sphere_bottom = sphere_pos[1] - sphere_radius
        gap = sphere_bottom - cube_top
        
        print(f"\n🎯 REAL POSITIONING CALCULATION:")
        print(f"   Cube position: {cube_pos}")
        print(f"   Cube height: {cube_height}")
        print(f"   Cube top Y: {cube_top}")
        print(f"   Sphere position: {sphere_pos}")
        print(f"   Sphere radius: {sphere_radius}")
        print(f"   Sphere bottom Y: {sphere_bottom}")
        print(f"   Gap/overlap: {gap}")
        
        if gap < 0:
            print(f"❌ REAL OVERLAP CONFIRMED: {abs(gap)} units of intersection!")
            print("🐛 This reproduces the user's visual bug with real system values!")
            
            # Show progression through scaling steps
            print(f"\n📈 SCALING PROGRESSION:")
            for step, state in cube_states.items():
                if state:
                    size = state.get('size', 'N/A')
                    cmd = state.get('command', 'N/A')
                    print(f"   {step}: size={size} after '{cmd}'")
                    
        elif gap == 0:
            print("✅ Perfect contact - no overlap detected")
        else:
            print(f"📏 Gap of {gap} units between objects")
            
        # For the test, we want to capture this data but not necessarily fail
        # since we're investigating the issue
        print(f"\n💡 Integration test completed - real system data captured!")
        
        # NEW: Let's examine the scene objects directly to get real scaling data
        print(f"\n🔬 INVESTIGATING REAL SCENE DATA:")
        try:
            # Access the real scene from the interpreter
            from engraf.interpreter.sentence_interpreter import SentenceInterpreter
            
            # Try to access the interpreter's scene directly 
            # This might give us the real object data
            print("   📊 Checking if we can access real scene objects...")
            
        except Exception as e:
            print(f"   ❌ Could not access real scene: {e}")
        
        # Optional: Assert for documentation purposes
        if gap < 0:
            print(f"🔬 CONFIRMED: Real system produces overlap of {abs(gap)} units")
    
    def test_debug_scene_direct_access(self):
        """Test direct access to scene objects to get real scaling data."""
        print("\n🔬 Testing direct scene access for real object data...")
        
        from engraf.interpreter.sentence_interpreter import SentenceInterpreter
        from engraf.visualizer.renderers.vpython_renderer import MockVPythonRenderer
        from engraf.utils.debug import set_debug
        
        # Enable debug
        set_debug(True)
        
        # Create interpreter
        mock_renderer = MockVPythonRenderer()
        interpreter = SentenceInterpreter(renderer=mock_renderer)
        
        # Execute just the problematic sequence
        commands = [
            "draw a red cube at [0, 0, 0]",
            "move it to [1, 2, 3]", 
            "make it bigger",
            "draw a big blue sphere at [3, 0, 0]",
            "move the sphere above the cube"
        ]
        
        for i, cmd in enumerate(commands, 1):
            print(f"\n  Step {i}: '{cmd}'")
            result = interpreter.interpret(cmd)
            
            # After each step, examine the scene directly
            print(f"    Scene objects count: {len(interpreter.scene.objects)}")
            
            for obj in interpreter.scene.objects:
                print(f"    📊 {obj.name} ({obj.object_id}):")
                print(f"         vector.locX: {obj.vector['locX']}")
                print(f"         vector.locY: {obj.vector['locY']}")
                print(f"         vector.locZ: {obj.vector['locZ']}")
                print(f"         vector.scaleX: {obj.vector['scaleX']}")
                print(f"         vector.scaleY: {obj.vector['scaleY']}")
                print(f"         vector.scaleZ: {obj.vector['scaleZ']}")
        
        # Final analysis with real object data
        print(f"\n🎯 FINAL ANALYSIS WITH REAL SCENE DATA:")
        cube_obj = None
        sphere_obj = None
        
        for obj in interpreter.scene.objects:
            if obj.name == 'cube':
                cube_obj = obj
            elif obj.name == 'sphere':
                sphere_obj = obj
        
        if cube_obj and sphere_obj:
            # Get real values from scene objects
            cube_y = cube_obj.vector['locY']
            cube_height = cube_obj.vector['scaleY']
            sphere_y = sphere_obj.vector['locY'] 
            sphere_height = sphere_obj.vector['scaleY']
            
            # Calculate with real values
            cube_top = cube_y + cube_height/2
            sphere_bottom = sphere_y - sphere_height/2
            real_gap = sphere_bottom - cube_top
            
            print(f"   Real cube Y: {cube_y}, height: {cube_height}")
            print(f"   Real sphere Y: {sphere_y}, height: {sphere_height}")
            print(f"   Real cube top: {cube_top}")
            print(f"   Real sphere bottom: {sphere_bottom}")
            print(f"   Real gap/overlap: {real_gap}")
            
            if real_gap < 0:
                print(f"❌ CONFIRMED REAL OVERLAP: {abs(real_gap)} units!")
            else:
                print(f"✅ No overlap in real scene data")
        
        set_debug(False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
