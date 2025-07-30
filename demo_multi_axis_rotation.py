#!/usr/bin/env python3
"""
Multi-Axis Rotation Demo Script

This script demonstrates the new multi-axis rotation capabilities 
implemented in the ENGRAF sentence interpreter.

Features demonstrated:
- Vector coordinate rotation [x,y,z]
- Symmetric and asymmetric rotations
- Negative rotation values
- Multiple object rotation
- Proper rotation vs scaling classification
"""

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer


def print_separator(title):
    """Print a formatted section separator."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_object_state(interpreter, obj_index=0, label="Object"):
    """Print the current rotation and scale state of an object."""
    if interpreter.scene.objects and obj_index < len(interpreter.scene.objects):
        obj = interpreter.scene.objects[obj_index]
        print(f"{label} ({obj.object_id}):")
        print(f"  Rotation: X={obj.vector['rotX']:>6.1f}°, Y={obj.vector['rotY']:>6.1f}°, Z={obj.vector['rotZ']:>6.1f}°")
        print(f"  Scale:    X={obj.vector['scaleX']:>6.1f},  Y={obj.vector['scaleY']:>6.1f},  Z={obj.vector['scaleZ']:>6.1f}")
    else:
        print(f"{label}: No object found")


def demo_basic_multi_axis_rotation():
    """Demonstrate basic multi-axis rotation functionality."""
    print_separator("Basic Multi-Axis Rotation")
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create a test object
    print("Creating a red cube...")
    result = interpreter.interpret('draw a red cube')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "Initial state")
    
    # Test symmetric rotation
    print("\n📐 Applying symmetric rotation [45,45,45]...")
    result = interpreter.interpret('rotate it by [45,45,45]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After rotation")
    
    return interpreter


def demo_asymmetric_rotation():
    """Demonstrate asymmetric multi-axis rotation."""
    print_separator("Asymmetric Multi-Axis Rotation")
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create a test object
    print("Creating a blue sphere...")
    result = interpreter.interpret('draw a blue sphere')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "Initial state")
    
    # Test asymmetric rotation
    print("\n📐 Applying asymmetric rotation [90,0,45]...")
    result = interpreter.interpret('rotate it by [90,0,45]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After rotation")
    
    return interpreter


def demo_negative_rotation():
    """Demonstrate negative rotation values."""
    print_separator("Negative Rotation Values")
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create a test object
    print("Creating a green cylinder...")
    result = interpreter.interpret('draw a green cylinder')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "Initial state")
    
    # Test negative rotation
    print("\n📐 Applying negative rotation [-45,-30,-15]...")
    result = interpreter.interpret('rotate it by [-45,-30,-15]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After rotation")
    
    return interpreter


def demo_incremental_rotation():
    """Demonstrate incremental rotation (overwrites)."""
    print_separator("Incremental Rotation (Overwrites)")
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create a test object
    print("Creating a yellow cube...")
    result = interpreter.interpret('draw a yellow cube')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "Initial state")
    
    # First rotation
    print("\n📐 First rotation [30,60,90]...")
    result = interpreter.interpret('rotate it by [30,60,90]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After first rotation")
    
    # Second rotation (should overwrite)
    print("\n📐 Second rotation [0,180,0] (should overwrite)...")
    result = interpreter.interpret('rotate it by [0,180,0]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After second rotation")
    
    return interpreter


def demo_rotation_vs_scaling():
    """Demonstrate that rotation is properly classified vs scaling."""
    print_separator("Rotation vs Scaling Classification")
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create a test object
    print("Creating a purple sphere...")
    result = interpreter.interpret('draw a purple sphere')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "Initial state")
    
    # Test rotation command
    print("\n📐 Applying rotation [60,120,240]...")
    result = interpreter.interpret('rotate it by [60,120,240]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After rotation")
    print("🔍 Notice: Scale values remain at 1.0 (unchanged)")
    print("🔍 Notice: Only rotation values were modified")
    
    return interpreter


def demo_multiple_objects():
    """Demonstrate rotation with multiple objects."""
    print_separator("Multiple Objects Rotation")
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create multiple objects
    print("Creating multiple objects...")
    interpreter.interpret('draw a red cube')
    interpreter.interpret('draw a green sphere')
    interpreter.interpret('draw a blue cylinder')
    
    print(f"✅ Created {len(interpreter.scene.objects)} objects")
    
    # Show all objects
    for i, obj in enumerate(interpreter.scene.objects):
        print_object_state(interpreter, i, f"Object {i+1}")
    
    # Rotate the most recent object (blue cylinder)
    print("\n📐 Rotating the most recent object 'it' by [15,25,35]...")
    result = interpreter.interpret('rotate it by [15,25,35]')
    print(f"✅ {result['message']}")
    
    print("\nFinal states:")
    for i, obj in enumerate(interpreter.scene.objects):
        print_object_state(interpreter, i, f"Object {i+1}")
    
    print("🔍 Notice: Only the last object (blue cylinder) was rotated")
    
    return interpreter


def demo_edge_cases():
    """Demonstrate edge cases and special scenarios."""
    print_separator("Edge Cases and Special Scenarios")
    
    interpreter = SentenceInterpreter(renderer=MockRenderer())
    
    # Create a test object
    print("Creating a white cube...")
    result = interpreter.interpret('draw a white cube')
    print(f"✅ {result['message']}")
    
    # Test zero rotation
    print("\n📐 Testing zero rotation [0,0,0]...")
    result = interpreter.interpret('rotate it by [0,0,0]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After zero rotation")
    
    # Test large values
    print("\n📐 Testing large rotation values [720,450,900]...")
    result = interpreter.interpret('rotate it by [720,450,900]')
    print(f"✅ {result['message']}")
    print_object_state(interpreter, 0, "After large rotation")
    print("🔍 Notice: Large values (>360°) are accepted without modulo")
    
    return interpreter


def main():
    """Run all demonstration scenarios."""
    print("🚀 ENGRAF Multi-Axis Rotation Demonstration")
    print("This demo showcases the new vector coordinate rotation system")
    
    try:
        # Run all demos
        demo_basic_multi_axis_rotation()
        demo_asymmetric_rotation()
        demo_negative_rotation()
        demo_incremental_rotation()
        demo_rotation_vs_scaling()
        demo_multiple_objects()
        demo_edge_cases()
        
        print_separator("Demo Complete!")
        print("🎉 All multi-axis rotation features demonstrated successfully!")
        print("\nKey Features Shown:")
        print("  ✓ Vector coordinate rotation with [x,y,z] syntax")
        print("  ✓ Symmetric and asymmetric rotations")  
        print("  ✓ Negative rotation values support")
        print("  ✓ Proper rotation vs scaling classification")
        print("  ✓ Multiple object handling with pronouns")
        print("  ✓ Edge cases (zero, large values)")
        print("\nThe ENGRAF system now supports sophisticated 3D rotation commands! 🎯")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
