#!/usr/bin/env python3
"""
ENGRAF Sentence Interpreter Demo

This demo shows the natural language processing capabilities of ENGRAF
for creating and manipulating 3D scenes using conversational commands.

Usage:
    python demo_sentence_interpreter.py

Requirements:
    - vpython (for 3D visualization)
    - numpy
    - All ENGRAF modules

Note: Run from the project root directory to ensure proper module imports.
"""

import os
import sys
import time

# Add the project root to the Python path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer


def demo_basic_commands():
    """Demonstrate basic sentence interpretation commands."""
    print("🎨 ENGRAF Sentence Interpreter Demo")
    print("=" * 50)
    print()
    
    # Create interpreter with visual renderer
    renderer = VPythonRenderer(
        width=1000,
        height=700,
        title="ENGRAF Sentence Interpreter Demo",
        headless=False
    )
    
    interpreter = SentenceInterpreter(renderer=renderer)
    
    # Demo sentences with positioning to spread objects out
    demo_sentences = [
        "draw a red cube at [0, 0, 0]",
        "move it to [2, 3, 4]",
        "make it bigger",
        "color it blue",
        "draw a big blue sphere at [3, 0, 0]",
        "move it above the cube",
        "draw a green cylinder at [-3, 0, 0]",
        "rotate it by 90 degrees",
        "draw a small yellow pyramid at [0, 3, 0]",
        "draw a purple box at [0, -3, 0]"
    ]
    
    print("🎯 Running basic drawing commands...")
    print()
    
    for i, sentence in enumerate(demo_sentences, 1):
        print(f"{i}. Processing: '{sentence}'")
        
        try:
            result = interpreter.interpret(sentence)
            
            if result['success']:
                print(f"   ✅ Success: {result['message']}")
                if result['objects_created']:
                    print(f"   📦 Created: {', '.join(result['objects_created'])}")
                if result['actions_performed']:
                    print(f"   🎬 Actions: {', '.join(result['actions_performed'])}")
            else:
                print(f"   ❌ Failed: {result['message']}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
        
        print()
        
        # Add a 2-second pause between demo sentences for better visualization
        if i < len(demo_sentences):
            time.sleep(2)
    
    print("✅ Basic demo complete! Objects created in 3D scene.")
    print()
    
    return interpreter


def interactive_mode(interpreter):
    """Run interactive mode for sentence interpretation."""
    print("🎮 Interactive Mode")
    print("=" * 30)
    print("Enter natural language commands to create and modify 3D objects.")
    print("Type 'help' for examples, 'summary' for scene info, 'temporal' for time travel status,")
    print("'clear' to reset, 'quit' to exit.")
    print()
    
    while True:
        try:
            user_input = input("🗣️  Enter command: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\n📚 Example commands:")
                print("  • draw a red cube at [1, 2, 3]")
                print("  • make it bigger")
                print("  • color it blue")
                print("  • move the cube to [5, 0, 0]")
                print("  • rotate it by 45 degrees")
                print("  • remove the sphere")
                print("  • go back in time")
                print("  • go forward in time")
                print("  • clear (reset scene)")
                print("  • summary (show current objects)")
                print("  • temporal (show time travel status)")
                print("  • quit (exit)")
                print()
                continue
            
            elif user_input.lower() == 'summary':
                summary = interpreter.get_scene_summary()
                print(f"\n📊 Scene Summary:")
                print(f"   Objects: {summary['object_count']}")
                if summary['objects']:
                    for obj_info in summary['objects']:
                        print(f"   • {obj_info}")
                else:
                    print("   • (no objects)")
                print()
                continue
            
            elif user_input.lower() == 'temporal':
                status = interpreter.get_temporal_status()
                print(f"\n🕰️ Temporal Status:")
                print(f"   Current scene: {status['current_scene_index']}")
                print(f"   Total scenes: {status['total_scenes']}")
                print(f"   Can go back: {status['can_go_back']}")
                print(f"   Can go forward: {status['can_go_forward']}")
                print()
                continue
            
            elif user_input.lower() == 'clear':
                print("\n🧹 Clearing scene...")
                interpreter.clear_scene()
                print("✅ Scene cleared!")
                print()
                continue
            
            # Regular sentence interpretation
            result = interpreter.interpret(user_input)
            
            if isinstance(result, dict):
                if result.get('success', False):
                    print(f"✅ {result.get('message', 'Success')}")
                    
                    # Show additional details if available
                    if result.get('objects_created'):
                        print(f"   📦 Created: {', '.join(result['objects_created'])}")
                    if result.get('objects_modified'):
                        print(f"   🔧 Modified: {', '.join(result['objects_modified'])}")
                    if result.get('objects_removed'):
                        print(f"   🗑️ Removed: {', '.join(result['objects_removed'])}")
                    if result.get('actions_performed'):
                        print(f"   🎬 Actions: {', '.join(result['actions_performed'])}")
                else:
                    print(f"❌ {result.get('message', 'Failed')}")
            else:
                print(f"✅ {result}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"💥 Error: {e}")
            print()


def main():
    """Main function to run the demo."""
    print("🚀 Starting ENGRAF Sentence Interpreter Demo")
    print()
    
    try:
        # Run basic demo
        interpreter = demo_basic_commands()
        
        # Go directly to interactive mode
        print("🎮 Entering interactive mode...")
        print()
        interactive_mode(interpreter)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install vpython")
        print("💡 Run from the project root directory")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
