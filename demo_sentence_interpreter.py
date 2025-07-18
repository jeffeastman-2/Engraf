#!/usr/bin/env python3
"""
Sentence Interpreter Demo

This script demonstrates the ENGRAF sentence interpreter by processing
natural language commands and rendering the results in 3D.

Usage:
    python demo_sentence_interpreter.py

Features demonstrated:
- Simple object creation
- Color and size adjectives
- Multiple objects
- Interactive command processing
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

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
        "draw a big blue sphere at [3, 0, 0]",
        "draw a small green cylinder at [6, 0, 0]",
        "draw a tall yellow cone at [9, 0, 0]",
        "draw a purple pyramid at [0, 3, 0]",
        "draw a orange arch at [3, 3, 0]",
        "draw a brown table at [6, 3, 0]",
        "draw a very tall white cube at [-2,0,0] and a huge black sphere at [-9, 3, 0]"
    ]
    
    print("🎯 Running demo sentences...")
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
    
    # Show scene summary
    summary = interpreter.get_scene_summary()
    print("📊 Scene Summary:")
    print(f"   Total objects: {summary['total_objects']}")
    print(f"   Object types: {', '.join(summary['object_types'])}")
    print(f"   Commands executed: {summary['execution_history']}")
    print()
    
    return interpreter


def demo_transform_commands(interpreter):
    """Demonstrate transformation commands on existing objects."""
    print("🔄 Transform Demo - Modifying Existing Objects")
    print("=" * 50)
    print()
    
    # Transform sentences that work with existing objects
    transform_sentences = [
        "move the red cube to [2, 1, 0]",
        "xrotate the orange arch by 45 degrees",
        "scale the green cylinder by [1, 3, 3]",
        "move it to [0, 0, 3]"  # Test pronoun resolution
    ]
    
    print("🎯 Running transform sentences...")
    print()
    
    for i, sentence in enumerate(transform_sentences, 1):
        print(f"{i}. Processing: '{sentence}'")
        
        try:
            result = interpreter.interpret(sentence)
            
            if result['success']:
                print(f"   ✅ Success: {result['message']}")
                if result['objects_modified']:
                    print(f"   🔧 Modified: {', '.join(result['objects_modified'])}")
                if result['actions_performed']:
                    print(f"   🎬 Actions: {', '.join(result['actions_performed'])}")
            else:
                print(f"   ❌ Failed: {result['message']}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
        
        print()
    
    return interpreter


def interactive_mode(interpreter):
    """Run interactive mode for sentence interpretation."""
    print("🎮 Interactive Mode")
    print("=" * 30)
    print("Enter natural language commands to create and modify 3D objects.")
    print("Type 'help' for examples, 'summary' for scene info, 'clear' to reset, 'quit' to exit.")
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
                print_help()
                continue
                
            elif user_input.lower() == 'summary':
                summary = interpreter.get_scene_summary()
                print(f"📊 Scene Summary:")
                print(f"   Total objects: {summary['total_objects']}")
                print(f"   Object types: {', '.join(summary['object_types'])}")
                print(f"   Commands executed: {summary['execution_history']}")
                print()
                continue
                
            elif user_input.lower() == 'clear':
                interpreter.clear_scene()
                print("✨ Scene cleared!")
                print()
                continue
            
            # Process the sentence
            result = interpreter.interpret(user_input)
            
            if result['success']:
                print(f"✅ {result['message']}")
                if result['objects_created']:
                    print(f"📦 Created: {', '.join(result['objects_created'])}")
                if result['objects_modified']:
                    print(f"🔧 Modified: {', '.join(result['objects_modified'])}")
                if result['actions_performed']:
                    print(f"🎬 Actions: {', '.join(result['actions_performed'])}")
            else:
                print(f"❌ {result['message']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"💥 Error: {e}")
        
        print()


def print_help():
    """Print help information."""
    print("🆘 Help - Example Commands:")
    print("   draw a red cube")
    print("   draw a big blue sphere")
    print("   draw a small green cylinder")
    print("   draw a tall yellow cone")
    print("   draw a purple pyramid")
    print("   draw a orange arch")
    print("   draw a brown table")
    print("   draw a tiny cube and a huge sphere")
    print()
    print("💡 Supported:")
    print("   Objects: cube, sphere, cylinder, cone, pyramid, arch, table")
    print("   Colors: red, green, blue, yellow, purple, orange, white, black")
    print("   Sizes: big/large/huge, small/little/tiny, tall, wide")
    print("   Verbs: draw, create, make, build")
    print()
    print("🎛️  Special commands:")
    print("   help - Show this help")
    print("   summary - Show scene summary")
    print("   clear - Clear the scene")
    print("   quit - Exit the program")
    print()


def main():
    """Main function to run the demo."""
    print("🚀 Starting ENGRAF Sentence Interpreter Demo")
    print()
    
    try:
        # Run basic demo
        interpreter = demo_basic_commands()
        
        # Run transform demo
        demo_transform_commands(interpreter)
        
        # Ask user if they want interactive mode
        if input("🎮 Would you like to try interactive mode? (y/n): ").lower().startswith('y'):
            interactive_mode(interpreter)
        else:
            print("👋 Demo completed! Close the browser window to exit.")
            input("Press Enter to exit...")
        
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
