#!/usr/bin/env python3
"""
Progressive debug to find which sentence causes the extra sphere issue
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer


def test_progressive_sentences():
    """Test sentences progressively to identify the problematic one."""
    
    all_sentences = [
        "draw a red cube at [0, 0, 0]",
        "draw a big blue sphere at [3, 0, 0]",
        "draw a small green cylinder at [6, 0, 0]",
        "draw a tall yellow cone at [9, 0, 0]",
        "draw a purple pyramid at [0, 3, 0]",
        "draw a orange arch at [3, 3, 0]",
        "draw a brown table at [6, 3, 0]",
        # Last sentence removed as user mentioned
    ]
    
    # Test each progressive subset
    for num_sentences in range(1, len(all_sentences) + 1):
        print(f"\n{'='*50}")
        print(f"🧪 Testing with {num_sentences} sentences")
        print(f"{'='*50}")
        
        # Create fresh interpreter for each test
        renderer = VPythonRenderer(
            width=800,
            height=600,
            title=f"Test {num_sentences} sentences",
            headless=False
        )
        
        interpreter = SentenceInterpreter(renderer=renderer)
        
        # Run the subset of sentences
        sentences_to_test = all_sentences[:num_sentences]
        
        for i, sentence in enumerate(sentences_to_test, 1):
            print(f"{i}. Processing: '{sentence}'")
            
            try:
                result = interpreter.interpret(sentence)
                
                if result['success']:
                    print(f"   ✅ Success: {result['message']}")
                    if result['objects_created']:
                        print(f"   📦 Created: {', '.join(result['objects_created'])}")
                else:
                    print(f"   ❌ Failed: {result['message']}")
                    
            except Exception as e:
                print(f"   💥 Error: {e}")
        
        # Check VPython objects
        try:
            import vpython as vp
            print(f"📊 VPython objects in scene: {len(vp.scene.objects)}")
            print(f"📊 Scene model objects: {len(interpreter.scene.objects)}")
            
            # List all VPython objects
            print("🔍 VPython objects:")
            for i, obj in enumerate(vp.scene.objects):
                print(f"   {i}: {type(obj).__name__} at {obj.pos}")
                
        except Exception as e:
            print(f"Error checking VPython objects: {e}")
        
        user_input = input(f"\n❓ After {num_sentences} sentences, do you see the extra white sphere? (y/n/q to quit): ").strip().lower()
        
        if user_input == 'q':
            print("👋 Quitting debug session")
            break
        elif user_input == 'y':
            print(f"🎯 Found it! The extra sphere appears after sentence {num_sentences}: '{sentences_to_test[-1]}'")
            break
        elif user_input == 'n':
            print("✅ No extra sphere yet, continuing...")
            continue
        else:
            print("⚠️  Unknown response, assuming no extra sphere, continuing...")
            continue
    
    print(f"\n🏁 Debug session complete")


if __name__ == "__main__":
    test_progressive_sentences()
