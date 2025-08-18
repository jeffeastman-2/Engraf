#!/usr/bin/env python3
"""
Demo: LATN Layer 4 - Verb Phrase Formation and Action Execution

This demo showcases Layer 4 of the LATN (Layered Augmented Transition Network) system,
which processes verb phrases and executes actions like object creation.

Layer 4 demonstrates:
- Verb phrase formation from tokens (Layers 1-3)
- Action verb recognition (create, move, rotate, delete)
- Object creation from verb phrases
- Scene object population for Layer 2 grounding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.utils.debug import set_debug


def demo_simple_creation_commands():
    """Demonstrate simple object creation commands."""
    print("🔧 LATN Layer 4: Simple Object Creation Demo")
    print("=" * 55)
    
    scene = SceneModel()
    executor = LATNLayerExecutor(scene)
    
    # Test cases showing simple creation commands
    creation_commands = [
        "create a red box",
        "make a blue sphere", 
        "build a green cube",
        "create a large red sphere",
        "make a small blue box"
    ]
    
    for command in creation_commands:
        print(f"\n🎯 Command: \"{command}\"")
        print("-" * 40)
        
        # Execute Layer 4 with action execution enabled
        result = executor.execute_layer4(command, enable_action_execution=True)
        
        if result.success:
            print(f"✅ Layer 4 Success!")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   VP Count: {len(result.verb_phrases)}")
            
            # Show the verb phrases formed
            for i, vp in enumerate(result.verb_phrases, 1):
                print(f"   VP {i}: {vp}")
                
            # Show scene objects created
            print(f"   Scene now has {len(scene.objects)} objects")
            
        else:
            print(f"❌ Layer 4 Failed: {result.description}")


def demo_scene_population():
    """Demonstrate how Layer 4 populates the scene with objects."""
    print("\n🏗️ Scene Population Demo")
    print("=" * 30)
    
    scene = SceneModel()
    executor = LATNLayerExecutor(scene)
    
    # Create multiple objects to populate the scene
    setup_commands = [
        "create a red box",
        "make a blue sphere",
        "build a green cube", 
        "create a large yellow sphere",
        "make a small purple box"
    ]
    
    print("Building scene with multiple objects...")
    for command in setup_commands:
        result = executor.execute_layer4(command, enable_action_execution=True)
        if result.success:
            print(f"  ✓ {command}")
        else:
            print(f"  ❌ {command} - Failed")
    
    print(f"\n📊 Final Scene Summary:")
    print(f"   Total objects: {len(scene.objects)}")
    
    for obj in scene.objects:
        color_name = "red" if obj.color == [1.0, 0.0, 0.0] else \
                    "blue" if obj.color == [0.0, 0.0, 1.0] else \
                    "green" if obj.color == [0.0, 1.0, 0.0] else \
                    "yellow" if obj.color == [1.0, 1.0, 0.0] else \
                    "purple" if obj.color == [1.0, 0.0, 1.0] else "gray"
        
        size_desc = "large" if max(obj.scale) > 1.5 else \
                   "small" if max(obj.scale) < 0.7 else "normal"
        
        print(f"   • {obj.object_id}: {size_desc} {color_name} {obj.shape}")
    
    return scene


def demo_vp_tokenization_only():
    """Demonstrate VP tokenization without action execution."""
    print("\n🔤 VP Tokenization Only Demo")
    print("=" * 35)
    
    executor = LATNLayerExecutor()  # No scene model
    
    # Test various verb phrase patterns
    vp_examples = [
        "create a red box",
        "move the sphere",
        "rotate 45 degrees", 
        "delete the blue cube",
        "scale the object",
        "build a very large green sphere"
    ]
    
    for phrase in vp_examples:
        print(f"\n📝 Input: \"{phrase}\"")
        print("-" * 25)
        
        # Execute with action execution disabled
        result = executor.execute_layer4(phrase, enable_action_execution=False)
        
        if result.success:
            print(f"✅ VP tokenization successful")
            print(f"   Hypotheses: {len(result.hypotheses)}")
            
            # Show the VP tokens
            if result.hypotheses:
                best_hyp = result.hypotheses[0]
                token_words = [t.word for t in best_hyp.tokens]
                print(f"   Tokens: {token_words}")
                
                # Show VP replacements
                if best_hyp.vp_replacements:
                    print(f"   VP Replacements: {len(best_hyp.vp_replacements)}")
                    for j, vp_replacement in enumerate(best_hyp.vp_replacements, 1):
                        start_idx, end_idx, vp_token = vp_replacement
                        print(f"     {j}. {vp_token.word}")
        else:
            print(f"❌ VP tokenization failed: {result.description}")


def demo_layer2_grounding_with_created_objects():
    """Demonstrate Layer 2 grounding using objects created by Layer 4."""
    print("\n🔗 Layer 2 Grounding with Created Objects Demo")
    print("=" * 52)
    
    # First, create a scene with objects using Layer 4
    scene = demo_scene_population()
    
    # Now test Layer 2 grounding with the populated scene
    executor = LATNLayerExecutor(scene)
    
    grounding_phrases = [
        "the red box",
        "the blue sphere", 
        "a green object",
        "the large sphere",
        "small objects"
    ]
    
    print("\nTesting Layer 2 grounding against created objects:")
    
    for phrase in grounding_phrases:
        print(f"\n🔍 Grounding: \"{phrase}\"")
        print("-" * 30)
        
        # Use Layer 2 with grounding enabled
        result = executor.execute_layer2(phrase, enable_semantic_grounding=True)
        
        if result.success:
            print(f"✅ Layer 2 grounding successful")
            print(f"   NP Count: {len(result.noun_phrases)}")
            print(f"   Grounding Results: {len(result.grounding_results)}")
            
            # Show grounding results
            for gr in result.grounding_results:
                if gr.success and gr.resolved_object:
                    print(f"   → Resolved to: {gr.resolved_object.object_id}")
                else:
                    print(f"   → No resolution: {gr.description}")
        else:
            print(f"❌ Layer 2 grounding failed")


def main():
    """Run all Layer 4 demonstrations."""
    # Suppress debug output for clean demo
    set_debug(False)
    
    print("LATN Layer 4 Verb Phrase Demo")
    print("=============================")
    print()
    print("This demo shows how Layer 4 of the Layered Augmented Transition Network")
    print("processes verb phrases and executes actions to create scene objects.")
    print()
    
    # Run the demonstrations
    demo_simple_creation_commands()
    demo_vp_tokenization_only()
    demo_layer2_grounding_with_created_objects()
    
    print("\n" + "=" * 60)
    print("Demo complete! Layer 4 successfully demonstrated:")
    print("✓ Verb phrase formation and tokenization")
    print("✓ Action execution (object creation)")
    print("✓ Scene population for Layer 2 grounding")
    print("✓ Integration with lower layers")


if __name__ == "__main__":
    main()
