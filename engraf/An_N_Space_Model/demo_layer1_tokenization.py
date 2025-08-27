#!/usr/bin/env python3
"""
Demo: LATN Layer 1 - Multi-Hypothesis Tokenization

This demo showcases Layer 1 of the LATN (Layered Augmented Transition Network) system,
which generates multiple tokenization hypotheses for phrases that reference the demo scene.

Layer 1 demonstrates:
- Basic tokenization of scene-relevant phrases
- Multi-hypothesis generation for ambiguous phrases
- Confidence scoring and hypothesis ranking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info, get_common_test_phrases, process_test_phrase_category


def demo_layer1_tokenization():
    """Demonstrate Layer 1 tokenization using only the demo scene and its test phrases."""
    print("🔤 LATN Layer 1: Scene-Based Tokenization Demo")
    print("=" * 60)
    
    # Set up the standardized demo scene
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    # Use ONLY the test phrases from the demo scene setup, organized by category
    test_phrases_dict = get_common_test_phrases()
    
    executor = LATNLayerExecutor()
    
    process_test_phrase_category(executor.execute_layer1, test_phrases_dict, enable_grounding=False)


def main():
    """Run the Layer 1 demo using ONLY the demo scene setup."""
    print("🚀 LATN Layer 1 Tokenization Demo")
    print("\nThis demo shows Layer 1 tokenization for phrases that reference")
    print("the standardized demo scene objects and relationships.\n")
    
    demo_layer1_tokenization()
    
    print("\n🎉 Demo Complete!")
    print("\nLayer 1 tokenized all phrases from the demo scene setup.")
    print("Next: Try demo_layer2_tokenization.py to see Layer 2 processing!")


if __name__ == "__main__":
    main()
