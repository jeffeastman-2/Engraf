#!/usr/bin/env python3
"""
Demo: LATN Layer 4 - Verb Phrase Formation

This demo showcases Layer 4 of the LATN (Layered Augmented Transition Network) system,
which processes prepositional phrases from Layer 3 and forms verb phrases for action commands.

Layer 4 demonstrates:
- Verb phrase formation from verbs, NPs, and PPs
- Action command tokenization
- VP attachment and resolution
- Multi-hypothesis VP alternatives
- Command structure validation and confidence scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info, get_common_test_phrases, process_test_phrase_category


def main():
    """Main demo function using standardized scene and test phrases."""
    print("🔗 LATN Layer 4: Verb Phrase Formation Demo")
    print("=" * 60)
    
    # Setup standardized demo scene
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    # Get standardized test phrases
    test_phrases_dict = get_common_test_phrases()
    
    # Create executor
    executor = LATNLayerExecutor()
    
    # Process test phrases with tokenization only (no grounding)
    process_test_phrase_category(executor.execute_layer4, test_phrases_dict, enable_grounding=False)


if __name__ == "__main__":
    main()
