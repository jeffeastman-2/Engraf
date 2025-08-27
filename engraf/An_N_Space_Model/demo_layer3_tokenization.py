#!/usr/bin/env python3
"""
Demo: LATN Layer 3 - Prepositional Phrase Formation

This demo showcases Layer 3 of the LATN (Layered Augmented Transition Network) system,
which processes noun phrases from Layer 2 and forms prepositional phrases for spatial relationships.

Layer 3 demonstrates:
- Prepositional phrase formation from NPs and prepositions
- Spatial relationship tokenization
- PP attachment and resolution
- Multi-hypothesis PP alternatives
- Grammatical validation and confidence scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info, get_common_test_phrases, process_test_phrase_category


def main():
    """Main demo function using standardized scene and test phrases."""
    print("🔗 LATN Layer 3: Prepositional Phrase Formation Demo")
    print("=" * 60)
    
    # Setup standardized demo scene
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    # Get standardized test phrases
    test_phrases_dict = get_common_test_phrases()
    
    # Create executor
    executor = LATNLayerExecutor()
    
    # Process test phrases with tokenization only (no grounding)
    process_test_phrase_category(executor.execute_layer3, test_phrases_dict, enable_grounding=False)


if __name__ == "__main__":
    main()
