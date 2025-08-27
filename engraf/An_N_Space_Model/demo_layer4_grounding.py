#!/usr/bin/env python3
"""
Demo: LATN Layer 4 - Verb Phrase Grounding

This demo showcases Layer 4 semantic grounding which validates action commands
and grounds verb phrases to executable operations in a 3D scene.

Key concepts demonstrated:
1. Verb phrase grounding to scene operations
2. Action command validation and parameter extraction
3. VP semantic resolution for object manipulation
4. Command structure validation with confidence scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info, get_common_test_phrases, process_test_phrase_category
from engraf.utils.debug import set_debug


def main():
    """Main demo function using standardized scene and test phrases."""
    # Suppress debug output for clean demo
    set_debug(False)
    
    print("🔗 LATN Layer 4: Verb Phrase Grounding Demo")
    print("=" * 60)
    
    # Setup standardized demo scene
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    # Get standardized test phrases
    test_phrases_dict = get_common_test_phrases()
    
    # Create executor with scene model for grounding
    executor = LATNLayerExecutor(scene_model=scene)
    
    # Process test phrases with grounding enabled
    process_test_phrase_category(executor.execute_layer4, test_phrases_dict, enable_grounding=True)


if __name__ == "__main__":
    main()
