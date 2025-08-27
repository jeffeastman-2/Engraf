#!/usr/bin/env python3
"""
Demo: LATN Layer 3 - Prepositional Phrase Grounding

This demo showcases Layer 3 semantic grounding which validates spatial relationships
and grounds prepositional phrases to actual spatial locations in a 3D scene.

Key concepts demonstrated:
1. Spatial relationship validation between grounded objects
2. Prepositional phrase grounding to spatial locations  
3. PP attachment resolution for complex spatial chains
4. Vector coordinate grounding (e.g., "to [3,4,5]")
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
    
    print("🔗 LATN Layer 3: Prepositional Phrase Grounding Demo")
    print("=" * 60)
    
    # Setup standardized demo scene
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    # Get standardized test phrases
    test_phrases_dict = get_common_test_phrases()
    
    # Create executor with scene model for grounding
    executor = LATNLayerExecutor(scene_model=scene)
    
    # Process test phrases with grounding enabled
    process_test_phrase_category(executor.execute_layer3, test_phrases_dict, enable_grounding=True)


if __name__ == "__main__":
    main()
