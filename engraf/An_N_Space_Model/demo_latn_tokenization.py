#!/usr/bin/env python3
"""
Demo: All LATN Layers   

This demo showcases all Layers of the LATN (Layered Augmented Transition Network) system,
using a standardized scene and test phrases.

"""

import sys
import os
from engraf.utils.debug import set_debug
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info, get_common_test_phrases, process_test_phrase_category

def run_layer_and_summarize(executor, layer, phrase, enable_semantic_grounding):
    # Call the executor method
    result = executor(phrase, enable_semantic_grounding=enable_semantic_grounding)
    if result.success:
        print(f"✅ Layer {layer} Generated {len(result.hypotheses)} tokenization hypothesis(es)")
        
        for i, hyp in enumerate(result.hypotheses[:3], 1):  # Show top 3
            tokens = [t.word for t in hyp.tokens]
            print(f"  {i}. {tokens}")
            print(f"     Confidence: {hyp.confidence:.3f}")
            hyp.print_tokens()
            
    else:
        print("❌ Tokenization failed")

def main():
    """Main demo function using no scene and test phrases."""
    print("🔗 LATN Overall Demo")
    print("=" * 60)
    #set_debug(True)

    # Get standardized test phrases
    test_phrases_dict = get_common_test_phrases()
    
    # Create executor
    executor = LATNLayerExecutor()
    
    # Process each category separately  
    for category_name, phrases in test_phrases_dict.items():
        print(f"\n=== {category_name.replace('_', ' ').title()} ===")
        
        for phrase in phrases:
            print(f"\n📝 Input: \"{phrase}\"")
            print("-" * 30)
            executor.execute_layer5(phrase, report=True)
        print("========")
            


if __name__ == "__main__":
    main()
