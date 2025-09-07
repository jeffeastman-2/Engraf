#!/usr/bin/env python3
"""Quick test of enhanced CONJ-NP display functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1
from engraf.lexer.latn_tokenizer_layer2 import latn_tokenize_layer2

def test_coordination_display():
    """Test the enhanced CONJ-NP token display."""
    text = ["the red cube and the blue sphere",
            "the red cube under the table and the blue sphere",
            "the red cube under the table and the blue sphere above the table"]
    print()

    executor = LATNLayerExecutor()
    for t in text:
        print()
        print(f'Testing input: "{t}"')
        executor.execute_layer2(t, report=True)

if __name__ == "__main__":
    test_coordination_display()
