#!/usr/bin/env python3
"""
LATN Semantic Grounding System - Legacy File

This module provided semantic grounding capabilities for LATN tokenization layers.
It has been split into separate layer-specific files:

- semantic_grounding_layer2.py: Layer 2 NounPhrase grounding
- semantic_grounding_layer3.py: Layer 3 PrepositionalPhrase grounding

This file is kept for backward compatibility and imports from the new files.
"""

# Import layer-specific grounders for backward compatibility
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder, Layer2GroundingResult
from engraf.lexer.semantic_grounding_layer3 import Layer3SemanticGrounder, Layer3GroundingResult

# Legacy aliases
SemanticGrounder = Layer2SemanticGrounder  # Default to Layer 2 for backward compatibility
GroundingResult = Layer2GroundingResult    # Default to Layer 2 result type

# For backward compatibility - create a combined executor class
class LATNSemanticGroundingExecutor:
    """Legacy executor that coordinates semantic grounding across LATN layers.
    
    NOTE: This is deprecated. Use LATNLayerExecutor instead.
    """
    
    def __init__(self, scene_model):
        from engraf.visualizer.scene.scene_model import SceneModel
        self.scene_model = scene_model
        self.layer2_grounder = Layer2SemanticGrounder(scene_model)
        self.layer3_grounder = Layer3SemanticGrounder(scene_model)
    
    def ground_layer2_tokens(self, np_tokens, return_all_matches=False):
        """Ground all NounPhrase tokens from Layer 2."""
        return self.layer2_grounder.ground_multiple(np_tokens, return_all_matches)
    
    def ground_layer3_tokens(self, pp_tokens, return_all_matches=False):
        """Ground all PrepositionalPhrase tokens from Layer 3."""
        return self.layer3_grounder.ground_multiple(pp_tokens, return_all_matches)
    
    def ground_all_layers(self, np_tokens, pp_tokens, return_all_matches=False):
        """Ground tokens from both Layer 2 and Layer 3."""
        np_results = self.ground_layer2_tokens(np_tokens, return_all_matches)
        pp_results = self.ground_layer3_tokens(pp_tokens, return_all_matches)
        return np_results, pp_results
