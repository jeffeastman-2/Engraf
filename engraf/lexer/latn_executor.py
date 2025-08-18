#!/usr/bin/env python3
"""
LATN (Layer-Aware Tokenization Network) Integrated Executor

This module provides a unified interface for LATN tokenization with semantic grounding.
It coordinates all layers of LATN processing and integrates with scene understanding.

This is a high-level convenience wrapper around the LATNLayerExecutor.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from engraf.lexer.latn_layer_executor import LATNLayerExecutor, Layer1Result, Layer2Result, Layer3Result
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.lexer.semantic_grounding_layer2 import Layer2GroundingResult
from engraf.lexer.semantic_grounding_layer3 import Layer3GroundingResult


@dataclass
class LATNProcessingResult:
    """Complete result of LATN processing with semantic grounding."""
    layer3_result: Layer3Result
    
    # Overall processing metadata
    success: bool
    confidence: float
    description: str = ""
    
    # Convenience properties to access nested data
    @property
    def layer1_result(self) -> Layer1Result:
        return self.layer3_result.layer2_result.layer1_result
    
    @property
    def layer2_result(self) -> Layer2Result:
        return self.layer3_result.layer2_result
    
    @property
    def noun_phrases(self) -> List[NounPhrase]:
        return self.layer2_result.noun_phrases
    
    @property
    def prepositional_phrases(self) -> List[PrepositionalPhrase]:
        return self.layer3_result.prepositional_phrases
    
    @property
    def np_grounding_results(self) -> List[Layer2GroundingResult]:
        return self.layer2_result.grounding_results
    
    @property
    def pp_grounding_results(self) -> List[Layer3GroundingResult]:
        return self.layer3_result.grounding_results


class LATNExecutor:
    """Unified executor for LATN tokenization with semantic grounding.
    
    This is a high-level convenience wrapper around LATNLayerExecutor.
    """
    
    def __init__(self, scene_model: Optional[SceneModel] = None):
        self.layer_executor = LATNLayerExecutor(scene_model)
    
    def process_sentence(self, sentence: str, enable_semantic_grounding: bool = True, 
                        return_all_matches: bool = False) -> LATNProcessingResult:
        """Process a sentence through all LATN layers with optional semantic grounding.
        
        Args:
            sentence: Input sentence to process
            enable_semantic_grounding: Whether to perform semantic grounding (requires scene_model)
            return_all_matches: Whether to return all possible groundings or just the best
            
        Returns:
            LATNProcessingResult with complete processing information
        """
        # Execute all layers through Layer 3
        layer3_result = self.layer_executor.execute_layer3(
            sentence, enable_semantic_grounding, return_all_matches
        )
        
        return LATNProcessingResult(
            layer3_result=layer3_result,
            success=layer3_result.success,
            confidence=layer3_result.confidence,
            description=layer3_result.description
        )
    
    def update_scene_model(self, scene_model: SceneModel):
        """Update the scene model used for semantic grounding."""
        self.layer_executor.update_scene_model(scene_model)
    
    def process_with_detailed_analysis(self, sentence: str) -> Dict[str, Any]:
        """Process sentence and return detailed analysis for debugging/inspection."""
        # Use the layer executor's analysis function
        return self.layer_executor.get_layer_analysis(sentence, target_layer=3)
