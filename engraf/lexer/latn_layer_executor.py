#!/usr/bin/env python3
"""
LATN Layer Executor

This module provides layered execution entry points for the LATN system.
Each layer automatically executes all prerequisite layers, allowing clean
entry at any layer for testing and processing.

Refactored for clean separation of concerns:
- Tokenization logic → respective tokenizer modules
- Grounding logic → respective grounder modules
- Coordination logic → this executor (lightweight)
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from engraf.lexer.latn_tokenizer_layer1 import latn_tokenize, TokenizationHypothesis
from engraf.lexer.latn_tokenizer_layer2 import latn_tokenize_layer2
from engraf.lexer.latn_tokenizer_layer3 import latn_tokenize_layer3
from engraf.lexer.latn_tokenizer_layer4 import latn_tokenize_layer4
from engraf.lexer.latn_tokenizer_layer5 import latn_tokenize_layer5, extract_sentence_phrases
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder, Layer2GroundingResult
from engraf.lexer.semantic_grounding_layer3 import Layer3SemanticGrounder, Layer3GroundingResult
from engraf.lexer.semantic_grounding_layer4 import Layer4SemanticGrounder, Layer4GroundingResult
from engraf.lexer.semantic_grounding_layer5 import Layer5SemanticGrounder, Layer5GroundingResult
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.sentence_phrase import SentencePhrase


@dataclass
class Layer1Result:
    """Result of Layer 1 execution (lexical tokenization)."""
    hypotheses: List[TokenizationHypothesis]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer2Result:
    """Result of Layer 2 execution (NP tokenization + grounding)."""
    layer1_result: Layer1Result
    hypotheses: List[TokenizationHypothesis]
    noun_phrases: List[NounPhrase]
    grounding_results: List[Layer2GroundingResult]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer3Result:
    """Result of Layer 3 execution (PP tokenization + grounding)."""
    layer2_result: Layer2Result
    hypotheses: List[TokenizationHypothesis]
    prepositional_phrases: List[PrepositionalPhrase]
    grounding_results: List[Layer3GroundingResult]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer4Result:
    """Result of Layer 4 execution (VP tokenization + execution)."""
    layer3_result: Layer3Result
    hypotheses: List[TokenizationHypothesis]
    verb_phrases: List[VerbPhrase]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer5Result:
    """Result of Layer 5 execution (sentence tokenization + execution)."""
    layer4_result: Layer4Result
    hypotheses: List[TokenizationHypothesis]
    sentence_phrases: List[SentencePhrase]
    grounding_results: List[Layer5GroundingResult]
    success: bool
    confidence: float
    description: str = ""


class LATNLayerExecutor:
    """Coordinates execution across all LATN layers with clean delegation to grounders."""
    
    def __init__(self, scene_model: Optional[SceneModel] = None):
        self.scene = scene_model
        self.layer2_grounder = Layer2SemanticGrounder(scene_model) if scene_model else None
        self.layer3_grounder = Layer3SemanticGrounder()  # Layer 3 doesn't use SceneModel
        self.layer4_grounder = Layer4SemanticGrounder(scene_model) if scene_model else None
        self.layer5_grounder = Layer5SemanticGrounder(scene_model) if scene_model else None
    
    def execute_layer1(self, sentence: str, enable_semantic_grounding=False) -> Layer1Result:
        """Execute Layer 1: Multi-hypothesis lexical tokenization.
        
        Args:
            sentence: Input sentence to tokenize
            
        Returns:
            Layer1Result with tokenization hypotheses
        """
        try:
            hypotheses = latn_tokenize(sentence)
            
            if not hypotheses or not sentence.strip():
                return Layer1Result(
                    hypotheses=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 1 tokenization failed for: '{sentence}'"
                )
            
            # Use confidence from best hypothesis
            best_confidence = hypotheses[0].confidence if hypotheses else 0.0
            
            return Layer1Result(
                hypotheses=hypotheses,
                success=True,
                confidence=best_confidence,
                description=f"Layer 1 tokenized '{sentence}' into {len(hypotheses)} hypotheses"
            )
            
        except Exception as e:
            return Layer1Result(
                hypotheses=[],
                success=False,
                confidence=0.0,
                description=f"Layer 1 failed: {e}"
            )
    
    def execute_layer2(self, sentence: str, enable_semantic_grounding: bool = True,
                      return_all_matches: bool = True) -> Layer2Result:
        """Execute Layer 2: NP tokenization (requires Layer 1).
        
        Args:
            sentence: Input sentence to process
            enable_semantic_grounding: Whether to perform semantic grounding
            return_all_matches: Whether to return all grounding matches
            
        Returns:
            Layer2Result with NP processing results
        """
        # First execute Layer 1
        layer1_result = self.execute_layer1(sentence)
        
        if not layer1_result.success:
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=[],
                noun_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 2 failed due to Layer 1 failure: {layer1_result.description}"
            )
        
        try:
            # Execute Layer 2 NP tokenization
            layer2_hypotheses = latn_tokenize_layer2(layer1_result.hypotheses)
            
            # Semantic grounding with hypothesis multiplication (if enabled)
            if enable_semantic_grounding and self.layer2_grounder:
                grounded_hypotheses, all_grounding_results = self.layer2_grounder.multiply_hypotheses_with_grounding(
                    layer2_hypotheses, return_all_matches
                )
            else:
                # No grounding - keep original hypotheses
                grounded_hypotheses = layer2_hypotheses
                all_grounding_results = []
            
            # Extract noun phrases from the (possibly grounded) hypotheses
            # The tokenizer creates NP tokens with _original_np references, so we can extract them even without grounding
            if self.layer2_grounder:
                all_noun_phrases = self.layer2_grounder.extract_noun_phrases(grounded_hypotheses)
            else:
                # Extract noun phrases directly from tokens when no grounder available
                all_noun_phrases = []
                for hypothesis in grounded_hypotheses:
                    for token in hypothesis.tokens:
                        if token._original_np is not None and token._original_np:
                            all_noun_phrases.append(token._original_np)
            
            # Calculate confidence based on best hypothesis
            layer2_confidence = grounded_hypotheses[0].confidence if grounded_hypotheses else layer1_result.confidence
            overall_confidence = (layer1_result.confidence + layer2_confidence) / 2
            
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=grounded_hypotheses,
                noun_phrases=all_noun_phrases,
                grounding_results=all_grounding_results,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 2 processed {len(all_noun_phrases)} noun phrases in {len(grounded_hypotheses)} hypotheses"
            )
            
        except Exception as e:
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=[],
                noun_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 2 failed: {e}"
            )
    
    def execute_layer3(self, sentence: str, enable_semantic_grounding: bool = True,
                      return_all_matches: bool = True) -> Layer3Result:
        """Execute Layer 3: PP tokenization (requires Layer 1-2).
        
        Args:
            sentence: Input sentence to process
            enable_semantic_grounding: Whether to perform semantic grounding
            return_all_matches: Whether to return all grounding matches
            
        Returns:
            Layer3Result with complete LATN processing results
        """
        # First execute Layer 2 (which includes Layer 1) - always enable grounding for Layer 2
        # Layer 3 spatial validation requires grounded NP tokens from Layer 2
        layer2_result = self.execute_layer2(sentence, enable_semantic_grounding=True, return_all_matches=return_all_matches)
        
        if not layer2_result.success:
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=[],
                prepositional_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 3 failed due to Layer 2 failure: {layer2_result.description}"
            )
        
        try:
            # Execute Layer 3 PP tokenization
            layer3_hypotheses = latn_tokenize_layer3(layer2_result.hypotheses)
            
            # Extract PrepositionalPhrase objects
            prepositional_phrases = self.layer3_grounder.extract_prepositional_phrases(layer3_hypotheses) if self.layer3_grounder else []
            
            # Layer 3 grounding - process PP attachments with spatial validation
            if enable_semantic_grounding and self.layer3_grounder:
                # Process PP attachment combinations with spatial validation
                grounded_hypotheses = self.layer3_grounder.process_pp_attachments(
                    layer3_hypotheses, return_all_matches=return_all_matches
                )
                
                # Use the grounded hypotheses as the final result
                final_hypotheses = grounded_hypotheses
                
                # Create grounding results for compatibility (could be empty list)
                grounding_results = []
            else:
                # No grounding - use original hypotheses
                final_hypotheses = layer3_hypotheses
                grounding_results = []
            
            # Calculate confidence
            layer3_confidence = final_hypotheses[0].confidence if final_hypotheses else layer2_result.confidence
            overall_confidence = (layer2_result.confidence + layer3_confidence) / 2
            
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=final_hypotheses,
                prepositional_phrases=prepositional_phrases,
                grounding_results=grounding_results,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 3 processed {len(prepositional_phrases)} prepositional phrases"
            )
            
        except Exception as e:
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=[],
                prepositional_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 3 failed: {e}"
            )
    
    def execute_layer4(self, sentence: str, enable_semantic_grounding: bool = True,
                      return_all_matches: bool = True) -> Layer4Result:
        """Execute Layer 4: VP tokenization and semantic grounding (requires Layer 1-3).
        
        Args:
            sentence: Input sentence to process
            enable_semantic_grounding: Whether to perform semantic grounding
            
        Returns:
            Layer4Result with complete LATN processing results including verb phrase grounding
        """
        try:
            # Execute Layer 3 first (which includes Layer 1 and 2)
            # Layer 4 should always start from Layer 3 grounding results, not tokenization
            layer3_result = self.execute_layer3(sentence, enable_semantic_grounding=True, return_all_matches=return_all_matches)
            
            if not layer3_result.success:
                return Layer4Result(
                    layer3_result=layer3_result,
                    hypotheses=[],
                    verb_phrases=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 4 failed due to Layer 3 failure: {layer3_result.description}"
                )
            
            # Execute Layer 4 VP tokenization
            layer4_hypotheses = latn_tokenize_layer4(layer3_result.hypotheses)
            
            # Extract verb phrases
            verb_phrases = self.layer4_grounder.extract_verb_phrases(layer4_hypotheses) if self.layer4_grounder else []
            
            # Layer 4 does semantic grounding of verb phrases but does NOT execute actions
            # Action execution should be handled by a separate system that consumes Layer 4 output
            
            layer4_confidence = layer4_hypotheses[0].confidence if layer4_hypotheses else layer3_result.confidence
            overall_confidence = (layer3_result.confidence + layer4_confidence) / 2
            
            return Layer4Result(
                layer3_result=layer3_result,
                hypotheses=layer4_hypotheses,
                verb_phrases=verb_phrases,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 4 processed {len(verb_phrases)} verb phrases"
            )
            
        except Exception as e:
            return Layer4Result(
                layer3_result=layer3_result,
                hypotheses=[],
                verb_phrases=[],
                success=False,
                confidence=0.0,
                description=f"Layer 4 failed: {e}"
            )

    def execute_layer5(self, sentence: str, enable_semantic_grounding: bool = True,
                      return_all_matches: bool = True) -> Layer5Result:
        """Execute Layer 5: Sentence tokenization + execution (requires Layer 1-4).
        
        Args:
            sentence: Input sentence to process
            enable_semantic_grounding: Whether to perform semantic grounding/execution
            return_all_matches: Whether to return all matches
            
        Returns:
            Layer5Result with complete sentence parsing and execution results
        """
        try:
            # Execute Layer 4 first (which includes Layers 1-3)
            layer4_result = self.execute_layer4(sentence, enable_semantic_grounding, return_all_matches)
            
            if not layer4_result.success:
                return Layer5Result(
                    layer4_result=layer4_result,
                    hypotheses=[],
                    sentence_phrases=[],
                    grounding_results=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 5 failed due to Layer 4 failure: {layer4_result.description}"
                )
            
            # Execute Layer 5 sentence tokenization
            layer5_hypotheses = latn_tokenize_layer5(layer4_result.hypotheses)
            
            # Semantic grounding/execution (if enabled)
            if enable_semantic_grounding and self.layer5_grounder:
                grounded_hypotheses, grounding_results = self.layer5_grounder.multiply_hypotheses_with_grounding(
                    layer5_hypotheses, return_all_matches
                )
            else:
                # No grounding - keep original hypotheses
                grounded_hypotheses = layer5_hypotheses
                grounding_results = []
            
            # Extract sentence phrases from grounded hypotheses
            sentence_phrases = extract_sentence_phrases(grounded_hypotheses)
            
            layer5_confidence = grounded_hypotheses[0].confidence if grounded_hypotheses else layer4_result.confidence
            overall_confidence = (layer4_result.confidence + layer5_confidence) / 2
            
            description = f"Layer 5 processed {len(sentence_phrases)} sentences"
            if enable_semantic_grounding and grounding_results:
                executed_count = sum(1 for gr in grounding_results if gr.success)
                description += f" and executed {executed_count} command(s)"
            
            return Layer5Result(
                layer4_result=layer4_result,
                hypotheses=grounded_hypotheses,
                sentence_phrases=sentence_phrases,
                grounding_results=grounding_results,
                success=True,
                confidence=overall_confidence,
                description=description
            )
            
        except Exception as e:
            return Layer5Result(
                layer4_result=layer4_result if 'layer4_result' in locals() else None,
                hypotheses=[],
                sentence_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 5 failed: {e}"
            )

    @property
    def scene_model(self):
        """Compatibility property for tests that expect scene_model attribute."""
        return self.scene
    
    def update_scene_model(self, scene_model: Optional[SceneModel]):
        """Update the scene model and reinitialize grounders."""
        self.scene = scene_model
        self.layer2_grounder = Layer2SemanticGrounder(scene_model) if scene_model else None
        self.layer3_grounder = Layer3SemanticGrounder() if scene_model else None
        self.layer4_grounder = Layer4SemanticGrounder(scene_model) if scene_model else None
        self.layer5_grounder = Layer5SemanticGrounder(scene_model) if scene_model else None
    
    def get_layer_analysis(self, sentence: str, target_layer: int = 3) -> dict:
        """Get detailed analysis of layer execution for debugging/testing."""
        analysis = {
            'input': sentence,
            'target_layer': target_layer
        }
        
        # Execute Layer 1
        layer1_result = self.execute_layer1(sentence)
        analysis['layer1'] = {
            'success': layer1_result.success,
            'confidence': layer1_result.confidence,
            'hypothesis_count': len(layer1_result.hypotheses),
            'description': layer1_result.description
        }
        
        if target_layer >= 2:
            # Execute Layer 2
            layer2_result = self.execute_layer2(sentence, enable_semantic_grounding=bool(self.scene))
            analysis['layer2'] = {
                'success': layer2_result.success,
                'confidence': layer2_result.confidence,
                'noun_phrase_count': len(layer2_result.noun_phrases),
                'grounding_count': len(layer2_result.grounding_results),
                'description': layer2_result.description
            }
        
        if target_layer >= 3:
            # Execute Layer 3
            layer3_result = self.execute_layer3(sentence, enable_semantic_grounding=bool(self.scene))
            analysis['layer3'] = {
                'success': layer3_result.success,
                'confidence': layer3_result.confidence,
                'pp_count': len(layer3_result.prepositional_phrases),
                'description': layer3_result.description
            }
        
        return analysis


# Convenience functions for direct layer execution
def execute_layer1(sentence: str) -> Layer1Result:
    """Convenience function to execute Layer 1 only."""
    executor = LATNLayerExecutor()
    return executor.execute_layer1(sentence)


def execute_layer2(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_semantic_grounding: bool = True, return_all_matches: bool = False) -> Layer2Result:
    """Convenience function to execute Layer 2 (includes Layer 1)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer2(sentence, enable_semantic_grounding, return_all_matches)


def execute_layer3(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_semantic_grounding: bool = True, return_all_matches: bool = False) -> Layer3Result:
    """Convenience function to execute Layer 3 (includes Layers 1 & 2)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer3(sentence, enable_semantic_grounding, return_all_matches)


def execute_layer4(sentence: str, scene_model: Optional[SceneModel] = None, 
                  enable_semantic_grounding: bool = True) -> Layer4Result:
    """Convenience function to execute Layer 4 (includes Layers 1-3)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer4(sentence, enable_semantic_grounding=enable_semantic_grounding)
