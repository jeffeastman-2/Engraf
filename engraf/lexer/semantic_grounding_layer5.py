#!/usr/bin/env python3
"""
LATN Layer 5 Semantic Grounding

This module provides semantic grounding and execution capabilities for LATN Layer 5 
Sentence tokens. It takes well-formed sentences and executes them in the scene.
"""

from typing import List
from dataclasses import dataclass

from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.lexer.hypothesis import TokenizationHypothesis


@dataclass
class Layer5GroundingResult:
    """Result of Layer 5 semantic grounding operation."""
    success: bool
    confidence: float
    executed_actions: List[str]
    scene_changes: List[str]
    description: str = ""


class Layer5SemanticGrounder:
    """Semantic grounding and execution for LATN Layer 5 Sentence tokens.

    Layer 5 evaluates the hypotheses to reject those that are not well-formed
    or do not align with the current scene context.
    """

    def __init__(self, scene_model: SceneModel):
        self.scene_model = scene_model
        
    
    def ground_layer5(self, hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
        """Apply grounding to hypotheses following the pattern of other layers.
        
        Args:
            hypotheses: Input hypotheses to ground

        Returns:
            List of grounded hypotheses.
        """
        if not hypotheses:
            return []

        grounded_hypotheses = []
        
        for hypothesis in hypotheses:
            # Extract and verify sentences from this hypothesis. Reject
            # hypotheses with anything other than SPs or CPs of SPs
            sentence_phrases = []
            for token in hypothesis.tokens:
                sp = token._original_sp if hasattr(token, '_original_sp') else None
                if sp and isinstance(sp, SentencePhrase):
                    sentence_phrases.append(sp)
                elif sp and isinstance(sp, ConjunctionPhrase):
                    for part in sp.flatten():
                        if isinstance(part, SentencePhrase):
                            sentence_phrases.append(part)
                        else:
                            sentence_phrases = []
                            break # Stop processing if not a recognized phrase
                else:
                    sentence_phrases = []
                    break   # Stop processing if not a recognized phrase
            if sentence_phrases:
                grounded_hypotheses.append(hypothesis)
            else:
                # No sentences found - reject original hypothesis
                continue

        return grounded_hypotheses

    def extract_sentence_phrases(self, hypothesis: TokenizationHypothesis) -> List[SentencePhrase]:
        """Extract sentence phrases from a tokenization hypothesis.

        Args:
            hypothesis: The tokenization hypothesis to extract from.

        Returns:
            List of extracted SentencePhrase objects.
        """
        sentence_phrases = []
        for token in hypothesis.tokens:
            sp = token._original_sp if hasattr(token, '_original_sp') else None
            if sp and isinstance(sp, SentencePhrase):
                sentence_phrases.append(sp)
            elif sp and isinstance(sp, ConjunctionPhrase):
                for part in sp.flatten():
                    if isinstance(part, SentencePhrase):
                        sentence_phrases.append(part)
        return sentence_phrases