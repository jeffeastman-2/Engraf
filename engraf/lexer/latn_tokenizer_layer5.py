#!/usr/bin/env python3
"""
LATN Layer 5: Sentence Tokenization

This layer transforms Layer 4 output (VP, NP, PP tokens) into complete sentence structures.
It uses the existing sentence ATN to parse tokenized elements into SentencePhrase objects.

Layer 5 Input: VP, NP, PP tokens from Layer 4
Layer 5 Output: Sentence tokens containing SentencePhrase structures

Example transformations:
- "VP(create NP(red cube))" → "SENTENCE(VP(create NP(red cube)))"
- "VP(move NP(it)) PP(to [2,3,4])" → "SENTENCE(VP(move NP(it)) PP(to [2,3,4]))"
- "NP(cube) VP(is) ADJ(red)" → "SENTENCE(NP(cube) VP(is) ADJ(red))"
"""

from typing import List, Optional, Tuple
import copy

from engraf.lexer.hypothesis import TokenizationHypothesis
from engraf.atn.subnet_sentence import run_sentence
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.lexer.vector_space import VectorSpace


def latn_tokenize_layer5(layer4_hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
    """Layer 5: Transform VP/NP/PP tokens into sentence structures.
    
    Args:
        layer4_hypotheses: List of Layer 4 tokenization hypotheses with VP tokens
        
    Returns:
        List of Layer 5 hypotheses with sentence tokens
    """
    layer5_hypotheses = []
    
    for hypothesis in layer4_hypotheses:
        try:
            # Attempt sentence parsing using the reworked sentence ATN
            sentence_result = run_sentence(hypothesis.tokens)
            
            if sentence_result and isinstance(sentence_result, SentencePhrase):
                # Successfully parsed into a sentence
                sentence_token = create_sentence_token(sentence_result, hypothesis.tokens)
                
                # Create new hypothesis with sentence token
                new_hypothesis = TokenizationHypothesis(
                    tokens=[sentence_token],
                    confidence=hypothesis.confidence * 0.95,  # Slight confidence penalty for complexity
                    description=f"L5: Sentence parsed from {len(hypothesis.tokens)} tokens"
                )
                
                # Preserve any grounding information from the original tokens
                sentence_token._original_tokens = hypothesis.tokens
                sentence_token._sentence_phrase = sentence_result
                
                layer5_hypotheses.append(new_hypothesis)
            else:
                # Sentence parsing failed - keep original hypothesis as fallback
                fallback_hypothesis = copy.deepcopy(hypothesis)
                fallback_hypothesis.confidence *= 0.8  # Lower confidence for non-sentence
                fallback_hypothesis.description = f"L5: Sentence parsing failed, keeping {len(hypothesis.tokens)} tokens"
                layer5_hypotheses.append(fallback_hypothesis)
                
        except Exception as e:
            # Parsing failed - keep original hypothesis with reduced confidence
            fallback_hypothesis = copy.deepcopy(hypothesis)
            fallback_hypothesis.confidence *= 0.7  # Further penalty for parse error
            fallback_hypothesis.description = f"L5: Parse error ({e}), keeping original tokens"
            layer5_hypotheses.append(fallback_hypothesis)
    
    # Sort hypotheses by confidence (best first)
    layer5_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    return layer5_hypotheses


def create_sentence_token(sentence_phrase: SentencePhrase, original_tokens: List) -> VectorSpace:
    """Create a sentence token from a SentencePhrase.
    
    Args:
        sentence_phrase: The parsed SentencePhrase object
        original_tokens: The original tokens that formed this sentence
        
    Returns:
        VectorSpace token representing the sentence
    """
    # Create sentence vector by combining constituent vectors
    sentence_vector = VectorSpace()
    
    # Add sentence dimension
    sentence_vector['Sentence'] = 1.0
    
    # Combine vectors from subject and predicate
    if sentence_phrase.subject:
        if isinstance(sentence_phrase.subject, VectorSpace):
            sentence_vector = sentence_vector + (sentence_phrase.subject * 0.4)
    
    if sentence_phrase.predicate:
        if isinstance(sentence_phrase.predicate, VectorSpace):
            sentence_vector = sentence_vector + (sentence_phrase.predicate * 0.6)  # Predicate typically more important
    
    # Preserve important semantic features from original tokens
    for token in original_tokens:
        if isinstance(token, VectorSpace):
            # Preserve action/verb semantics
            if 'action' in token and token['action'] > 0:
                sentence_vector['action'] = max(sentence_vector['action'], token['action'] * 0.5)
            if 'verb' in token and token['verb'] > 0:
                sentence_vector['verb'] = max(sentence_vector['verb'], token['verb'] * 0.5)
    
    # Create readable word representation
    subject_text = _get_phrase_text(sentence_phrase.subject) if sentence_phrase.subject else ""
    predicate_text = _get_phrase_text(sentence_phrase.predicate) if sentence_phrase.predicate else ""
    
    if subject_text and predicate_text:
        sentence_word = f"SENTENCE({subject_text} {predicate_text})"
    elif predicate_text:
        sentence_word = f"SENTENCE({predicate_text})"
    else:
        sentence_word = f"SENTENCE({len(original_tokens)} tokens)"
    
    sentence_vector.word = sentence_word
    
    return sentence_vector


def _get_phrase_text(phrase) -> str:
    """Extract readable text from a phrase object."""
    if hasattr(phrase, 'word'):
        return phrase.word
    elif hasattr(phrase, '__str__'):
        return str(phrase)
    elif hasattr(phrase, '__repr__'):
        return repr(phrase)
    else:
        return type(phrase).__name__


def extract_sentence_phrases(layer5_hypotheses: List[TokenizationHypothesis]) -> List[SentencePhrase]:
    """Extract SentencePhrase objects from Layer 5 processing.
    
    Args:
        layer5_hypotheses: List of Layer 5 tokenization hypotheses
        
    Returns:
        List of SentencePhrase objects found in the hypotheses
    """
    sentence_phrases = []
    
    for hypothesis in layer5_hypotheses:
        for token in hypothesis.tokens:
            if hasattr(token, '_sentence_phrase') and token._sentence_phrase:
                sentence_phrases.append(token._sentence_phrase)
    
    return sentence_phrases
