#!/usr/bin/env python3

import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.conjunction_phrase import ConjunctionPhrase

def test_complex_sentence():
    """Test parsing of 'color a tall box green and move it to [7, 7, 7]'"""
    sentence = "color a tall box green and move it to [7, 7, 7]"
    executor = LATNLayerExecutor()
    result = executor.execute_layer5(sentence)
    # Assert successful parsing
    assert result.success, f"Sentence '{sentence}' should parse successfully"
    assert len(result.hypotheses) > 0, "Should have at least one hypothesis"
    hypo_0 = result.hypotheses[0]
    assert len(hypo_0.tokens) == 1, "Hypothesis should 1 token"
    sent = hypo_0.tokens[0].phrase
    assert isinstance(sent, ConjunctionPhrase), "Token should be a ConjunctionPhrase"
    parts = [sp for sp in sent.flatten()]
    assert len(parts) == 2, "Should have two coordinated NPs"

def test_comparative_sentence():
    """Test parsing 'make them more transparent than the purple circle at [3, 3, 3]'"""
    sentence = "make them more transparent than the purple circle at [3, 3, 3]"
    executor = LATNLayerExecutor()
    result = executor.execute_layer5(sentence)
    # Assert successful parsing
    assert result.success, f"Sentence '{sentence}' should parse successfully"
    assert len(result.hypotheses) > 0, "Should have at least one hypothesis"
    hypo_0 = result.hypotheses[0]
    assert len(hypo_0.tokens) == 1, "Hypothesis should 1 token"
