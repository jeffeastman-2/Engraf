#!/usr/bin/env python3

import pytest
from engraf.lexer.token_stream import tokenize, TokenStream
from engraf.atn.sentence import build_sentence_atn
from engraf.atn.core import run_atn
from engraf.pos.sentence_phrase import SentencePhrase

def test_complex_sentence():
    """Test parsing of 'color a tall box green and move it to [7, 7, 7]'"""
    sentence = "color a tall box green and move it to [7, 7, 7]"
    
    # Tokenize the sentence
    tokens = tokenize(sentence)
    assert len(tokens) > 0, "Tokenization should produce tokens"
    
    # Parse the sentence
    ts = TokenStream(tokens)
    sent = SentencePhrase()
    start, end = build_sentence_atn(sent, ts)
    result = run_atn(start, end, ts, sent)
    
    # Assert successful parsing
    assert result is not None, f"Sentence '{sentence}' should parse successfully"
    assert hasattr(result, 'predicate'), "Result should have a predicate"
    
    # Check that we have conjunction (compound sentence)
    predicate = result.predicate
    assert predicate.__class__.__name__ == 'ConjunctionPhrase', f"Should be ConjunctionPhrase, got {predicate.__class__.__name__}"
    
    # Verify the structure contains both verbs
    assert hasattr(predicate, 'left'), "ConjunctionPhrase should have 'left' attribute"
    assert hasattr(predicate, 'right'), "ConjunctionPhrase should have 'right' attribute"
    
    # Check first verb phrase (color)
    left_vp = predicate.left
    assert hasattr(left_vp, 'verb'), "Left phrase should have verb"
    assert left_vp.verb == 'color', f"Left verb should be 'color', got '{left_vp.verb}'"
    assert 'green' in left_vp.adjective_complement, "Left VP should have 'green' adjective complement"
    
    # Check second verb phrase (move)
    right_vp = predicate.right
    assert hasattr(right_vp, 'verb'), "Right phrase should have verb"
    assert right_vp.verb == 'move', f"Right verb should be 'move', got '{right_vp.verb}'"
    
    # Check that it has a noun phrase with a PP (since "it to [7, 7, 7]" is parsed as NP with PP)
    assert hasattr(right_vp, 'noun_phrase'), "Right VP should have noun_phrase"
    assert right_vp.noun_phrase is not None, "Right VP noun_phrase should not be None"
    
    # The PP should be attached to the NP "it"
    it_np = right_vp.noun_phrase
    assert hasattr(it_np, 'preps'), "NP should have preps attribute"
    assert len(it_np.preps) > 0, "NP should have prepositional phrases"
    assert it_np.preps[0].preposition == 'to', "First PP should be 'to'"
    
    print("✅ Test passed! Complex sentence parsed correctly.")
