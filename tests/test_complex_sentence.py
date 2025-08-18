#!/usr/bin/env python3

import pytest
from engraf.lexer.latn_tokenizer import latn_tokenize_best as tokenize
from engraf.lexer.token_stream import TokenStream
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

def test_comparative_sentence():
    """Test parsing 'make them more transparent than the purple circle at [3, 3, 3]'"""
    sentence = "make them more transparent than the purple circle at [3, 3, 3]"
    
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
    
    # Check that this is an imperative sentence (no subject)
    assert result.subject is None, "Imperative sentence should have no subject"
    
    # Check the verb
    predicate = result.predicate
    assert predicate.verb == 'make', f"Verb should be 'make', got '{predicate.verb}'"
    
    # Check the noun phrase structure
    np = predicate.noun_phrase
    assert np is not None, "Should have a noun phrase"
    assert np.pronoun == 'them', "Should have pronoun 'them'"
    
    # Check that transparency was scaled by "more" (should be 3.0: 2.0 * 1.5)
    assert np.vector['transparency'] == 3.0, f"Transparency should be 3.0 (2.0 * 1.5), got {np.vector['transparency']}"
    
    # Check that there is a prepositional phrase with "than"
    assert len(np.preps) > 0, "Should have prepositional phrases"
    than_pp = np.preps[0]
    assert than_pp.preposition == 'than', f"First PP should be 'than', got '{than_pp.preposition}'"
    
    # Check the comparison target: "the purple circle at [3, 3, 3]"
    target_np = than_pp.noun_phrase
    assert target_np.noun == 'circle', f"Target noun should be 'circle', got '{target_np.noun}'"
    assert target_np.determiner == 'the', f"Target determiner should be 'the', got '{target_np.determiner}'"
    assert target_np.vector['red'] == 0.5, "Target should be purple (red=0.5)"
    assert target_np.vector['blue'] == 0.5, "Target should be purple (blue=0.5)"
    
    # Check that the target has a location prepositional phrase
    assert len(target_np.preps) > 0, "Target should have prepositional phrases"
    at_pp = target_np.preps[0]
    assert at_pp.preposition == 'at', f"Target PP should be 'at', got '{at_pp.preposition}'"
    
    # Check the vector coordinates [3, 3, 3]
    vector_np = at_pp.noun_phrase
    assert vector_np.vector['vector'] == 1.0, "Should be recognized as a vector"
    assert vector_np.vector['locX'] == 3.0, "X coordinate should be 3"
    assert vector_np.vector['locY'] == 3.0, "Y coordinate should be 3"
    assert vector_np.vector['locZ'] == 3.0, "Z coordinate should be 3"
    
    print("✅ Test passed! Comparative sentence parsed correctly.")

