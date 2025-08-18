import pytest
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.latn_tokenizer import latn_tokenize_best as tokenize
from engraf.atn.subnet_sentence import run_sentence
from engraf.pos.conjunction_phrase import ConjunctionPhrase


def test_adverb_adjective_in_coordinated_np():
    """Test parsing 'draw a tall red box and a small bright blue circle'"""
    sentence = run_sentence(TokenStream(tokenize('draw a tall red box and a small bright blue circle')))
    
    assert sentence is not None, "Failed to parse sentence with adverb-adjective sequence"
    assert sentence.predicate.verb == "draw", "Verb should be 'draw'"
    
    # Check that the noun phrase is a coordination
    np = sentence.predicate.noun_phrase
    assert isinstance(np, ConjunctionPhrase), "Noun phrase should be a ConjunctionPhrase"
    
    # Extract the two objects
    objects = list(np.flatten())
    assert len(objects) == 2, "Should have exactly 2 objects"
    
    # Check first object: "a tall red box"
    box = objects[0]
    assert box.noun == "box", "First object should be 'box'"
    assert box.determiner == "a", "First object should have determiner 'a'"
    assert box.vector["red"] == 1.0, "Box should be red"
    assert box.vector["scaleY"] == 1.5, "Box should be tall (scaleY = 1.5)"
    
    # Check second object: "a small bright blue circle"
    circle = objects[1]
    assert circle.noun == "circle", "Second object should be 'circle'"
    assert circle.determiner == "a", "Second object should have determiner 'a'"
    assert circle.vector["blue"] == 1.5, "Circle should be bright blue (blue value boosted by 'bright')"
    assert circle.vector["scaleX"] == -0.5, "Circle should be small (scaleX = -0.5)"
    assert circle.vector["scaleY"] == -0.5, "Circle should be small (scaleY = -0.5)"
    assert circle.vector["scaleZ"] == -0.5, "Circle should be small (scaleZ = -0.5)"


def test_simple_adverb_adjective_sequence():
    """Test parsing 'draw a very tall cylinder'"""
    sentence = run_sentence(TokenStream(tokenize('draw a very tall cylinder')))
    
    assert sentence is not None, "Failed to parse sentence with simple adverb-adjective sequence"
    assert sentence.predicate.verb == "draw", "Verb should be 'draw'"
    
    np = sentence.predicate.noun_phrase
    assert np.noun == "cylinder", "Object should be 'cylinder'"
    assert np.vector["scaleY"] > 2.0, "Cylinder should be very tall (scaleY boosted by 'very')"


def test_multiple_adverb_adjective_sequences():
    """Test parsing 'draw a very tall red box'"""
    sentence = run_sentence(TokenStream(tokenize('draw a very tall red box')))
    
    assert sentence is not None, "Failed to parse sentence with multiple adjectives"
    assert sentence.predicate.verb == "draw", "Verb should be 'draw'"
    
    np = sentence.predicate.noun_phrase
    assert np.noun == "box", "Object should be 'box'"
    assert np.vector["scaleY"] > 2.0, "Box should be very tall (scaleY boosted by 'very')"
    assert np.vector["red"] == 1.0, "Box should be red"
