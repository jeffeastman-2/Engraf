import numpy as np
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_np import run_np
from engraf.lexer.vector_space import VectorSpace

 
def test_simple_np():
    result = run_np(TokenStream(tokenize("the red cube")))
   
    assert result is not None
    assert result.noun == "cube"
    assert isinstance(result.vector, VectorSpace)

def test_np_with_adjectives():
    result = run_np(TokenStream(tokenize("a large blue sphere")))
   
    assert result is not None
    assert result.noun == "sphere"
    assert isinstance(result.vector, VectorSpace)
    assert result.vector["scaleY"] > 1.5  # Check if the height is scaled
    assert result.vector["blue"] > 0.5  # Check if the color is blue

def test_np_with_adverbs():
    result = run_np(TokenStream(tokenize("a very tall red cylinder")))
   
    assert result is not None
    print("Final vector:", result.vector)
    assert result.noun == "cylinder"
    assert isinstance(result.vector, VectorSpace)
    assert result.vector["scaleY"] == 2.25  # "very tall" = 1.5 * 1.5 = 2.25
    assert result.vector["red"] > 0.5  # Check if the color is red

def test_np_failure():
    result = run_np(TokenStream(tokenize("draw red cube")))
    assert result is None

def test_np_with_preposition():
    result = run_np(TokenStream(tokenize("a large blue sphere over the green cube")))
   
    assert result is not None
    assert result.noun == "sphere"
    assert isinstance(result.vector, VectorSpace)
    assert len(result.preps) == 1
    assert result.preps[0].preposition == "over"
    assert result.preps[0].noun_phrase.noun == "cube"

def test_np_with_pronoun():
    np = run_np(TokenStream(tokenize("it")))
    assert np is not None
    assert np.pronoun == "it"

def test_np_with_vector():
    np = run_np(TokenStream(tokenize("[5,6,7]")))
    assert np is not None
    assert np.noun == "vector"
    assert np.vector is not None
    v = np.vector
    assert v["locX"] == 5.0
    assert v["locY"] == 6.0
    assert v["locZ"] == 7.0


def test_np_with_explicit_vector():
    tokens = TokenStream(tokenize("a tall blue cube at [3.0, 4.0, 5.0]"))
    np = run_np(tokens)
    assert np is not None
    assert np.noun == "cube"
    assert isinstance(np.vector, VectorSpace)
    
    pp = np.preps[0]
    assert pp.preposition == "at"
    pp_np = pp.noun_phrase
    assert pp_np.vector["locX"] == 3.0
    assert pp_np.vector["locY"] == 4.0
    assert pp_np.vector["locZ"] == 5.0

    # Optional: check if the vector is untouched by the position spec
    v = np.vector
    assert v["scaleY"] == 1.5  # from "tall"
    assert v["blue"] == 1.0    # from "blue"