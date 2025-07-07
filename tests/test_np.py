import numpy as np
from engraf.atn.np import build_np_atn
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_np import run_np
from engraf.lexer.vector_space import VectorSpace, vector_from_features, is_verb, is_adverb

 
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
    result = run_np(TokenStream(tokenize("a very tall red arch")))
   
    assert result is not None
    print("Final vector:", result.vector)
    assert result.noun == "arch"
    assert isinstance(result.vector, VectorSpace)
    assert result.vector["scaleY"] > 2.5  # Check if the height is scaled
    assert result.vector["red"] > 0.5  # Check if the color is red

def test_np_failure():
    result = run_np(TokenStream(tokenize("draw red cube")))
    assert result is None

def test_np_with_preposition():
    result = run_np(TokenStream(tokenize("a large blue sphere over the green cube")))
   
    assert result is not None
    assert result.noun == "sphere"
    assert isinstance(result.vector, VectorSpace)
    assert result.preps is not None
    assert len(result.preps) == 1
    assert result.preps[0]["prep"] == "over"
    assert result.preps[0]["object"] == "cube"

